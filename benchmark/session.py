"""
Benchmark session orchestrator.

Manages the full benchmark lifecycle including optional server management.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from benchmark.metrics import MetricsCollector, AggregateMetrics, SystemInfo, ImageMetrics
from benchmark.profiler import TorchProfilerWrapper
from benchmark.client import BenchmarkClient, ProcessingResult, wait_for_server

logger = logging.getLogger(__name__)

# Supported RAW image extensions
RAW_EXTENSIONS = {".nef", ".dng", ".cr2", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    input_dir: Path
    output_dir: Path
    model: str = "TreeNetDenoise"
    warmup_images: int = 2

    # Server configuration
    server_url: str = "http://localhost:8000"
    start_server: bool = True  # If True, start server subprocess
    server_startup_timeout: float = 120.0  # Seconds to wait for server

    # Profiler (only works if running server in same process - disabled for HTTP mode)
    enable_torch_profiler: bool = False

    def __post_init__(self) -> None:
        """Convert paths to Path objects if needed."""
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    metrics: AggregateMetrics
    output_dir: Path
    start_time: datetime
    end_time: datetime
    images_processed: int
    warmup_images: int

    @property
    def wall_time_s(self) -> float:
        """Total wall clock time in seconds."""
        return (self.end_time - self.start_time).total_seconds()


class BenchmarkSession:
    """Manages a complete benchmark run via HTTP.

    Orchestrates:
    - Server startup (optional)
    - Image discovery
    - Warmup phase
    - Measured processing via HTTP API
    - Report generation
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ):
        """Initialize benchmark session.

        Args:
            config: Benchmark configuration
            progress_callback: Optional callback for progress updates.
                              Called with (current, total, message).
        """
        self.config = config
        self.collector = MetricsCollector()
        self.profiler = TorchProfilerWrapper(
            config.output_dir,
            enabled=config.enable_torch_profiler,
        )
        self._server_process: subprocess.Popen | None = None
        self._progress_callback = progress_callback
        self._processing_results: list[ProcessingResult] = []

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def _start_server(self) -> None:
        """Start the FastAPI server as a subprocess."""
        logger.info("Starting server subprocess...")
        self._report_progress(0, 1, "Starting server...")

        # Start uvicorn as subprocess
        self._server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.config.input_dir.parent if self.config.input_dir.is_absolute() else Path.cwd(),
        )

        # Wait for server to be ready
        logger.info(f"Waiting for server at {self.config.server_url}...")
        if not wait_for_server(self.config.server_url, timeout=self.config.server_startup_timeout):
            self._stop_server()
            raise RuntimeError(
                f"Server failed to start within {self.config.server_startup_timeout}s. "
                "Check that the server can start without errors."
            )

        logger.info("Server is ready")

    def _stop_server(self) -> None:
        """Stop the server subprocess if running."""
        if self._server_process:
            logger.info("Stopping server...")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait()
            self._server_process = None

    def setup(self) -> None:
        """Initialize output directories and optionally start server."""
        logger.info("Setting up benchmark session...")

        # Create output directory structure
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "traces").mkdir(exist_ok=True)
        (self.config.output_dir / "flamegraphs").mkdir(exist_ok=True)
        (self.config.output_dir / "charts").mkdir(exist_ok=True)
        (self.config.output_dir / "logs").mkdir(exist_ok=True)
        (self.config.output_dir / "processed").mkdir(exist_ok=True)

        # Set up logging to file
        log_path = self.config.output_dir / "logs" / "benchmark.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        # Start server if configured
        if self.config.start_server:
            self._start_server()

        logger.info("Benchmark session setup complete")

    def discover_images(self) -> list[Path]:
        """Find all processable images in input directory.

        Returns:
            Sorted list of image paths.
        """
        images = []
        for ext in RAW_EXTENSIONS:
            images.extend(self.config.input_dir.rglob(f"*{ext}"))
            images.extend(self.config.input_dir.rglob(f"*{ext.upper()}"))

        # Remove duplicates and sort
        images = sorted(set(images))
        logger.info(f"Discovered {len(images)} images in {self.config.input_dir}")
        return images

    def run(self) -> BenchmarkResult:
        """Execute the full benchmark.

        Returns:
            BenchmarkResult with aggregate metrics and metadata.
        """
        start_time = datetime.now()
        logger.info(f"Starting benchmark at {start_time}")

        try:
            # Setup
            self.setup()
            images = self.discover_images()

            if not images:
                logger.warning("No images found to process!")
                system_info = self._collect_system_info()
                return BenchmarkResult(
                    metrics=self.collector.compute_aggregate(system_info),
                    output_dir=self.config.output_dir,
                    start_time=start_time,
                    end_time=datetime.now(),
                    images_processed=0,
                    warmup_images=0,
                )

            # Determine warmup and measured images
            warmup_count = min(self.config.warmup_images, len(images))
            warmup_images = images[:warmup_count]
            measured_images = images[warmup_count:]

            total_images = len(images)
            output_dir = self.config.output_dir / "processed"

            # Process via HTTP client
            with BenchmarkClient(self.config.server_url, collector=self.collector) as client:
                # Verify server is reachable
                if not client.health_check():
                    raise RuntimeError(f"Cannot reach server at {self.config.server_url}")

                # Warmup phase
                logger.info(f"Running warmup with {warmup_count} images...")
                for i, img in enumerate(warmup_images):
                    self._report_progress(i + 1, total_images, f"Warmup: {img.name}")
                    try:
                        client.process_image(img, output_dir, self.config.model)
                        logger.debug(f"Warmup: processed {img.name}")
                    except Exception as e:
                        logger.warning(f"Warmup failed for {img.name}: {e}")

                # Measured phase
                logger.info(f"Processing {len(measured_images)} images...")
                for i, img in enumerate(measured_images):
                    self._report_progress(
                        warmup_count + i + 1,
                        total_images,
                        f"Processing: {img.name}",
                    )
                    try:
                        result = client.process_image(img, output_dir, self.config.model)
                        self._processing_results.append(result)
                        self._record_image_metrics(result)
                        logger.info(
                            f"Processed {img.name}: "
                            f"upload={result.upload_ms:.0f}ms, "
                            f"job={result.job_creation_ms:.0f}ms, "
                            f"download={result.download_ms:.0f}ms, "
                            f"total={result.total_ms:.0f}ms"
                        )
                    except Exception as e:
                        logger.error(f"Failed to process {img.name}: {e}")

            end_time = datetime.now()
            logger.info(f"Benchmark completed at {end_time}")

            # Compute aggregate metrics
            system_info = self._collect_system_info()
            metrics = self.collector.compute_aggregate(system_info)

            # Generate reports
            self._report_progress(total_images, total_images, "Generating reports...")
            self._generate_reports(metrics)

            return BenchmarkResult(
                metrics=metrics,
                output_dir=self.config.output_dir,
                start_time=start_time,
                end_time=end_time,
                images_processed=len(self._processing_results),
                warmup_images=warmup_count,
            )

        finally:
            # Always stop server if we started it
            if self.config.start_server:
                self._stop_server()

    def _record_image_metrics(self, result: ProcessingResult) -> None:
        """Convert ProcessingResult to ImageMetrics and record it."""
        metrics = ImageMetrics(
            file_path=result.file_path,
            file_size_bytes=result.file_size_bytes,
            # Map HTTP phases to our metrics structure
            # Note: "inference" here is the job processing which includes network overhead
            upload_ms=result.upload_ms,
            job_processing_ms=result.job_creation_ms,
            download_ms=result.download_ms,
            total_ms=result.total_ms,
        )
        self.collector.add_image_metrics(metrics)

    def _collect_system_info(self) -> SystemInfo:
        """Collect system information for the report."""
        # Import here to avoid issues if torch not available
        try:
            import torch
            pytorch_version = torch.__version__
            cuda_version = torch.version.cuda if torch.cuda.is_available() else None
            cuda_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_version = "unknown"
            cuda_version = None
            cuda_device = None
            device = "unknown"

        import sys
        return SystemInfo(
            device=device,
            model_name=self.config.model,
            pytorch_version=pytorch_version,
            cuda_version=cuda_version,
            cuda_device_name=cuda_device,
            python_version=sys.version,
            tile_size=256,  # Default, server doesn't expose this
        )

    def _generate_reports(self, metrics: AggregateMetrics) -> None:
        """Generate all output reports."""
        from benchmark.report import ReportGenerator

        generator = ReportGenerator(
            self.config.output_dir, self.collector, metrics
        )
        generator.generate_all()

        # Export raw data
        self.collector.export_json(self.config.output_dir / "raw_metrics.json")
        self.collector.export_chrome_trace(
            self.config.output_dir / "traces" / "benchmark_timeline.json"
        )

        logger.info(f"Reports generated in {self.config.output_dir}")
