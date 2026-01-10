"""
Benchmark session orchestrator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from worker import Worker, ProcessConfig
from benchmark.metrics import MetricsCollector, AggregateMetrics, SystemInfo
from benchmark.profiler import TorchProfilerWrapper
from benchmark.instrumented_worker import InstrumentedWorker

logger = logging.getLogger(__name__)

# Supported RAW image extensions
RAW_EXTENSIONS = {".nef", ".dng", ".cr2", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    input_dir: Path
    output_dir: Path
    model: str = "TreeNetDenoise"
    tile_size: int = 256
    warmup_images: int = 2
    enable_torch_profiler: bool = True
    device: str | None = None  # Auto-detect if None

    # Profiler scheduling (for step-based profiling)
    profiler_wait: int = 0
    profiler_warmup: int = 1
    profiler_active: int = 5

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
    """Manages a complete benchmark run.

    Orchestrates:
    - Worker initialization with model loading
    - Image discovery
    - Warmup phase
    - Measured processing with profiling
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
        self._worker: Worker | None = None
        self._instrumented: InstrumentedWorker | None = None
        self._progress_callback = progress_callback

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def setup(self) -> None:
        """Initialize worker and create output directories."""
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

        self._report_progress(0, 1, "Loading model...")

        # Load model (timed)
        with self.collector.time("model_load") as ctx:
            self._worker = Worker(self.config.model, device=self.config.device)

        if ctx.record:
            self.collector.set_model_load_time(ctx.record.duration_ms)
            logger.info(f"Model loaded in {ctx.record.duration_ms:.2f}ms")

        self._instrumented = InstrumentedWorker(
            self._worker, self.collector, self.profiler
        )

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

        # Setup
        self.setup()
        images = self.discover_images()

        if not images:
            logger.warning("No images found to process!")
            system_info = SystemInfo.collect(
                self.config.model, self.config.tile_size, self.config.device
            )
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

        # Warmup phase
        logger.info(f"Running warmup with {warmup_count} images...")
        for i, img in enumerate(warmup_images):
            self._report_progress(
                i + 1, total_images, f"Warmup: {img.name}"
            )
            self._process_image(img, warmup=True)

        # Measured phase with profiler
        logger.info(f"Processing {len(measured_images)} images with profiling...")

        # Use simple profiling that captures everything
        with self.profiler.profile_simple():
            for i, img in enumerate(measured_images):
                self._report_progress(
                    warmup_count + i + 1,
                    total_images,
                    f"Processing: {img.name}",
                )
                self._process_image(img, warmup=False)

        end_time = datetime.now()
        logger.info(f"Benchmark completed at {end_time}")

        # Compute aggregate metrics
        system_info = SystemInfo.collect(
            self.config.model, self.config.tile_size, self.config.device
        )
        metrics = self.collector.compute_aggregate(system_info)

        # Generate reports
        self._report_progress(total_images, total_images, "Generating reports...")
        self._generate_reports(metrics)

        return BenchmarkResult(
            metrics=metrics,
            output_dir=self.config.output_dir,
            start_time=start_time,
            end_time=end_time,
            images_processed=len(measured_images),
            warmup_images=warmup_count,
        )

    def _process_image(self, path: Path, warmup: bool) -> None:
        """Process a single image with instrumentation."""
        out_path = self.config.output_dir / "processed" / f"{path.stem}_denoised.dng"

        config = ProcessConfig(
            in_file=str(path),
            out_file=str(out_path),
            tile_size=self.config.tile_size,
        )

        try:
            self._instrumented.process(config, warmup=warmup)
            if warmup:
                logger.debug(f"Warmup: processed {path.name}")
            else:
                logger.info(f"Processed {path.name}")
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            raise

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
