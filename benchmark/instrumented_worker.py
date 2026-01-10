"""
Instrumented worker that wraps the standard Worker with timing hooks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from worker import Worker, ProcessConfig
from benchmark.metrics import MetricsCollector, ImageMetrics

if TYPE_CHECKING:
    from benchmark.profiler import TorchProfilerWrapper


class InstrumentedWorker:
    """Wraps Worker with comprehensive timing instrumentation.

    Measures each phase of image processing:
    - load_raw: Loading the RAW image file
    - preprocessing: Preparing data for inference (extracting ISO, etc.)
    - inference: Running the neural network
    - postprocessing: Saving the output file
    """

    def __init__(
        self,
        worker: Worker,
        collector: MetricsCollector,
        profiler: TorchProfilerWrapper | None = None,
    ):
        self._worker = worker
        self._collector = collector
        self._profiler = profiler

    @property
    def worker(self) -> Worker:
        """Access the underlying worker."""
        return self._worker

    def process(self, config: ProcessConfig, warmup: bool = False) -> ImageMetrics | None:
        """Process an image with full instrumentation.

        Args:
            config: Processing configuration
            warmup: If True, don't record metrics (warmup run)

        Returns:
            ImageMetrics for the processed image, or None if warmup
        """
        file_path = config.in_file
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        # Track individual phase timings
        load_raw_ms = 0.0
        preprocessing_ms = 0.0
        inference_ms = 0.0
        postprocessing_ms = 0.0

        with self._collector.time("total", file=file_path) as total_ctx:
            # Phase 1: Load raw image
            with self._collector.time("load_raw", file=file_path) as ctx:
                iso = self._worker.handler.load_rh(config.in_file)
            load_raw_ms = ctx.record.duration_ms if ctx.record else 0.0

            # Phase 2: Preprocessing (prepare conditioning)
            with self._collector.time("preprocessing", file=file_path) as ctx:
                conditioning = [iso, 0]
                inference_kwargs = {
                    "disable_tqdm": True,
                    "tile_size": config.tile_size,
                }
            preprocessing_ms = ctx.record.duration_ms if ctx.record else 0.0

            # Phase 3: Inference (the main computation)
            with self._collector.time("inference", file=file_path) as ctx:
                # If we have a profiler and it's in step mode, step it
                if self._profiler and self._profiler.is_active:
                    self._profiler.step()

                _, denoised_image = self._worker.handler.run_inference(
                    conditioning=conditioning,
                    dims=None,
                    inference_kwargs=inference_kwargs,
                )
            inference_ms = ctx.record.duration_ms if ctx.record else 0.0

            # Phase 4: Postprocessing (save result)
            with self._collector.time("postprocessing", file=file_path) as ctx:
                # Ensure output directory exists
                out_path = Path(config.out_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                self._worker.handler.handle_full_image(
                    denoised_image, config.out_file, config.cfa
                )
            postprocessing_ms = ctx.record.duration_ms if ctx.record else 0.0

        total_ms = total_ctx.record.duration_ms if total_ctx.record else 0.0

        if warmup:
            return None

        # Create and record image metrics
        metrics = ImageMetrics(
            file_path=file_path,
            file_size_bytes=file_size,
            load_raw_ms=load_raw_ms,
            preprocessing_ms=preprocessing_ms,
            inference_ms=inference_ms,
            postprocessing_ms=postprocessing_ms,
            total_ms=total_ms,
        )

        self._collector.add_image_metrics(metrics)
        return metrics
