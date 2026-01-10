"""
Benchmarking system for the remote denoiser.

Usage:
    python -m benchmark ./test_images/
    python -m benchmark ./test_images/ --output ./results --model TreeNetDenoise
"""

from benchmark.timing import TimingRecord, TimingContext
from benchmark.metrics import (
    ImageMetrics,
    PhaseStats,
    AggregateMetrics,
    MetricsCollector,
)
from benchmark.session import BenchmarkConfig, BenchmarkSession
from benchmark.profiler import TorchProfilerWrapper
from benchmark.instrumented_worker import InstrumentedWorker
from benchmark.report import ReportGenerator

__all__ = [
    "TimingRecord",
    "TimingContext",
    "ImageMetrics",
    "PhaseStats",
    "AggregateMetrics",
    "MetricsCollector",
    "BenchmarkConfig",
    "BenchmarkSession",
    "TorchProfilerWrapper",
    "InstrumentedWorker",
    "ReportGenerator",
]
