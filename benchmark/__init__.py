"""
Benchmarking system for the remote denoiser.

This module provides HTTP-based benchmarking of the denoiser API,
measuring network latency (upload/download) vs server-side processing time.

Usage:
    python -m benchmark ./test_images/
    python -m benchmark ./test_images/ --output ./results --model TreeNetDenoise
    python -m benchmark ./test_images/ --no-start-server --server http://localhost:8000
"""

from benchmark.timing import TimingRecord, TimingContext
from benchmark.metrics import (
    ImageMetrics,
    PhaseStats,
    AggregateMetrics,
    MetricsCollector,
)
from benchmark.session import BenchmarkConfig, BenchmarkSession, BenchmarkResult
from benchmark.client import BenchmarkClient, ProcessingResult
from benchmark.report import ReportGenerator

__all__ = [
    # Timing primitives
    "TimingRecord",
    "TimingContext",
    # Metrics
    "ImageMetrics",
    "PhaseStats",
    "AggregateMetrics",
    "MetricsCollector",
    # Session management
    "BenchmarkConfig",
    "BenchmarkSession",
    "BenchmarkResult",
    # HTTP client
    "BenchmarkClient",
    "ProcessingResult",
    # Reporting
    "ReportGenerator",
]
