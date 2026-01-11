"""
Metrics collection and aggregation for benchmarking.
"""

from __future__ import annotations

import json
import statistics
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from benchmark.timing import TimingRecord, TimingContext


@dataclass
class PhaseStats:
    """Statistical summary for a single phase of processing."""

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    total_ms: float
    count: int
    percentage_of_total: float = 0.0

    @classmethod
    def from_durations(cls, name: str, durations_ms: list[float], total_benchmark_ms: float = 0.0) -> PhaseStats:
        """Create stats from a list of durations in milliseconds."""
        if not durations_ms:
            return cls(
                name=name,
                mean_ms=0.0,
                std_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                total_ms=0.0,
                count=0,
                percentage_of_total=0.0,
            )

        sorted_durations = sorted(durations_ms)
        total = sum(durations_ms)

        def percentile(data: list[float], p: float) -> float:
            """Calculate percentile value."""
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]

        return cls(
            name=name,
            mean_ms=statistics.mean(durations_ms),
            std_ms=statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0.0,
            min_ms=min(durations_ms),
            max_ms=max(durations_ms),
            p50_ms=percentile(sorted_durations, 50),
            p95_ms=percentile(sorted_durations, 95),
            p99_ms=percentile(sorted_durations, 99),
            total_ms=total,
            count=len(durations_ms),
            percentage_of_total=(total / total_benchmark_ms * 100) if total_benchmark_ms > 0 else 0.0,
        )


@dataclass
class ImageMetrics:
    """All metrics for a single processed image via HTTP API."""

    file_path: str
    file_size_bytes: int

    # Timing breakdown (all in milliseconds)
    # These reflect the HTTP-based flow:
    upload_ms: float           # POST /upload - send file to server
    job_processing_ms: float   # POST /job - server-side processing (includes inference!)
    download_ms: float         # GET /job/{id}/download - retrieve result
    total_ms: float            # Total end-to-end time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @property
    def network_ms(self) -> float:
        """Total network time (upload + download)."""
        return self.upload_ms + self.download_ms


@dataclass
class SystemInfo:
    """System information for benchmark context."""

    device: str
    model_name: str
    pytorch_version: str
    cuda_version: str | None
    cuda_device_name: str | None
    python_version: str
    tile_size: int

    @classmethod
    def collect(cls, model_name: str, tile_size: int, device: str | None = None) -> SystemInfo:
        """Collect current system information."""
        import sys
        import torch

        actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cuda_version = None
        cuda_device_name = None

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if actual_device.startswith("cuda"):
                device_idx = 0
                if ":" in actual_device:
                    device_idx = int(actual_device.split(":")[1])
                cuda_device_name = torch.cuda.get_device_name(device_idx)

        return cls(
            device=actual_device,
            model_name=model_name,
            pytorch_version=torch.__version__,
            cuda_version=cuda_version,
            cuda_device_name=cuda_device_name,
            python_version=sys.version,
            tile_size=tile_size,
        )


@dataclass
class AggregateMetrics:
    """Statistical summary across all processed images."""

    total_images: int
    total_time_s: float
    throughput_images_per_sec: float

    # Per-phase statistics (HTTP-based)
    upload_stats: PhaseStats        # Network: uploading files
    job_processing_stats: PhaseStats  # Server: inference + file I/O
    download_stats: PhaseStats      # Network: downloading results

    # Derived stats
    network_stats: PhaseStats       # Combined upload + download

    # System info
    system_info: SystemInfo

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_images": self.total_images,
            "total_time_s": self.total_time_s,
            "throughput_images_per_sec": self.throughput_images_per_sec,
            "phases": {
                "upload": asdict(self.upload_stats),
                "job_processing": asdict(self.job_processing_stats),
                "download": asdict(self.download_stats),
                "network_total": asdict(self.network_stats),
            },
            "system_info": asdict(self.system_info),
        }


class MetricsCollector:
    """Collects and aggregates timing metrics.

    Thread-safe for future parallel processing support.
    """

    def __init__(self) -> None:
        self._records: list[TimingRecord] = []
        self._image_metrics: list[ImageMetrics] = []
        self._lock = threading.Lock()

    def record(self, record: TimingRecord) -> None:
        """Thread-safe recording of timing data."""
        with self._lock:
            self._records.append(record)

    def time(self, name: str, **metadata: Any) -> TimingContext:
        """Create a timing context manager."""
        return TimingContext(name, self, **metadata)

    def add_image_metrics(self, metrics: ImageMetrics) -> None:
        """Add completed image metrics."""
        with self._lock:
            self._image_metrics.append(metrics)

    def get_records_for_file(self, file_path: str) -> list[TimingRecord]:
        """Get all timing records for a specific file."""
        with self._lock:
            return [r for r in self._records if r.metadata.get("file") == file_path]

    def get_duration_for_phase(self, file_path: str, phase: str) -> float:
        """Get duration in ms for a specific phase and file."""
        records = self.get_records_for_file(file_path)
        for r in records:
            if r.name == phase:
                return r.duration_ms
        return 0.0

    def compute_aggregate(self, system_info: SystemInfo) -> AggregateMetrics:
        """Compute aggregate statistics across all images."""
        with self._lock:
            if not self._image_metrics:
                # Return empty metrics
                empty_stats = PhaseStats.from_durations("empty", [])
                return AggregateMetrics(
                    total_images=0,
                    total_time_s=0.0,
                    throughput_images_per_sec=0.0,
                    upload_stats=empty_stats,
                    job_processing_stats=empty_stats,
                    download_stats=empty_stats,
                    network_stats=empty_stats,
                    system_info=system_info,
                )

            # Collect durations by phase
            upload_durations = [m.upload_ms for m in self._image_metrics]
            job_durations = [m.job_processing_ms for m in self._image_metrics]
            download_durations = [m.download_ms for m in self._image_metrics]
            network_durations = [m.network_ms for m in self._image_metrics]

            total_processing_ms = sum(m.total_ms for m in self._image_metrics)
            total_time_s = total_processing_ms / 1000

            return AggregateMetrics(
                total_images=len(self._image_metrics),
                total_time_s=total_time_s,
                throughput_images_per_sec=len(self._image_metrics) / total_time_s if total_time_s > 0 else 0.0,
                upload_stats=PhaseStats.from_durations("upload", upload_durations, total_processing_ms),
                job_processing_stats=PhaseStats.from_durations("job_processing", job_durations, total_processing_ms),
                download_stats=PhaseStats.from_durations("download", download_durations, total_processing_ms),
                network_stats=PhaseStats.from_durations("network", network_durations, total_processing_ms),
                system_info=system_info,
            )

    def get_all_image_metrics(self) -> list[ImageMetrics]:
        """Get all collected image metrics."""
        with self._lock:
            return list(self._image_metrics)

    def get_all_records(self) -> list[TimingRecord]:
        """Get all timing records."""
        with self._lock:
            return list(self._records)

    def export_json(self, path: Path) -> None:
        """Export all raw metrics to JSON."""
        with self._lock:
            data = {
                "records": [r.to_dict() for r in self._records],
                "image_metrics": [m.to_dict() for m in self._image_metrics],
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_chrome_trace(self, path: Path) -> None:
        """Export timeline as Chrome Trace format for Perfetto/Chrome DevTools.

        Format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
        """
        with self._lock:
            events = []

            # Build events from image metrics for a cleaner timeline
            current_time_us = 0
            for m in self._image_metrics:
                file_name = Path(m.file_path).name

                # Upload event
                events.append({
                    "name": "upload",
                    "cat": "network",
                    "ph": "X",
                    "ts": current_time_us,
                    "dur": m.upload_ms * 1000,
                    "pid": 1,
                    "tid": 1,
                    "args": {"file": file_name, "bytes": m.file_size_bytes},
                })
                current_time_us += m.upload_ms * 1000

                # Job processing event
                events.append({
                    "name": "job_processing",
                    "cat": "server",
                    "ph": "X",
                    "ts": current_time_us,
                    "dur": m.job_processing_ms * 1000,
                    "pid": 1,
                    "tid": 1,
                    "args": {"file": file_name},
                })
                current_time_us += m.job_processing_ms * 1000

                # Download event
                events.append({
                    "name": "download",
                    "cat": "network",
                    "ph": "X",
                    "ts": current_time_us,
                    "dur": m.download_ms * 1000,
                    "pid": 1,
                    "tid": 1,
                    "args": {"file": file_name},
                })
                current_time_us += m.download_ms * 1000

            trace_data = {
                "traceEvents": events,
                "displayTimeUnit": "ms",
                "metadata": {
                    "benchmark": "remote-denoiser",
                },
            }

        with open(path, "w") as f:
            json.dump(trace_data, f, indent=2)
