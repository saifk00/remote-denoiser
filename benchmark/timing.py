"""
Core timing primitives for the benchmarking system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmark.metrics import MetricsCollector


@dataclass
class TimingRecord:
    """A single timing measurement."""

    name: str
    start_ns: int
    end_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        """Duration in nanoseconds."""
        return self.end_ns - self.start_ns

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_ns / 1_000_000

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.duration_ns / 1_000_000_000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class TimingContext:
    """Context manager for timing code blocks.

    Usage:
        with collector.time("inference", file="image.dng"):
            # code to time
            pass
    """

    def __init__(self, name: str, collector: MetricsCollector, **metadata: Any):
        self.name = name
        self.collector = collector
        self.metadata = metadata
        self._start_ns: int = 0
        self._record: TimingRecord | None = None

    def __enter__(self) -> TimingContext:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        end_ns = time.perf_counter_ns()
        self._record = TimingRecord(
            name=self.name,
            start_ns=self._start_ns,
            end_ns=end_ns,
            metadata=self.metadata,
        )
        self.collector.record(self._record)

    @property
    def record(self) -> TimingRecord | None:
        """Get the timing record after context exit."""
        return self._record
