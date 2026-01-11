"""
HTTP client for benchmarking the remote denoiser API.

Measures network latency (upload/download) vs server-side processing time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from benchmark.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class RequestTiming:
    """Timing breakdown for an HTTP request."""

    total_ms: float
    # For uploads: time to send the file
    # For downloads: time to receive the file
    transfer_ms: float


@dataclass
class ProcessingResult:
    """Result of processing a single image through the API."""

    file_id: str
    file_path: str
    file_size_bytes: int

    # Timing breakdown
    upload_ms: float
    job_creation_ms: float
    polling_ms: float  # Time spent waiting/polling for completion
    download_ms: float
    total_ms: float

    # Downloaded file info
    output_path: Path | None = None
    output_size_bytes: int = 0


class BenchmarkClient:
    """HTTP client for benchmarking the denoiser API.

    Measures timing for each phase:
    - Upload: POST /upload (network + server file write)
    - Job creation: POST /job (includes inference time!)
    - Download: GET /job/{id}/download (network + server file read)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 300.0,  # 5 minutes for large images
        collector: MetricsCollector | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.collector = collector
        self._client: httpx.Client | None = None

    def __enter__(self) -> BenchmarkClient:
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    def __exit__(self, *args) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'with' context manager.")
        return self._client

    def health_check(self) -> bool:
        """Check if the server is responding."""
        try:
            # Try to hit any endpoint - use a simple GET
            response = self.client.get("/")
            # FastAPI returns 404 for root but that's fine - server is up
            return response.status_code in (200, 404, 405)
        except httpx.RequestError as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def upload_file(self, file_path: Path) -> tuple[str, float]:
        """Upload a file and return (file_id, duration_ms)."""
        start = time.perf_counter_ns()

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            response = self.client.post("/upload", files=files)

        duration_ms = (time.perf_counter_ns() - start) / 1_000_000

        response.raise_for_status()
        data = response.json()
        file_id = data["file_id"]

        logger.debug(f"Uploaded {file_path.name} -> {file_id} in {duration_ms:.1f}ms")
        return file_id, duration_ms

    def create_job(self, file_ids: list[str], model: str = "TreeNetDenoise") -> float:
        """Create a processing job and return duration_ms.

        Note: This is SYNCHRONOUS on the server - it blocks until inference completes!
        So this timing includes the actual denoising work.
        """
        start = time.perf_counter_ns()

        response = self.client.post(
            "/job",
            json={"files": file_ids, "model": model},
        )

        duration_ms = (time.perf_counter_ns() - start) / 1_000_000

        response.raise_for_status()
        logger.debug(f"Job completed for {len(file_ids)} files in {duration_ms:.1f}ms")
        return duration_ms

    def download_file(self, file_id: str, output_path: Path) -> tuple[int, float]:
        """Download a processed file and return (size_bytes, duration_ms)."""
        start = time.perf_counter_ns()

        response = self.client.get(f"/job/{file_id}/download")

        duration_ms = (time.perf_counter_ns() - start) / 1_000_000

        response.raise_for_status()

        # Write to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        size_bytes = len(response.content)
        logger.debug(f"Downloaded {file_id} ({size_bytes} bytes) in {duration_ms:.1f}ms")
        return size_bytes, duration_ms

    def process_image(
        self,
        file_path: Path,
        output_dir: Path,
        model: str = "TreeNetDenoise",
    ) -> ProcessingResult:
        """Process a single image through the full API flow.

        Flow: upload -> create job (blocks for inference) -> download
        """
        file_size = file_path.stat().st_size
        total_start = time.perf_counter_ns()

        # 1. Upload
        file_id, upload_ms = self.upload_file(file_path)

        # Record timing if collector available
        if self.collector:
            with self.collector.time("upload", file=str(file_path), file_id=file_id):
                pass  # Already timed above, just record

        # 2. Create job (this blocks until inference completes!)
        job_ms = self.create_job([file_id], model)

        if self.collector:
            with self.collector.time("job_processing", file=str(file_path), file_id=file_id):
                pass

        # 3. Download result
        output_path = output_dir / f"{file_path.stem}_denoised.dng"
        output_size, download_ms = self.download_file(file_id, output_path)

        if self.collector:
            with self.collector.time("download", file=str(file_path), file_id=file_id):
                pass

        total_ms = (time.perf_counter_ns() - total_start) / 1_000_000

        return ProcessingResult(
            file_id=file_id,
            file_path=str(file_path),
            file_size_bytes=file_size,
            upload_ms=upload_ms,
            job_creation_ms=job_ms,
            polling_ms=0.0,  # No polling in current sync API
            download_ms=download_ms,
            total_ms=total_ms,
            output_path=output_path,
            output_size_bytes=output_size,
        )


def wait_for_server(base_url: str, timeout: float = 60.0, poll_interval: float = 0.5) -> bool:
    """Wait for the server to become available.

    Args:
        base_url: Server URL to check
        timeout: Maximum time to wait in seconds
        poll_interval: Time between checks in seconds

    Returns:
        True if server is available, False if timeout reached
    """
    import time

    start = time.time()
    with httpx.Client(base_url=base_url, timeout=5.0) as client:
        while time.time() - start < timeout:
            try:
                response = client.get("/")
                if response.status_code in (200, 404, 405):
                    return True
            except httpx.RequestError:
                pass
            time.sleep(poll_interval)

    return False
