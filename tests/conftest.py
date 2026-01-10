import os
import sys
import time
import socket
import subprocess
import shutil
from contextlib import closing
from pathlib import Path

import pytest


def pytest_configure(config):
    """
    Hook that runs before test collection.
    Set DATA_DIR environment variable for all tests.
    """
    temp_dir = Path("pytest-data-tmp")

    # Clean up if it exists from a previous run
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Create the directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable BEFORE any test code imports modules
    os.environ["DATA_DIR"] = str(temp_dir)


def pytest_unconfigure(config):
    """
    Hook that runs after all tests complete.
    Clean up temporary data directory.
    """
    temp_dir = Path("pytest-data-tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait until a TCP port is accepting connections or timeout."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with closing(socket.create_connection((host, port), timeout=0.5)):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="session")
def api_server(tmp_path_factory):
    """
    Start the FastAPI app from `main:app` in a subprocess using uvicorn.

    Yields the base URL (e.g. http://127.0.0.1:8787) to run tests against.
    """
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8787"))

    # Launch uvicorn as a subprocess: python -m uvicorn main:app --host HOST --port PORT
    python = sys.executable
    cmd = [python, "-m", "uvicorn", "main:app", "--host", host, "--port", str(port)]

    env = os.environ.copy()

    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    started = _wait_for_port(host, port, timeout=15.0)
    if not started:
        # Capture some output for debugging
        try:
            out, err = proc.communicate(timeout=1.0)
        except Exception:
            proc.kill()
            out, err = b"", b""
        raise RuntimeError(f"Server failed to start (port {port} not open). stdout:\n{out.decode(errors='ignore')}\nstderr:\n{err.decode(errors='ignore')}")

    base_url = f"http://{host}:{port}"

    try:
        yield base_url
    finally:
        # Terminate the server subprocess
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
