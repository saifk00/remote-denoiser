"""
PyTorch profiler wrapper for generating flamegraphs and traces.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import torch
import torch.profiler

logger = logging.getLogger(__name__)


class TorchProfilerWrapper:
    """Wraps PyTorch profiler for inference flamegraphs.

    Generates Chrome Trace format output that can be viewed in:
    - Perfetto UI (https://ui.perfetto.dev/)
    - Chrome DevTools (chrome://tracing)

    Also exports stack traces for flamegraph generation.
    """

    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        self._profiler: torch.profiler.profile | None = None
        self._is_active = False
        self._step_count = 0

    @property
    def is_active(self) -> bool:
        """Check if profiler is currently active."""
        return self._is_active

    def _get_activities(self) -> list[torch.profiler.ProfilerActivity]:
        """Get profiler activities based on available hardware."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return activities

    def _trace_handler(self, prof: torch.profiler.profile) -> None:
        """Handle completed trace - export to various formats."""
        traces_dir = self.output_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        # Export Chrome Trace (viewable in Perfetto/Chrome)
        trace_path = traces_dir / "pytorch_trace.json"
        try:
            prof.export_chrome_trace(str(trace_path))
            logger.info(f"Exported Chrome trace to {trace_path}")
        except Exception as e:
            logger.warning(f"Failed to export Chrome trace: {e}")

        # Export stacks for flamegraph (CPU time)
        stacks_cpu_path = traces_dir / "cpu_stacks.txt"
        try:
            prof.export_stacks(str(stacks_cpu_path), "self_cpu_time_total")
            logger.info(f"Exported CPU stacks to {stacks_cpu_path}")
        except Exception as e:
            logger.warning(f"Failed to export CPU stacks: {e}")

        # Export CUDA stacks if available
        if torch.cuda.is_available():
            stacks_cuda_path = traces_dir / "cuda_stacks.txt"
            try:
                prof.export_stacks(str(stacks_cuda_path), "self_cuda_time_total")
                logger.info(f"Exported CUDA stacks to {stacks_cuda_path}")
            except Exception as e:
                logger.warning(f"Failed to export CUDA stacks: {e}")

        # Print summary table
        summary_path = traces_dir / "profiler_summary.txt"
        try:
            summary = prof.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                row_limit=50,
            )
            with open(summary_path, "w") as f:
                f.write(summary)
            logger.info(f"Exported profiler summary to {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to export profiler summary: {e}")

    @contextmanager
    def profile(self, wait: int = 1, warmup: int = 1, active: int = 10, repeat: int = 1) -> Generator[None, None, None]:
        """Context manager for profiling a block of code.

        Args:
            wait: Number of steps to wait before starting warmup
            warmup: Number of warmup steps (profiler overhead may affect these)
            active: Number of steps to actively profile
            repeat: Number of times to repeat the wait/warmup/active cycle
        """
        if not self.enabled:
            yield
            return

        schedule = torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )

        try:
            self._profiler = torch.profiler.profile(
                activities=self._get_activities(),
                schedule=schedule,
                on_trace_ready=self._trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
            self._profiler.__enter__()
            self._is_active = True
            self._step_count = 0
            yield
        finally:
            if self._profiler is not None:
                self._profiler.__exit__(None, None, None)
                self._is_active = False
                self._profiler = None

    @contextmanager
    def profile_simple(self) -> Generator[torch.profiler.profile | None, None, None]:
        """Simple profiling without scheduling - profiles everything.

        Use this when you want to profile a specific block without step-based scheduling.
        """
        if not self.enabled:
            yield None
            return

        try:
            self._profiler = torch.profiler.profile(
                activities=self._get_activities(),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
            self._profiler.__enter__()
            self._is_active = True
            yield self._profiler
        finally:
            if self._profiler is not None:
                self._profiler.__exit__(None, None, None)
                self._trace_handler(self._profiler)
                self._is_active = False
                self._profiler = None

    def step(self) -> None:
        """Signal the profiler that a step has completed.

        Call this after each image is processed when using schedule-based profiling.
        """
        if self._profiler is not None and self._is_active:
            self._profiler.step()
            self._step_count += 1


def generate_flamegraph_html(stacks_file: Path, output_file: Path, title: str = "Flamegraph") -> bool:
    """Generate an HTML flamegraph from stack traces.

    This creates a simple HTML page with instructions for viewing the flamegraph,
    since generating actual flamegraphs requires external tools like flamegraph.pl
    or speedscope.

    Args:
        stacks_file: Path to the stacks file exported by PyTorch profiler
        output_file: Path to write the HTML file
        title: Title for the flamegraph

    Returns:
        True if successful, False otherwise
    """
    if not stacks_file.exists():
        return False

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{ color: #333; }}
        .instructions {{
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        code {{
            background: #e0e0e0;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        a {{ color: #0066cc; }}
        .option {{
            margin: 15px 0;
            padding: 15px;
            border-left: 4px solid #0066cc;
            background: #f9f9f9;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="instructions">
        <h2>View the Flamegraph</h2>
        <p>The stack traces have been exported to: <code>{stacks_file.name}</code></p>

        <div class="option">
            <h3>Option 1: Speedscope (Recommended)</h3>
            <ol>
                <li>Go to <a href="https://www.speedscope.app/" target="_blank">speedscope.app</a></li>
                <li>Drag and drop the <code>{stacks_file.name}</code> file</li>
                <li>Explore the interactive flamegraph</li>
            </ol>
        </div>

        <div class="option">
            <h3>Option 2: Perfetto UI (for Chrome Trace)</h3>
            <ol>
                <li>Go to <a href="https://ui.perfetto.dev/" target="_blank">ui.perfetto.dev</a></li>
                <li>Click "Open trace file"</li>
                <li>Select <code>pytorch_trace.json</code> from the traces folder</li>
            </ol>
        </div>

        <div class="option">
            <h3>Option 3: Chrome DevTools</h3>
            <ol>
                <li>Open Chrome and go to <code>chrome://tracing</code></li>
                <li>Click "Load" and select <code>pytorch_trace.json</code></li>
            </ol>
        </div>
    </div>

    <h2>Files Generated</h2>
    <ul>
        <li><code>pytorch_trace.json</code> - Chrome Trace format (Perfetto/Chrome compatible)</li>
        <li><code>cpu_stacks.txt</code> - CPU time stack traces</li>
        <li><code>cuda_stacks.txt</code> - CUDA time stack traces (if GPU available)</li>
        <li><code>profiler_summary.txt</code> - Text summary of profiling results</li>
    </ul>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)

    return True
