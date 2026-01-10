"""
Report generation for benchmark results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.metrics import MetricsCollector, AggregateMetrics

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates benchmark reports and visualizations."""

    def __init__(
        self,
        output_dir: Path,
        collector: MetricsCollector,
        metrics: AggregateMetrics,
    ):
        self.output_dir = output_dir
        self.collector = collector
        self.metrics = metrics

    def generate_all(self) -> None:
        """Generate all report artifacts."""
        logger.info("Generating benchmark reports...")

        self.generate_summary_json()
        self.generate_timing_chart()
        self.generate_latency_histogram()
        self.generate_html_report()
        self.generate_flamegraph_instructions()

        logger.info("Report generation complete")

    def generate_summary_json(self) -> None:
        """Write aggregate metrics to summary.json."""
        path = self.output_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        logger.info(f"Wrote summary to {path}")

    def generate_timing_chart(self) -> None:
        """Create stacked bar chart showing time breakdown by phase."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            logger.warning("matplotlib not available, skipping timing chart")
            return

        image_metrics = self.collector.get_all_image_metrics()
        if not image_metrics:
            logger.warning("No image metrics available for timing chart")
            return

        # Prepare data
        labels = [Path(m.file_path).stem[:20] for m in image_metrics]
        load_raw = [m.load_raw_ms for m in image_metrics]
        preprocessing = [m.preprocessing_ms for m in image_metrics]
        inference = [m.inference_ms for m in image_metrics]
        postprocessing = [m.postprocessing_ms for m in image_metrics]

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(labels))
        width = 0.8

        ax.bar(x, load_raw, width, label='Load RAW', color='#3498db')
        ax.bar(x, preprocessing, width, bottom=load_raw, label='Preprocessing', color='#2ecc71')
        ax.bar(
            x, inference, width,
            bottom=[a + b for a, b in zip(load_raw, preprocessing)],
            label='Inference', color='#e74c3c'
        )
        ax.bar(
            x, postprocessing, width,
            bottom=[a + b + c for a, b, c in zip(load_raw, preprocessing, inference)],
            label='Postprocessing', color='#9b59b6'
        )

        ax.set_xlabel('Image')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Processing Time Breakdown by Image')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        chart_path = self.output_dir / "charts" / "timing_breakdown.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"Saved timing chart to {chart_path}")

    def generate_latency_histogram(self) -> None:
        """Create histogram of inference latencies."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            logger.warning("matplotlib not available, skipping latency histogram")
            return

        image_metrics = self.collector.get_all_image_metrics()
        if not image_metrics:
            return

        inference_times = [m.inference_ms for m in image_metrics]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(inference_times, bins=20, color='#3498db', edgecolor='white', alpha=0.7)
        ax.axvline(
            self.metrics.inference_stats.mean_ms,
            color='#e74c3c', linestyle='--', linewidth=2,
            label=f'Mean: {self.metrics.inference_stats.mean_ms:.1f}ms'
        )
        ax.axvline(
            self.metrics.inference_stats.p95_ms,
            color='#f39c12', linestyle='--', linewidth=2,
            label=f'P95: {self.metrics.inference_stats.p95_ms:.1f}ms'
        )

        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Inference Latency Distribution')
        ax.legend()

        plt.tight_layout()
        chart_path = self.output_dir / "charts" / "latency_distribution.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"Saved latency histogram to {chart_path}")

    def generate_phase_pie_chart(self) -> None:
        """Create pie chart showing time spent in each phase."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return

        phases = [
            ('Load RAW', self.metrics.load_raw_stats.total_ms, '#3498db'),
            ('Preprocessing', self.metrics.preprocessing_stats.total_ms, '#2ecc71'),
            ('Inference', self.metrics.inference_stats.total_ms, '#e74c3c'),
            ('Postprocessing', self.metrics.postprocessing_stats.total_ms, '#9b59b6'),
        ]

        labels = [p[0] for p in phases]
        sizes = [p[1] for p in phases]
        colors = [p[2] for p in phases]

        # Filter out zero values
        non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if not non_zero:
            return

        labels, sizes, colors = zip(*non_zero)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=[0.02] * len(sizes)
        )
        ax.set_title('Time Distribution by Phase')

        chart_path = self.output_dir / "charts" / "phase_distribution.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()

    def generate_flamegraph_instructions(self) -> None:
        """Generate HTML with instructions for viewing flamegraphs."""
        from benchmark.profiler import generate_flamegraph_html

        traces_dir = self.output_dir / "traces"
        flamegraph_dir = self.output_dir / "flamegraphs"

        # Check for CPU stacks
        cpu_stacks = traces_dir / "cpu_stacks.txt"
        if cpu_stacks.exists():
            generate_flamegraph_html(
                cpu_stacks,
                flamegraph_dir / "view_flamegraph.html",
                "CPU Flamegraph"
            )

    def generate_html_report(self) -> None:
        """Generate comprehensive HTML report."""
        image_metrics = self.collector.get_all_image_metrics()

        # Build per-image table rows
        image_rows = ""
        for m in image_metrics:
            image_rows += f"""
            <tr>
                <td>{Path(m.file_path).name}</td>
                <td>{m.file_size_bytes / 1024 / 1024:.1f}</td>
                <td>{m.load_raw_ms:.1f}</td>
                <td>{m.preprocessing_ms:.1f}</td>
                <td>{m.inference_ms:.1f}</td>
                <td>{m.postprocessing_ms:.1f}</td>
                <td><strong>{m.total_ms:.1f}</strong></td>
            </tr>
            """

        # Check which charts exist
        timing_chart_exists = (self.output_dir / "charts" / "timing_breakdown.png").exists()
        latency_chart_exists = (self.output_dir / "charts" / "latency_distribution.png").exists()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report - Remote Denoiser</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #e94560;
            --success: #2ecc71;
            --border: #333;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 40px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            color: var(--accent);
        }}

        h2 {{
            font-size: 1.5rem;
            margin: 30px 0 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--accent);
        }}

        h3 {{
            font-size: 1.2rem;
            margin: 20px 0 10px;
            color: var(--text-secondary);
        }}

        .subtitle {{
            color: var(--text-secondary);
            margin-bottom: 30px;
        }}

        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}

        .card-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent);
        }}

        .card-label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 5px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}

        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: var(--bg-card);
            font-weight: 600;
            color: var(--accent);
        }}

        tr:hover {{
            background: var(--bg-card);
        }}

        .chart-container {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }}

        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        .system-info {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }}

        .system-info dl {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 10px 20px;
        }}

        .system-info dt {{
            color: var(--text-secondary);
        }}

        .system-info dd {{
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        }}

        .links {{
            display: flex;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}

        .links a {{
            background: var(--bg-card);
            color: var(--text-primary);
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            transition: background 0.2s;
        }}

        .links a:hover {{
            background: var(--accent);
        }}

        .phase-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .phase-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
        }}

        .phase-card h4 {{
            color: var(--accent);
            margin-bottom: 15px;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid var(--border);
        }}

        .stat-row:last-child {{
            border-bottom: none;
        }}

        .stat-label {{
            color: var(--text-secondary);
        }}

        .stat-value {{
            font-family: 'SF Mono', Monaco, monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Benchmark Report</h1>
        <p class="subtitle">Remote Denoiser Performance Analysis</p>

        <h2>Summary</h2>
        <div class="cards">
            <div class="card">
                <div class="card-value">{self.metrics.total_images}</div>
                <div class="card-label">Images Processed</div>
            </div>
            <div class="card">
                <div class="card-value">{self.metrics.total_time_s:.2f}s</div>
                <div class="card-label">Total Processing Time</div>
            </div>
            <div class="card">
                <div class="card-value">{self.metrics.throughput_images_per_sec:.2f}</div>
                <div class="card-label">Images/Second</div>
            </div>
            <div class="card">
                <div class="card-value">{self.metrics.model_load_ms:.0f}ms</div>
                <div class="card-label">Model Load Time</div>
            </div>
        </div>

        <h2>Phase Statistics</h2>
        <div class="phase-stats">
            {self._render_phase_card("Load RAW", self.metrics.load_raw_stats)}
            {self._render_phase_card("Preprocessing", self.metrics.preprocessing_stats)}
            {self._render_phase_card("Inference", self.metrics.inference_stats)}
            {self._render_phase_card("Postprocessing", self.metrics.postprocessing_stats)}
        </div>

        {"<h2>Timing Breakdown</h2><div class='chart-container'><img src='charts/timing_breakdown.png' alt='Timing breakdown chart'></div>" if timing_chart_exists else ""}

        {"<h2>Latency Distribution</h2><div class='chart-container'><img src='charts/latency_distribution.png' alt='Latency distribution'></div>" if latency_chart_exists else ""}

        <h2>Per-Image Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Image</th>
                    <th>Size (MB)</th>
                    <th>Load (ms)</th>
                    <th>Preprocess (ms)</th>
                    <th>Inference (ms)</th>
                    <th>Postprocess (ms)</th>
                    <th>Total (ms)</th>
                </tr>
            </thead>
            <tbody>
                {image_rows}
            </tbody>
        </table>

        <h2>Trace Files</h2>
        <div class="links">
            <a href="traces/pytorch_trace.json" download>PyTorch Trace (Chrome Format)</a>
            <a href="traces/benchmark_timeline.json" download>Benchmark Timeline</a>
            <a href="traces/profiler_summary.txt">Profiler Summary</a>
            <a href="flamegraphs/view_flamegraph.html">View Flamegraph</a>
        </div>
        <p style="color: var(--text-secondary); margin-top: 10px;">
            Open trace files in <a href="https://ui.perfetto.dev/" style="color: var(--accent);">Perfetto UI</a>
            or <code>chrome://tracing</code>
        </p>

        <h2>System Information</h2>
        <div class="system-info">
            <dl>
                <dt>Model</dt>
                <dd>{self.metrics.system_info.model_name}</dd>
                <dt>Device</dt>
                <dd>{self.metrics.system_info.device}</dd>
                <dt>PyTorch Version</dt>
                <dd>{self.metrics.system_info.pytorch_version}</dd>
                <dt>CUDA Version</dt>
                <dd>{self.metrics.system_info.cuda_version or 'N/A'}</dd>
                <dt>GPU</dt>
                <dd>{self.metrics.system_info.cuda_device_name or 'N/A'}</dd>
                <dt>Tile Size</dt>
                <dd>{self.metrics.system_info.tile_size}px</dd>
                <dt>Python Version</dt>
                <dd>{self.metrics.system_info.python_version.split()[0]}</dd>
            </dl>
        </div>

        <h2>Raw Data</h2>
        <div class="links">
            <a href="summary.json" download>Summary JSON</a>
            <a href="raw_metrics.json" download>Raw Metrics JSON</a>
        </div>
    </div>
</body>
</html>
"""

        report_path = self.output_dir / "report.html"
        with open(report_path, "w") as f:
            f.write(html)
        logger.info(f"Generated HTML report at {report_path}")

    def _render_phase_card(self, name: str, stats) -> str:
        """Render a phase statistics card."""
        return f"""
        <div class="phase-card">
            <h4>{name}</h4>
            <div class="stat-row">
                <span class="stat-label">Mean</span>
                <span class="stat-value">{stats.mean_ms:.2f} ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Std Dev</span>
                <span class="stat-value">{stats.std_ms:.2f} ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Min</span>
                <span class="stat-value">{stats.min_ms:.2f} ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Max</span>
                <span class="stat-value">{stats.max_ms:.2f} ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">P50</span>
                <span class="stat-value">{stats.p50_ms:.2f} ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">P95</span>
                <span class="stat-value">{stats.p95_ms:.2f} ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">P99</span>
                <span class="stat-value">{stats.p99_ms:.2f} ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total</span>
                <span class="stat-value">{stats.total_ms:.2f} ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">% of Total</span>
                <span class="stat-value">{stats.percentage_of_total:.1f}%</span>
            </div>
        </div>
        """
