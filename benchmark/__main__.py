#!/usr/bin/env python3
"""
CLI entry point for the benchmarking system.

Usage:
    python -m benchmark ./test_images/
    python -m benchmark ./test_images/ --output ./results --model TreeNetDenoise
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from benchmark.session import BenchmarkConfig, BenchmarkSession


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the benchmark run."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def create_progress_callback():
    """Create a progress callback using rich if available, otherwise simple print."""
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        from rich.console import Console

        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )

        task_id = None
        started = False

        def callback(current: int, total: int, message: str) -> None:
            nonlocal task_id, started

            if not started:
                progress.start()
                task_id = progress.add_task(message, total=total)
                started = True
            else:
                progress.update(task_id, completed=current, description=message)

            if current >= total:
                progress.stop()

        return callback, lambda: progress.stop() if started else None

    except ImportError:
        # Fallback to simple print-based progress
        def callback(current: int, total: int, message: str) -> None:
            pct = (current / total * 100) if total > 0 else 0
            print(f"\r[{current}/{total}] {pct:.0f}% - {message}", end="", flush=True)
            if current >= total:
                print()  # Newline at end

        return callback, lambda: None


def main() -> int:
    """Main entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark the remote denoiser HTTP API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - starts server automatically and processes images
    python -m benchmark ./test_images/

    # Specify output directory
    python -m benchmark ./test_images/ -o ./benchmark_results

    # Use a different model
    python -m benchmark ./test_images/ -m TreeNetDenoiseSuperLight

    # Connect to an already-running server
    python -m benchmark ./test_images/ --no-start-server --server http://localhost:8000

    # Skip warmup for faster benchmarks
    python -m benchmark ./test_images/ --warmup 0
        """,
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing RAW images to process (.nef, .dng, .cr2, .arw, etc.)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for results (default: benchmark_YYYYMMDD_HHMMSS)",
    )

    parser.add_argument(
        "-m", "--model",
        default="TreeNetDenoise",
        choices=["TreeNetDenoise", "DeepSharpen", "TreeNetDenoiseSuperLight"],
        help="Model to use for denoising (default: TreeNetDenoise)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup images to skip in metrics (default: 2)",
    )

    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="Server URL to connect to (default: http://localhost:8000)",
    )

    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start server automatically - connect to existing server",
    )

    parser.add_argument(
        "--server-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds waiting for server to start (default: 120)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}", file=sys.stderr)
        return 1

    # Generate output directory name if not specified
    output_dir = args.output or Path(f"benchmark_{datetime.now():%Y%m%d_%H%M%S}")

    # Create configuration
    config = BenchmarkConfig(
        input_dir=args.input_dir,
        output_dir=output_dir,
        model=args.model,
        warmup_images=args.warmup,
        server_url=args.server,
        start_server=not args.no_start_server,
        server_startup_timeout=args.server_timeout,
    )

    # Print configuration
    print("=" * 60)
    print("Remote Denoiser Benchmark (HTTP API)")
    print("=" * 60)
    print(f"  Input directory:  {config.input_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Model:            {config.model}")
    print(f"  Warmup images:    {config.warmup_images}")
    print(f"  Server URL:       {config.server_url}")
    print(f"  Start server:     {config.start_server}")
    print("=" * 60)
    print()

    # Create progress callback
    progress_callback, cleanup = create_progress_callback()

    try:
        # Run benchmark
        session = BenchmarkSession(config, progress_callback=progress_callback)
        result = session.run()

        cleanup()

        # Print results
        print()
        print("=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)
        print(f"  Images processed: {result.images_processed}")
        print(f"  Warmup images:    {result.warmup_images}")
        print(f"  Wall clock time:  {result.wall_time_s:.2f}s")
        print(f"  Processing time:  {result.metrics.total_time_s:.2f}s")
        print(f"  Throughput:       {result.metrics.throughput_images_per_sec:.2f} images/sec")
        print()
        print("Phase breakdown (mean per image):")
        print(f"  Upload:          {result.metrics.upload_stats.mean_ms:>8.1f}ms ({result.metrics.upload_stats.percentage_of_total:>5.1f}%)")
        print(f"  Server processing:{result.metrics.job_processing_stats.mean_ms:>8.1f}ms ({result.metrics.job_processing_stats.percentage_of_total:>5.1f}%)")
        print(f"  Download:        {result.metrics.download_stats.mean_ms:>8.1f}ms ({result.metrics.download_stats.percentage_of_total:>5.1f}%)")
        print(f"  Network total:   {result.metrics.network_stats.mean_ms:>8.1f}ms ({result.metrics.network_stats.percentage_of_total:>5.1f}%)")
        print()
        print(f"Results saved to: {result.output_dir}")
        print(f"  - HTML Report:  {result.output_dir / 'report.html'}")
        print(f"  - Summary JSON: {result.output_dir / 'summary.json'}")
        print(f"  - Timeline:     {result.output_dir / 'traces' / 'benchmark_timeline.json'}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        cleanup()
        print("\nBenchmark interrupted by user")
        return 130

    except Exception as e:
        cleanup()
        logging.exception("Benchmark failed with error")
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
