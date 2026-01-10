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
        description="Benchmark the remote denoiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - process all images in a folder
    python -m benchmark ./test_images/

    # Specify output directory
    python -m benchmark ./test_images/ -o ./benchmark_results

    # Use a different model
    python -m benchmark ./test_images/ -m TreeNetDenoiseSuperLight

    # Adjust tile size for memory/speed tradeoff
    python -m benchmark ./test_images/ --tile-size 512

    # Skip profiler for faster benchmarks
    python -m benchmark ./test_images/ --no-profiler

    # Force CPU execution
    python -m benchmark ./test_images/ --device cpu
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
        "--tile-size",
        type=int,
        default=256,
        help="Tile size for inference in pixels (default: 256)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup images to skip in metrics (default: 2)",
    )

    parser.add_argument(
        "--no-profiler",
        action="store_true",
        help="Disable PyTorch profiler (faster but no flamegraph)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference: cuda, cuda:0, cpu, mps (default: auto-detect)",
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
        tile_size=args.tile_size,
        warmup_images=args.warmup,
        enable_torch_profiler=not args.no_profiler,
        device=args.device,
    )

    # Print configuration
    print("=" * 60)
    print("Remote Denoiser Benchmark")
    print("=" * 60)
    print(f"  Input directory:  {config.input_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Model:            {config.model}")
    print(f"  Tile size:        {config.tile_size}px")
    print(f"  Warmup images:    {config.warmup_images}")
    print(f"  Profiler enabled: {config.enable_torch_profiler}")
    print(f"  Device:           {config.device or 'auto'}")
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
        print(f"  Load RAW:      {result.metrics.load_raw_stats.mean_ms:>8.1f}ms ({result.metrics.load_raw_stats.percentage_of_total:>5.1f}%)")
        print(f"  Preprocessing: {result.metrics.preprocessing_stats.mean_ms:>8.1f}ms ({result.metrics.preprocessing_stats.percentage_of_total:>5.1f}%)")
        print(f"  Inference:     {result.metrics.inference_stats.mean_ms:>8.1f}ms ({result.metrics.inference_stats.percentage_of_total:>5.1f}%)")
        print(f"  Postprocessing:{result.metrics.postprocessing_stats.mean_ms:>8.1f}ms ({result.metrics.postprocessing_stats.percentage_of_total:>5.1f}%)")
        print()
        print(f"Results saved to: {result.output_dir}")
        print(f"  - HTML Report:  {result.output_dir / 'report.html'}")
        print(f"  - Summary JSON: {result.output_dir / 'summary.json'}")
        if config.enable_torch_profiler:
            print(f"  - Trace:        {result.output_dir / 'traces' / 'pytorch_trace.json'}")
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
