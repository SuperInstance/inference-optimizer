"""
Inference Latency Benchmarks

Benchmarks inference latency (time per inference) across different optimizations.
Measures mean, median, P95, P99 latencies with percentile breakdowns.
"""

import time
import json
import numpy as np
from typing import Dict, List
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


def benchmark_inference_latency(
    model_name: str,
    optimizer: InferenceOptimizer,
    optimization: str,
    iterations: int = 1000,
    warmup_iterations: int = 10
) -> Dict:
    """
    Benchmark inference latency with percentile breakdowns.

    Args:
        model_name: Name of the SentenceTransformers model
        optimizer: InferenceOptimizer instance
        optimization: Type of optimization (baseline, fp16, int8, tensorrt_fp16, etc.)
        iterations: Number of inference iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with latency statistics
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Applying optimization: {optimization}")

    # Prepare optimized model
    if optimization == "baseline":
        optimized_model = model
    elif optimization == "fp16":
        optimized_model = optimizer.quantize(model, precision="fp16")
    elif optimization == "int8":
        calibration_data = ["calibration sentence"] * 100
        optimized_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
    elif optimization == "tensorrt_fp16":
        optimized_model = optimizer.to_tensorrt(model, precision="fp16")
    elif optimization == "tensorrt_int8":
        calibration_data = ["calibration sentence"] * 100
        optimized_model = optimizer.to_tensorrt(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
    elif optimization == "onnx":
        optimized_model = optimizer.to_onnx(model)
    elif optimization == "onnx_optimized":
        optimized_model = optimizer.to_onnx(model, optimization_level="all")
    else:
        raise ValueError(f"Unknown optimization: {optimization}")

    # Warmup
    print(f"Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        optimized_model.encode(["warmup text"])

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()
        optimized_model.encode(["test text"])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    latencies = np.array(latencies)

    stats = {
        "mean_ms": round(float(np.mean(latencies)), 3),
        "median_ms": round(float(np.median(latencies)), 3),
        "std_ms": round(float(np.std(latencies)), 3),
        "min_ms": round(float(np.min(latencies)), 3),
        "max_ms": round(float(np.max(latencies)), 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p90_ms": round(float(np.percentile(latencies, 90)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "p999_ms": round(float(np.percentile(latencies, 99.9)), 3),
        "iterations": iterations,
        "optimization": optimization,
        "success": True
    }

    print(f"  Mean: {stats['mean_ms']:.3f}ms")
    print(f"  Median: {stats['median_ms']:.3f}ms")
    print(f"  P95: {stats['p95_ms']:.3f}ms")
    print(f"  P99: {stats['p99_ms']:.3f}ms")
    print()

    return stats


def run_latency_benchmarks(
    model: str,
    optimizations: List[str],
    iterations: int = 1000,
    output_file: str = "/tmp/inference_latency_benchmarks.json"
) -> Dict:
    """
    Run inference latency benchmarks across optimizations.

    Args:
        model: Model name to benchmark
        optimizations: List of optimization types
        iterations: Number of iterations per benchmark
        output_file: Path to save results

    Returns:
        Dictionary of benchmark results
    """
    results = {}
    optimizer = InferenceOptimizer()

    print(f"\n{'='*70}")
    print(f"INFERENCE LATENCY BENCHMARKS")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    for optimization in optimizations:
        print(f"\n{'-'*70}")
        print(f"Optimization: {optimization}")
        print(f"{'-'*70}")

        try:
            stats = benchmark_inference_latency(
                model,
                optimizer,
                optimization,
                iterations=iterations
            )
            results[optimization] = stats

        except Exception as e:
            print(f"ERROR: {optimization} failed - {e}")
            results[optimization] = {
                "success": False,
                "error": str(e)
            }

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return results


def print_latency_comparison(results: Dict):
    """Print latency comparison table"""
    print(f"\n{'='*80}")
    print("INFERENCE LATENCY COMPARISON")
    print(f"{'='*80}\n")

    # Filter successful results
    successful = {
        k: v for k, v in results.items()
        if v.get("success", False)
    }

    if not successful:
        print("No successful benchmarks to compare.")
        return

    # Find baseline
    if "baseline" in successful:
        baseline_mean = successful["baseline"]["mean_ms"]
    else:
        baseline_mean = list(successful.values())[0]["mean_ms"]

    # Print table header
    print(f"{'Optimization':<25} {'Mean':>10} {'Median':>10} {'P95':>10} {'P99':>10} {'Speedup':>10}")
    print("-" * 80)

    # Sort by mean latency
    sorted_results = sorted(
        successful.items(),
        key=lambda x: x[1]["mean_ms"]
    )

    for opt_name, stats in sorted_results:
        mean = stats["mean_ms"]
        median = stats["median_ms"]
        p95 = stats["p95_ms"]
        p99 = stats["p99_ms"]

        # Calculate speedup vs baseline
        speedup = baseline_mean / mean

        print(f"{opt_name:<25} {mean:>10.3f} {median:>10.3f} {p95:>10.3f} {p99:>10.3f} {speedup:>10.2f}×")

    # Print latency consistency analysis
    print(f"\n{'='*80}")
    print("LATENCY CONSISTENCY ANALYSIS")
    print(f"{'='*80}\n")

    print(f"{'Optimization':<25} {'P50':>10} {'P99':>10} {'P99/P50':>12} {'Consistency':>15}")
    print("-" * 80)

    for opt_name, stats in sorted_results:
        p50 = stats["p50_ms"]
        p99 = stats["p99_ms"]
        ratio = p99 / p50

        # Consistency rating
        if ratio < 2.0:
            consistency = "Excellent"
        elif ratio < 3.0:
            consistency = "Good"
        elif ratio < 5.0:
            consistency = "Fair"
        else:
            consistency = "Poor"

        print(f"{opt_name:<25} {p50:>10.3f} {p99:>10.3f} {ratio:>12.2f}× {consistency:>15}")


def main():
    """Main benchmark execution"""
    # Configuration
    model = "all-MiniLM-L6-v2"

    optimizations = [
        "baseline",
        "fp16",
        "int8",
        "tensorrt_fp16",
        "tensorrt_int8",
        "onnx",
        "onnx_optimized",
    ]

    print("="*80)
    print("INFERENCE LATENCY BENCHMARKS")
    print("="*80)
    print(f"\nModel: {model}")
    print(f"Iterations: 1000 per optimization")
    print("\nThis will benchmark inference latency across multiple optimizations.")
    print("Please ensure your system is under low load for accurate results.\n")

    try:
        results = run_latency_benchmarks(model, optimizations, iterations=1000)
        print_latency_comparison(results)

    except KeyboardInterrupt:
        print("\n\nBenchmarks interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: Benchmarks failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("ERROR: sentence-transformers not available")
        print("Install with: pip install sentence-transformers")
        exit(1)

    main()
