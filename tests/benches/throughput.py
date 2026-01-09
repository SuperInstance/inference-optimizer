"""
Throughput Benchmarks

Benchmarks inference throughput (samples/second) across optimizations.
Tests both single-threaded and batch inference throughput.
"""

import time
import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


def benchmark_single_threaded_throughput(
    model_name: str,
    optimizer: InferenceOptimizer,
    optimization: str,
    duration_seconds: int = 10
) -> Dict:
    """
    Benchmark single-threaded inference throughput.

    Args:
        model_name: Name of the SentenceTransformers model
        optimizer: InferenceOptimizer instance
        optimization: Type of optimization
        duration_seconds: Benchmark duration

    Returns:
        Dictionary with throughput statistics
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
    else:
        raise ValueError(f"Unknown optimization: {optimization}")

    # Warmup
    print("Warming up...")
    for _ in range(10):
        optimized_model.encode(["warmup"])

    # Benchmark throughput
    print(f"Benchmarking throughput ({duration_seconds}s)...")
    start = time.time()
    sample_count = 0

    while time.time() - start < duration_seconds:
        optimized_model.encode(["throughput test text"])
        sample_count += 1

    elapsed = time.time() - start
    throughput = sample_count / elapsed

    result = {
        "samples_per_second": round(throughput, 2),
        "total_samples": sample_count,
        "duration_seconds": round(elapsed, 2),
        "optimization": optimization,
        "success": True
    }

    print(f"  Throughput: {result['samples_per_second']:.1f} samples/sec")
    print(f"  Total samples: {result['total_samples']}")
    print()

    return result


def benchmark_batch_throughput(
    model_name: str,
    optimizer: InferenceOptimizer,
    optimization: str,
    batch_size: int,
    duration_seconds: int = 10
) -> Dict:
    """
    Benchmark batch inference throughput.

    Args:
        model_name: Name of the SentenceTransformers model
        optimizer: InferenceOptimizer instance
        optimization: Type of optimization
        batch_size: Batch size
        duration_seconds: Benchmark duration

    Returns:
        Dictionary with batch throughput statistics
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Applying optimization: {optimization}")

    # Prepare optimized model with batch support
    if optimization == "baseline":
        optimized_model = model
    elif optimization == "fp16":
        optimized_model = optimizer.quantize(model, precision="fp16")
    elif optimization == "tensorrt_fp16":
        optimized_model = optimizer.to_tensorrt(
            model,
            precision="fp16",
            max_batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown optimization: {optimization}")

    # Warmup
    print("Warming up...")
    batch_texts = ["batch warmup"] * batch_size
    for _ in range(10):
        optimized_model.encode(batch_texts)

    # Benchmark batch throughput
    print(f"Benchmarking batch {batch_size} throughput ({duration_seconds}s)...")
    start = time.time()
    sample_count = 0

    batch_texts = ["batch throughput test"] * batch_size

    while time.time() - start < duration_seconds:
        optimized_model.encode(batch_texts)
        sample_count += batch_size

    elapsed = time.time() - start
    throughput = sample_count / elapsed

    result = {
        "samples_per_second": round(throughput, 2),
        "total_samples": sample_count,
        "batch_size": batch_size,
        "duration_seconds": round(elapsed, 2),
        "optimization": optimization,
        "success": True
    }

    print(f"  Throughput: {result['samples_per_second']:.1f} samples/sec")
    print(f"  Total samples: {result['total_samples']}")
    print()

    return result


def benchmark_concurrent_throughput(
    model_name: str,
    optimizer: InferenceOptimizer,
    optimization: str,
    num_workers: int,
    duration_seconds: int = 10
) -> Dict:
    """
    Benchmark concurrent inference throughput with multiple workers.

    Args:
        model_name: Name of the SentenceTransformers model
        optimizer: InferenceOptimizer instance
        optimization: Type of optimization
        num_workers: Number of concurrent workers
        duration_seconds: Benchmark duration

    Returns:
        Dictionary with concurrent throughput statistics
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Applying optimization: {optimization}")

    # Prepare optimized model
    if optimization == "baseline":
        optimized_model = model
    elif optimization == "fp16":
        optimized_model = optimizer.quantize(model, precision="fp16")
    elif optimization == "tensorrt_fp16":
        optimized_model = optimizer.to_tensorrt(model, precision="fp16")
    else:
        raise ValueError(f"Unknown optimization: {optimization}")

    # Warmup
    print("Warming up...")
    for _ in range(10):
        optimized_model.encode(["warmup"])

    # Benchmark concurrent throughput
    print(f"Benchmarking with {num_workers} workers ({duration_seconds}s)...")

    sample_counts = []
    start_time = time.time()

    def worker_task():
        """Worker task for concurrent encoding"""
        count = 0
        while time.time() - start_time < duration_seconds:
            optimized_model.encode(["concurrent test"])
            count += 1
        return count

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task) for _ in range(num_workers)]

        for future in as_completed(futures):
            sample_counts.append(future.result())

    elapsed = time.time() - start_time
    total_samples = sum(sample_counts)
    throughput = total_samples / elapsed

    result = {
        "samples_per_second": round(throughput, 2),
        "total_samples": total_samples,
        "num_workers": num_workers,
        "duration_seconds": round(elapsed, 2),
        "optimization": optimization,
        "success": True
    }

    print(f"  Throughput: {result['samples_per_second']:.1f} samples/sec")
    print(f"  Total samples: {result['total_samples']}")
    print()

    return result


def run_throughput_benchmarks(
    model: str,
    optimizations: List[str],
    output_file: str = "/tmp/throughput_benchmarks.json"
) -> Dict:
    """Run comprehensive throughput benchmarks"""
    results = {}
    optimizer = InferenceOptimizer()

    print(f"\n{'='*70}")
    print(f"THROUGHPUT BENCHMARKS")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    # Single-threaded throughput
    print(f"\n{'='*70}")
    print("SINGLE-THREADED THROUGHPUT")
    print(f"{'='*70}\n")

    for optimization in optimizations:
        print(f"\n{'-'*70}")
        print(f"Optimization: {optimization}")
        print(f"{'-'*70}")

        try:
            stats = benchmark_single_threaded_throughput(
                model,
                optimizer,
                optimization,
                duration_seconds=5
            )
            key = f"single_threaded_{optimization}"
            results[key] = stats

        except Exception as e:
            print(f"ERROR: {optimization} failed - {e}")
            key = f"single_threaded_{optimization}"
            results[key] = {"success": False, "error": str(e)}

    # Batch throughput
    print(f"\n{'='*70}")
    print("BATCH THROUGHPUT")
    print(f"{'='*70}\n")

    for optimization in ["baseline", "tensorrt_fp16", "fp16"]:
        print(f"\n{'-'*70}")
        print(f"Optimization: {optimization}")
        print(f"{'-'*70}")

        for batch_size in [1, 4, 8, 16]:
            print(f"Batch size: {batch_size}")
            try:
                stats = benchmark_batch_throughput(
                    model,
                    optimizer,
                    optimization,
                    batch_size,
                    duration_seconds=3
                )
                key = f"batch_{optimization}_bs{batch_size}"
                results[key] = stats

            except Exception as e:
                print(f"ERROR: batch size {batch_size} failed - {e}")
                key = f"batch_{optimization}_bs{batch_size}"
                results[key] = {"success": False, "error": str(e)}

    # Concurrent throughput
    print(f"\n{'='*70}")
    print("CONCURRENT THROUGHPUT")
    print(f"{'='*70}\n")

    for optimization in ["baseline", "fp16", "tensorrt_fp16"]:
        print(f"\n{'-'*70}")
        print(f"Optimization: {optimization}")
        print(f"{'-'*70}")

        for num_workers in [1, 2, 4]:
            print(f"Workers: {num_workers}")
            try:
                stats = benchmark_concurrent_throughput(
                    model,
                    optimizer,
                    optimization,
                    num_workers,
                    duration_seconds=5
                )
                key = f"concurrent_{optimization}_w{num_workers}"
                results[key] = stats

            except Exception as e:
                print(f"ERROR: {num_workers} workers failed - {e}")
                key = f"concurrent_{optimization}_w{num_workers}"
                results[key] = {"success": False, "error": str(e)}

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return results


def print_throughput_comparison(results: Dict):
    """Print throughput comparison table"""
    print(f"\n{'='*80}")
    print("SINGLE-THREADED THROUGHPUT COMPARISON")
    print(f"{'='*80}\n")

    # Extract single-threaded results
    single_threaded = {
        k.replace("single_threaded_", ""): v
        for k, v in results.items()
        if k.startswith("single_threaded_") and v.get("success", False)
    }

    if single_threaded:
        # Find baseline
        if "baseline" in single_threaded:
            baseline_throughput = single_threaded["baseline"]["samples_per_second"]
        else:
            baseline_throughput = list(single_threaded.values())[0]["samples_per_second"]

        # Print table
        print(f"{'Optimization':<25} {'Samples/sec':>15} {'Speedup':>10}")
        print("-" * 55)

        sorted_results = sorted(
            single_threaded.items(),
            key=lambda x: x[1]["samples_per_second"],
            reverse=True
        )

        for opt_name, stats in sorted_results:
            throughput = stats["samples_per_second"]
            speedup = throughput / baseline_throughput
            print(f"{opt_name:<25} {throughput:>15.1f} {speedup:>10.2f}×")

    # Batch throughput comparison
    print(f"\n{'='*80}")
    print("BATCH THROUGHPUT COMPARISON")
    print(f"{'='*80}\n")

    batch_results = {
        k.replace("batch_", ""): v
        for k, v in results.items()
        if k.startswith("batch_") and v.get("success", False)
    }

    if batch_results:
        print(f"{'Optimization':<25} {'Batch':>8} {'Samples/sec':>15} {'Efficiency':>12}")
        print("-" * 65)

        # Group by optimization
        optimizations = set(k.rsplit("_bs", 1)[0] for k in batch_results.keys())

        for opt in sorted(optimizations):
            batch_sizes = []
            throughputs = []

            for bs in [1, 4, 8, 16]:
                key = f"{opt}_bs{bs}"
                if key in batch_results:
                    batch_sizes.append(bs)
                    throughputs.append(batch_results[key]["samples_per_second"])

            for bs, throughput in zip(batch_sizes, throughputs):
                # Calculate efficiency (throughput per batch size / batch 1 throughput)
                if throughputs[0] > 0:
                    efficiency = (throughput / bs) / (throughputs[0] / batch_sizes[0]) * 100
                else:
                    efficiency = 0

                print(f"{opt:<25} {bs:>8} {throughput:>15.1f} {efficiency:>11.0f}%")

    # Concurrent throughput comparison
    print(f"\n{'='*80}")
    print("CONCURRENT THROUGHPUT COMPARISON")
    print(f"{'='*80}\n")

    concurrent_results = {
        k.replace("concurrent_", ""): v
        for k, v in results.items()
        if k.startswith("concurrent_") and v.get("success", False)
    }

    if concurrent_results:
        print(f"{'Optimization':<25} {'Workers':>8} {'Samples/sec':>15} {'Scaling':>10}")
        print("-" * 63)

        # Group by optimization
        optimizations = set(k.rsplit("_w", 1)[0] for k in concurrent_results.keys())

        for opt in sorted(optimizations):
            workers_list = []
            throughputs = []

            for w in [1, 2, 4]:
                key = f"{opt}_w{w}"
                if key in concurrent_results:
                    workers_list.append(w)
                    throughputs.append(concurrent_results[key]["samples_per_second"])

            for workers, throughput in zip(workers_list, throughputs):
                # Calculate scaling efficiency (vs 1 worker)
                if throughputs[0] > 0:
                    scaling = (throughput / workers) / throughputs[0] * 100
                else:
                    scaling = 0

                print(f"{opt:<25} {workers:>8} {throughput:>15.1f} {scaling:>10.0f}%")


def main():
    """Main benchmark execution"""
    model = "all-MiniLM-L6-v2"

    optimizations = [
        "baseline",
        "fp16",
        "int8",
        "tensorrt_fp16",
        "tensorrt_int8",
        "onnx",
    ]

    print("="*80)
    print("THROUGHPUT BENCHMARKS")
    print("="*80)
    print(f"\nModel: {model}")
    print("\nThis will benchmark inference throughput across multiple scenarios:")
    print("  1. Single-threaded throughput")
    print("  2. Batch throughput (various batch sizes)")
    print("  3. Concurrent throughput (multiple workers)")
    print("\nThis may take several minutes...\n")

    try:
        results = run_throughput_benchmarks(model, optimizations)
        print_throughput_comparison(results)

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
