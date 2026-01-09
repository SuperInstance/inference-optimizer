"""
Memory Usage Profiling

Benchmarks memory usage across different optimizations.
Measures both RAM and GPU memory consumption.
"""

import json
import tracemalloc
from typing import Dict, List
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


def profile_ram_usage(
    model_name: str,
    optimizer: InferenceOptimizer,
    optimization: str,
    num_inferences: int = 100
) -> Dict:
    """
    Profile RAM usage during inference.

    Args:
        model_name: Name of the SentenceTransformers model
        optimizer: InferenceOptimizer instance
        optimization: Type of optimization
        num_inferences: Number of inferences to profile

    Returns:
        Dictionary with memory statistics
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

    # Profile memory
    print(f"Profiling memory ({num_inferences} inferences)...")
    tracemalloc.start()

    # Warmup
    for _ in range(10):
        optimized_model.encode(["warmup"])

    # Peak memory during inference
    for _ in range(num_inferences):
        optimized_model.encode(["memory profiling test"])

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "current_mb": round(current / 1024 / 1024, 2),
        "peak_mb": round(peak / 1024 / 1024, 2),
        "num_inferences": num_inferences,
        "optimization": optimization,
        "success": True
    }

    print(f"  Current memory: {result['current_mb']:.1f}MB")
    print(f"  Peak memory: {result['peak_mb']:.1f}MB")
    print()

    return result


def profile_gpu_memory(
    model_name: str,
    optimizer: InferenceOptimizer,
    optimization: str,
    num_inferences: int = 100
) -> Dict:
    """
    Profile GPU memory usage during inference.

    Args:
        model_name: Name of the SentenceTransformers model
        optimizer: InferenceOptimizer instance
        optimization: Type of optimization
        num_inferences: Number of inferences to profile

    Returns:
        Dictionary with GPU memory statistics
    """
    try:
        import torch
    except ImportError:
        return {"success": False, "error": "PyTorch not available"}

    if not torch.cuda.is_available():
        return {"success": False, "error": "CUDA not available"}

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

    # Profile GPU memory
    print(f"Profiling GPU memory ({num_inferences} inferences)...")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Warmup
    for _ in range(10):
        optimized_model.encode(["warmup"])

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_inferences):
        optimized_model.encode(["gpu memory test"])

    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

    result = {
        "current_mb": round(current_memory_mb, 2),
        "peak_mb": round(peak_memory_mb, 2),
        "num_inferences": num_inferences,
        "optimization": optimization,
        "success": True
    }

    print(f"  Current GPU memory: {result['current_mb']:.1f}MB")
    print(f"  Peak GPU memory: {result['peak_mb']:.1f}MB")
    print()

    return result


def profile_memory_overhead(
    model_name: str,
    optimizer: InferenceOptimizer,
    optimization: str
) -> Dict:
    """
    Profile memory overhead of model conversion.

    Args:
        model_name: Name of the SentenceTransformers model
        optimizer: InferenceOptimizer instance
        optimization: Type of optimization

    Returns:
        Dictionary with conversion overhead statistics
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Profiling conversion overhead: {optimization}")

    # Profile memory during conversion
    tracemalloc.start()

    try:
        if optimization == "fp16":
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

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = {
            "current_mb": round(current / 1024 / 1024, 2),
            "peak_mb": round(peak / 1024 / 1024, 2),
            "optimization": optimization,
            "success": True
        }

        print(f"  Conversion peak memory: {result['peak_mb']:.1f}MB")
        print()

        return result

    except Exception as e:
        tracemalloc.stop()
        print(f"ERROR: {optimization} conversion failed - {e}")
        return {
            "optimization": optimization,
            "success": False,
            "error": str(e)
        }


def run_memory_profiles(
    model: str,
    optimizations: List[str],
    output_file: str = "/tmp/memory_usage_profiles.json"
) -> Dict:
    """Run comprehensive memory profiling"""
    results = {}
    optimizer = InferenceOptimizer()

    print(f"\n{'='*70}")
    print(f"MEMORY USAGE PROFILES")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    # RAM usage profiling
    print(f"\n{'='*70}")
    print("RAM USAGE PROFILING")
    print(f"{'='*70}\n")

    for optimization in optimizations:
        print(f"\n{'-'*70}")
        print(f"Optimization: {optimization}")
        print(f"{'-'*70}")

        try:
            stats = profile_ram_usage(
                model,
                optimizer,
                optimization,
                num_inferences=100
            )
            key = f"ram_{optimization}"
            results[key] = stats

        except Exception as e:
            print(f"ERROR: {optimization} failed - {e}")
            key = f"ram_{optimization}"
            results[key] = {"success": False, "error": str(e)}

    # GPU memory profiling (if available)
    print(f"\n{'='*70}")
    print("GPU MEMORY PROFILING")
    print(f"{'='*70}\n")

    for optimization in ["baseline", "fp16", "tensorrt_fp16"]:
        print(f"\n{'-'*70}")
        print(f"Optimization: {optimization}")
        print(f"{'-'*70}")

        try:
            stats = profile_gpu_memory(
                model,
                optimizer,
                optimization,
                num_inferences=100
            )
            key = f"gpu_{optimization}"
            results[key] = stats

            if not stats.get("success", False):
                print(f"Skipped: {stats.get('error', 'Unknown error')}\n")

        except Exception as e:
            print(f"ERROR: {optimization} failed - {e}\n")
            key = f"gpu_{optimization}"
            results[key] = {"success": False, "error": str(e)}

    # Conversion overhead profiling
    print(f"\n{'='*70}")
    print("CONVERSION OVERHEAD PROFILING")
    print(f"{'='*70}\n")

    conversion_opts = ["fp16", "int8", "tensorrt_fp16", "tensorrt_int8", "onnx"]

    for optimization in conversion_opts:
        print(f"\n{'-'*70}")
        print(f"Conversion: {optimization}")
        print(f"{'-'*70}")

        try:
            stats = profile_memory_overhead(model, optimizer, optimization)
            key = f"conversion_{optimization}"
            results[key] = stats

        except Exception as e:
            print(f"ERROR: {optimization} failed - {e}")
            key = f"conversion_{optimization}"
            results[key] = {"success": False, "error": str(e)}

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return results


def print_memory_comparison(results: Dict):
    """Print memory usage comparison"""
    print(f"\n{'='*80}")
    print("RAM USAGE COMPARISON")
    print(f"{'='*80}\n")

    # Extract RAM results
    ram_results = {
        k.replace("ram_", ""): v
        for k, v in results.items()
        if k.startswith("ram_") and v.get("success", False)
    }

    if ram_results:
        # Find baseline
        if "baseline" in ram_results:
            baseline_peak = ram_results["baseline"]["peak_mb"]
        else:
            baseline_peak = list(ram_results.values())[0]["peak_mb"]

        # Print table
        print(f"{'Optimization':<25} {'Peak (MB)':>15} {'Current (MB)':>15} {'Reduction':>12}")
        print("-" * 70)

        sorted_results = sorted(
            ram_results.items(),
            key=lambda x: x[1]["peak_mb"]
        )

        for opt_name, stats in sorted_results:
            peak = stats["peak_mb"]
            current = stats["current_mb"]
            reduction = baseline_peak / peak

            print(f"{opt_name:<25} {peak:>15.1f} {current:>15.1f} {reduction:>11.2f}×")

    # GPU memory comparison
    print(f"\n{'='*80}")
    print("GPU MEMORY COMPARISON")
    print(f"{'='*80}\n")

    gpu_results = {
        k.replace("gpu_", ""): v
        for k, v in results.items()
        if k.startswith("gpu_") and v.get("success", False)
    }

    if gpu_results:
        if "baseline" in gpu_results:
            baseline_peak = gpu_results["baseline"]["peak_mb"]
        else:
            baseline_peak = list(gpu_results.values())[0]["peak_mb"]

        print(f"{'Optimization':<25} {'Peak (MB)':>15} {'Current (MB)':>15} {'Reduction':>12}")
        print("-" * 70)

        sorted_results = sorted(
            gpu_results.items(),
            key=lambda x: x[1]["peak_mb"]
        )

        for opt_name, stats in sorted_results:
            peak = stats["peak_mb"]
            current = stats["current_mb"]
            reduction = baseline_peak / peak

            print(f"{opt_name:<25} {peak:>15.1f} {current:>15.1f} {reduction:>11.2f}×")

    # Conversion overhead
    print(f"\n{'='*80}")
    print("CONVERSION OVERHEAD")
    print(f"{'='*80}\n")

    conversion_results = {
        k.replace("conversion_", ""): v
        for k, v in results.items()
        if k.startswith("conversion_") and v.get("success", False)
    }

    if conversion_results:
        print(f"{'Conversion':<25} {'Peak Memory (MB)':>20}")
        print("-" * 48)

        sorted_results = sorted(
            conversion_results.items(),
            key=lambda x: x[1]["peak_mb"]
        )

        for conv_type, stats in sorted_results:
            peak = stats["peak_mb"]
            print(f"{conv_type:<25} {peak:>20.1f}")


def main():
    """Main profiling execution"""
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
    print("MEMORY USAGE PROFILING")
    print("="*80)
    print(f"\nModel: {model}")
    print("\nThis will profile memory usage across multiple optimizations:")
    print("  1. RAM usage during inference")
    print("  2. GPU memory usage (if CUDA available)")
    print("  3. Memory overhead during model conversion")
    print("\nPlease ensure minimal system load for accurate results.\n")

    try:
        results = run_memory_profiles(model, optimizations)
        print_memory_comparison(results)

    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: Profiling failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("ERROR: sentence-transformers not available")
        print("Install with: pip install sentence-transformers")
        exit(1)

    main()
