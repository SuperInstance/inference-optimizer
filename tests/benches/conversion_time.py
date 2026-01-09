"""
Model Conversion Time Benchmarks

Benchmarks the time required to convert models to different optimization formats.
Mealsures TensorRT, ONNX, and quantization conversion times.
"""

import time
import json
from pathlib import Path
from typing import Dict, List

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


def benchmark_conversion_time(
    model_name: str,
    optimizer: InferenceOptimizer,
    conversion_type: str
) -> float:
    """
    Benchmark model conversion time.

    Args:
        model_name: Name of the SentenceTransformers model
        optimizer: InferenceOptimizer instance
        conversion_type: Type of conversion (tensorrt_fp16, tensorrt_int8, onnx, fp16, int8)

    Returns:
        Conversion time in seconds
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Converting to: {conversion_type}")
    start_time = time.time()

    if conversion_type == "tensorrt_fp16":
        engine = optimizer.to_tensorrt(model, precision="fp16")
    elif conversion_type == "tensorrt_int8":
        calibration_data = ["calibration sentence"] * 100
        engine = optimizer.to_tensorrt(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
    elif conversion_type == "onnx":
        onnx_model = optimizer.to_onnx(model)
    elif conversion_type == "onnx_optimized":
        onnx_model = optimizer.to_onnx(model, optimization_level="all")
    elif conversion_type == "fp16":
        fp16_model = optimizer.quantize(model, precision="fp16")
    elif conversion_type == "int8":
        calibration_data = ["calibration sentence"] * 100
        int8_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
    elif conversion_type == "int4":
        if not optimizer.supports_int4:
            return None
        calibration_data = ["calibration sentence"] * 200
        int4_model = optimizer.quantize(
            model,
            precision="int4",
            calibration_data=calibration_data
        )
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")

    conversion_time = time.time() - start_time
    print(f"Conversion completed in {conversion_time:.2f}s\n")

    return conversion_time


def run_conversion_benchmarks(
    models: List[str],
    conversions: List[str],
    output_file: str = "/tmp/conversion_benchmarks.json"
) -> Dict:
    """
    Run conversion benchmarks across multiple models and conversion types.

    Args:
        models: List of model names to benchmark
        conversions: List of conversion types
        output_file: Path to save benchmark results

    Returns:
        Dictionary of benchmark results
    """
    results = {}
    optimizer = InferenceOptimizer()

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Benchmarking Model: {model_name}")
        print(f"{'='*60}\n")

        model_results = {}

        for conversion_type in conversions:
            try:
                conversion_time = benchmark_conversion_time(
                    model_name,
                    optimizer,
                    conversion_type
                )

                if conversion_time is not None:
                    model_results[conversion_type] = {
                        "conversion_time_seconds": round(conversion_time, 2),
                        "success": True
                    }
                else:
                    model_results[conversion_type] = {
                        "success": False,
                        "reason": "Unsupported on this system"
                    }

            except Exception as e:
                print(f"ERROR: {conversion_type} failed - {e}\n")
                model_results[conversion_type] = {
                    "success": False,
                    "error": str(e)
                }

        results[model_name] = model_results

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return results


def print_summary(results: Dict):
    """Print benchmark summary"""
    print(f"\n{'='*80}")
    print("CONVERSION TIME SUMMARY")
    print(f"{'='*80}\n")

    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 60)

        successful_conversions = {
            k: v for k, v in model_results.items()
            if v.get("success", False)
        }

        if successful_conversions:
            # Sort by conversion time
            sorted_conversions = sorted(
                successful_conversions.items(),
                key=lambda x: x[1]["conversion_time_seconds"]
            )

            for conv_type, data in sorted_conversions:
                time_str = f"{data['conversion_time_seconds']:>6.2f}s"
                print(f"  {conv_type:25s}: {time_str}")

        # Show failed conversions
        failed_conversions = [
            k for k, v in model_results.items()
            if not v.get("success", False)
        ]

        if failed_conversions:
            print("\n  Failed conversions:")
            for conv_type in failed_conversions:
                reason = model_results[conv_type].get("reason", "Unknown error")
                print(f"    - {conv_type}: {reason}")


def main():
    """Main benchmark execution"""
    # Models to benchmark
    models = [
        "all-MiniLM-L6-v2",        # Small (80MB, 384 dim)
        "all-mpnet-base-v2",       # Medium (420MB, 768 dim)
    ]

    # Conversion types to benchmark
    conversions = [
        "tensorrt_fp16",
        "tensorrt_int8",
        "onnx",
        "onnx_optimized",
        "fp16",
        "int8",
        "int4",  # May not be supported on all systems
    ]

    print("="*80)
    print("MODEL CONVERSION TIME BENCHMARKS")
    print("="*80)
    print("\nThis will benchmark conversion times for multiple models and formats.")
    print("Please ensure you have sufficient disk space and time.")
    print("\nPress Ctrl+C to interrupt at any time.\n")

    try:
        results = run_conversion_benchmarks(models, conversions)
        print_summary(results)

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
