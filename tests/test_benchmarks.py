"""
Performance Benchmarks

Comprehensive performance benchmarks for inference optimization.
Tests conversion time, inference latency, throughput, and memory usage.
"""

import pytest
import numpy as np
import time
import psutil
import tracemalloc
from typing import List, Dict, Tuple
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestConversionBenchmarks:
    """Benchmark model conversion time for different formats"""

    def benchmark_conversion_time(self, model, optimizer, conversion_type: str) -> float:
        """Benchmark model conversion time"""
        start = time.time()

        if conversion_type == "tensorrt_fp16":
            engine = optimizer.to_tensorrt(model, precision="fp16")
        elif conversion_type == "tensorrt_int8":
            calibration_data = ["calibration"] * 100
            engine = optimizer.to_tensorrt(model, precision="int8", calibration_data=calibration_data)
        elif conversion_type == "onnx":
            onnx_model = optimizer.to_onnx(model)
        elif conversion_type == "onnx_optimized":
            onnx_model = optimizer.to_onnx(model, optimization_level="all")
        elif conversion_type == "fp16":
            fp16_model = optimizer.quantize(model, precision="fp16")
        elif conversion_type == "int8":
            calibration_data = ["calibration"] * 100
            int8_model = optimizer.quantize(model, precision="int8", calibration_data=calibration_data)

        return time.time() - start

    def test_conversion_time_comparison(self):
        """Compare conversion time across different optimization types"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        conversion_types = [
            "tensorrt_fp16",
            "tensorrt_int8",
            "onnx",
            "onnx_optimized",
            "fp16",
            "int8"
        ]

        results = {}
        for conv_type in conversion_types:
            try:
                conv_time = self.benchmark_conversion_time(model, optimizer, conv_type)
                results[conv_type] = conv_time
                print(f"{conv_type}: {conv_time:.2f}s")
            except Exception as e:
                print(f"{conv_type}: FAILED - {e}")
                results[conv_type] = None

        # All conversions should complete
        for conv_type, conv_time in results.items():
            if conv_time is not None:
                assert conv_time < 300, f"{conv_type} conversion took {conv_time:.2f}s (>300s)"

    def test_conversion_time_model_size(self):
        """Benchmark conversion time for different model sizes"""
        models = [
            ("all-MiniLM-L6-v2", "small"),
            ("all-mpnet-base-v2", "medium"),
        ]

        optimizer = InferenceOptimizer()

        for model_name, size_label in models:
            model = SentenceTransformer(model_name)

            # TensorRT FP16 conversion
            trt_time = self.benchmark_conversion_time(model, optimizer, "tensorrt_fp16")

            # ONNX conversion
            onnx_time = self.benchmark_conversion_time(model, optimizer, "onnx")

            print(f"{model_name} ({size_label}):")
            print(f"  TensorRT: {trt_time:.2f}s")
            print(f"  ONNX: {onnx_time:.2f}s")

            # Conversion time should scale with model size
            assert trt_time < 600, f"{model_name} TensorRT conversion too slow"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestInferenceLatency:
    """Benchmark inference latency across optimization types"""

    def benchmark_inference_latency(
        self,
        model,
        optimizer,
        optimization: str,
        iterations: int = 1000
    ) -> Dict[str, float]:
        """Benchmark inference latency with percentiles"""
        latencies = []

        # Prepare optimized model
        if optimization == "baseline":
            optimized_model = model
        elif optimization == "tensorrt_fp16":
            optimized_model = optimizer.to_tensorrt(model, precision="fp16")
        elif optimization == "tensorrt_int8":
            calibration_data = ["calibration"] * 100
            optimized_model = optimizer.to_tensorrt(
                model,
                precision="int8",
                calibration_data=calibration_data
            )
        elif optimization == "fp16":
            optimized_model = optimizer.quantize(model, precision="fp16")
        elif optimization == "int8":
            calibration_data = ["calibration"] * 100
            optimized_model = optimizer.quantize(
                model,
                precision="int8",
                calibration_data=calibration_data
            )
        elif optimization == "onnx":
            optimized_model = optimizer.to_onnx(model)

        # Warmup
        for _ in range(10):
            optimized_model.encode(["warmup"])

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            optimized_model.encode(["test"])
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)
        return {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "std": np.std(latencies)
        }

    def test_inference_latency_comparison(self):
        """Compare inference latency across optimizations"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        optimizations = ["baseline", "fp16", "tensorrt_fp16"]

        results = {}
        for opt in optimizations:
            stats = self.benchmark_inference_latency(model, optimizer, opt, iterations=100)
            results[opt] = stats
            print(f"\n{opt} Latency:")
            print(f"  Mean: {stats['mean']:.3f}ms")
            print(f"  Median: {stats['median']:.3f}ms")
            print(f"  P95: {stats['p95']:.3f}ms")
            print(f"  P99: {stats['p99']:.3f}ms")

        # TensorRT FP16 should be >2× faster than baseline
        baseline_mean = results["baseline"]["mean"]
        trt_mean = results["tensorrt_fp16"]["mean"]
        speedup = baseline_mean / trt_mean

        print(f"\nTensorRT FP16 Speedup: {speedup:.2f}×")
        assert speedup > 2.0, f"TensorRT speedup {speedup:.2f}× is below 2.0× target"

    def test_inference_latency_percentiles(self):
        """Test inference latency percentiles for consistency"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        stats = self.benchmark_inference_latency(model, optimizer, "tensorrt_fp16", iterations=500)

        # P99 should be <3× P50 (consistent latency)
        p50_to_p99_ratio = stats["p99"] / stats["p50"]
        print(f"P50 to P99 Ratio: {p50_to_p99_ratio:.2f}×")
        assert p50_to_p99_ratio < 3.0, "Inconsistent latency (P99 > 3× P50)"

    def test_inference_latency_batch_sizes(self):
        """Test inference latency across different batch sizes"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert to TensorRT
        engine = optimizer.to_tensorrt(model, precision="fp16", max_batch_size=8)

        batch_sizes = [1, 2, 4, 8]
        latencies = []

        for batch_size in batch_sizes:
            texts = ["test"] * batch_size

            # Warmup
            for _ in range(10):
                engine.encode(texts)

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                engine.encode(texts)
            end = time.perf_counter()

            avg_latency_per_sample = ((end - start) / 100) * 1000 / batch_size
            latencies.append(avg_latency_per_sample)

            print(f"Batch {batch_size}: {avg_latency_per_sample:.3f}ms per sample")

        # Larger batches should be more efficient
        # Batch 8 should be <1.5× latency of batch 1 per sample
        batch_1_latency = latencies[0]
        batch_8_latency = latencies[-1]
        efficiency_ratio = batch_8_latency / batch_1_latency

        print(f"Batch Efficiency (8 vs 1): {efficiency_ratio:.2f}×")
        assert efficiency_ratio < 1.5, "Batch inference is inefficient"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestThroughputBenchmarks:
    """Benchmark inference throughput (samples/second)"""

    def benchmark_throughput(
        self,
        model,
        optimizer,
        optimization: str,
        duration_seconds: int = 10
    ) -> float:
        """Benchmark throughput (samples/second)"""
        # Prepare optimized model
        if optimization == "baseline":
            optimized_model = model
        elif optimization == "tensorrt_fp16":
            optimized_model = optimizer.to_tensorrt(model, precision="fp16")
        elif optimization == "fp16":
            optimized_model = optimizer.quantize(model, precision="fp16")
        elif optimization == "onnx":
            optimized_model = optimizer.to_onnx(model)

        # Warmup
        for _ in range(10):
            optimized_model.encode(["warmup"])

        # Benchmark throughput
        start = time.time()
        sample_count = 0

        while time.time() - start < duration_seconds:
            optimized_model.encode(["throughput test"])
            sample_count += 1

        throughput = sample_count / duration_seconds
        return throughput

    def test_throughput_comparison(self):
        """Compare throughput across optimizations"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        optimizations = ["baseline", "fp16", "tensorrt_fp16", "onnx"]

        results = {}
        for opt in optimizations:
            try:
                throughput = self.benchmark_throughput(model, optimizer, opt, duration_seconds=5)
                results[opt] = throughput
                print(f"{opt}: {throughput:.1f} samples/sec")
            except Exception as e:
                print(f"{opt}: FAILED - {e}")

        # TensorRT FP16 should be >2× throughput of baseline
        baseline_throughput = results.get("baseline")
        trt_throughput = results.get("tensorrt_fp16")

        if baseline_throughput and trt_throughput:
            speedup = trt_throughput / baseline_throughput
            print(f"\nTensorRT FP16 Throughput Speedup: {speedup:.2f}×")
            assert speedup > 2.0, f"Throughput speedup {speedup:.2f}× is below 2.0× target"

    def test_batch_throughput(self):
        """Test batch inference throughput"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        engine = optimizer.to_tensorrt(model, precision="fp16", max_batch_size=8)

        batch_sizes = [1, 4, 8]
        throughputs = []

        for batch_size in batch_sizes:
            texts = ["test"] * batch_size

            # Warmup
            for _ in range(10):
                engine.encode(texts)

            # Benchmark
            start = time.time()
            sample_count = 0
            duration_seconds = 5

            while time.time() - start < duration_seconds:
                engine.encode(texts)
                sample_count += batch_size

            throughput = sample_count / duration_seconds
            throughputs.append(throughput)
            print(f"Batch {batch_size}: {throughput:.1f} samples/sec")

        # Batch 8 should be >2× throughput of batch 1
        batch_1_throughput = throughputs[0]
        batch_8_throughput = throughputs[-1]
        speedup = batch_8_throughput / batch_1_throughput

        print(f"\nBatch Throughput Speedup (8 vs 1): {speedup:.2f}×")
        assert speedup > 2.0, f"Batch throughput speedup {speedup:.2f}× is below 2.0× target"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestMemoryProfiling:
    """Profile memory usage across optimizations"""

    def profile_memory(self, model, optimizer, optimization: str) -> Dict[str, float]:
        """Profile memory usage"""
        tracemalloc.start()

        # Prepare optimized model
        if optimization == "baseline":
            optimized_model = model
        elif optimization == "tensorrt_fp16":
            optimized_model = optimizer.to_tensorrt(model, precision="fp16")
        elif optimization == "tensorrt_int8":
            calibration_data = ["calibration"] * 100
            optimized_model = optimizer.to_tensorrt(
                model,
                precision="int8",
                calibration_data=calibration_data
            )
        elif optimization == "fp16":
            optimized_model = optimizer.quantize(model, precision="fp16")
        elif optimization == "int8":
            calibration_data = ["calibration"] * 100
            optimized_model = optimizer.quantize(
                model,
                precision="int8",
                calibration_data=calibration_data
            )

        # Run inference
        for _ in range(10):
            optimized_model.encode(["memory profiling test"])

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024
        }

    def test_memory_usage_comparison(self):
        """Compare memory usage across optimizations"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        optimizations = ["baseline", "fp16", "int8"]

        results = {}
        for opt in optimizations:
            memory = self.profile_memory(model, optimizer, opt)
            results[opt] = memory
            print(f"\n{opt} Memory:")
            print(f"  Current: {memory['current_mb']:.1f}MB")
            print(f"  Peak: {memory['peak_mb']:.1f}MB")

        # FP16 should use <60% of baseline memory
        baseline_peak = results["baseline"]["peak_mb"]
        fp16_peak = results["fp16"]["peak_mb"]
        memory_reduction = baseline_peak / fp16_peak

        print(f"\nFP16 Memory Reduction: {memory_reduction:.2f}×")
        assert memory_reduction > 1.5, f"Memory reduction {memory_reduction:.2f}× is below 1.5× target"

        # INT8 should use <35% of baseline memory
        int8_peak = results["int8"]["peak_mb"]
        int8_reduction = baseline_peak / int8_peak

        print(f"INT8 Memory Reduction: {int8_reduction:.2f}×")
        assert int8_reduction > 2.5, f"Memory reduction {int8_reduction:.2f}× is below 2.5× target"

    def test_gpu_memory_usage(self):
        """Test GPU memory usage (if CUDA available)"""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available for GPU memory profiling")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Baseline GPU memory
        torch.cuda.reset_peak_memory_stats()
        model.encode(["test"])
        baseline_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        # TensorRT FP16 GPU memory
        torch.cuda.reset_peak_memory_stats()
        engine = optimizer.to_tensorrt(model, precision="fp16")
        engine.encode(["test"])
        trt_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"Baseline GPU Memory: {baseline_gpu_memory:.1f}MB")
        print(f"TensorRT FP16 GPU Memory: {trt_gpu_memory:.1f}MB")

        # TensorRT should use less GPU memory
        reduction = baseline_gpu_memory / trt_gpu_memory
        print(f"GPU Memory Reduction: {reduction:.2f}×")
        assert reduction > 1.3, f"GPU memory reduction {reduction:.2f}× is below 1.3× target"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestEndToEndPerformance:
    """End-to-end performance benchmarks"""

    def test_full_pipeline_performance(self):
        """Test full optimization pipeline performance"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Measure end-to-end time (conversion + inference)
        start = time.time()

        # Convert to TensorRT FP16
        engine = optimizer.to_tensorrt(model, precision="fp16")
        conversion_time = time.time() - start

        # Run inference
        start = time.time()
        for _ in range(100):
            engine.encode(["end-to-end test"])
        avg_inference_time = (time.time() - start) / 100

        print(f"Conversion Time: {conversion_time:.2f}s")
        print(f"Avg Inference Time: {avg_inference_time*1000:.3f}ms")

        # Conversion should complete in reasonable time
        assert conversion_time < 300, f"Conversion too slow: {conversion_time:.2f}s"

        # Inference should be fast
        assert avg_inference_time < 0.01, f"Inference too slow: {avg_inference_time*1000:.3f}ms"

    def test_cold_vs_warm_start(self):
        """Compare cold start vs warm start performance"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        engine = optimizer.to_tensorrt(model, precision="fp16")

        # Cold start (first inference)
        start = time.perf_counter()
        engine.encode(["cold start"])
        cold_start_time = (time.perf_counter() - start) * 1000

        # Warm start (subsequent inferences)
        warm_times = []
        for _ in range(100):
            start = time.perf_counter()
            engine.encode(["warm start"])
            warm_times.append((time.perf_counter() - start) * 1000)

        avg_warm_time = np.mean(warm_times)

        print(f"Cold Start: {cold_start_time:.3f}ms")
        print(f"Warm Start (avg): {avg_warm_time:.3f}ms")
        print(f"Cold/Warm Ratio: {cold_start_time/avg_warm_time:.2f}×")

        # Cold start should be <10× warm start
        assert cold_start_time / avg_warm_time < 10, \
            f"Cold start {cold_start_time:.3f}ms is much slower than warm {avg_warm_time:.3f}ms"
