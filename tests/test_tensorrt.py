"""
TensorRT Conversion and Inference Tests

Tests for TensorRT model conversion, accuracy validation, and performance optimization.
Based on NVIDIA TensorRT 10.x and TensorRT-LLM capabilities.
"""

import pytest
import numpy as np
from typing import List
from pathlib import Path
import time

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not TENSORRT_AVAILABLE, reason="TensorRT not available")
class TestTensorRTConversion:
    """Test TensorRT model conversion from various formats"""

    def test_sentencetransformers_to_tensorrt(self):
        """Convert SentenceTransformers model to TensorRT engine"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert to TensorRT with FP16 precision
        engine = optimizer.to_tensorrt(
            model,
            precision="fp16",
            max_batch_size=1,
            max_sequence_length=512
        )

        assert engine is not None
        assert engine.precision == "fp16"
        assert engine.max_batch_size == 1

    def test_sentencetransformers_to_tensorrt_fp32(self):
        """Convert SentenceTransformers model to TensorRT with FP32"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert to TensorRT with FP32 precision
        engine = optimizer.to_tensorrt(
            model,
            precision="fp32",
            max_batch_size=1
        )

        assert engine is not None
        assert engine.precision == "fp32"

    def test_sentencetransformers_to_tensorrt_int8(self):
        """Convert SentenceTransformers model to TensorRT with INT8 quantization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert to TensorRT with INT8 precision
        # Requires calibration dataset
        calibration_data = ["test sentence"] * 100
        engine = optimizer.to_tensorrt(
            model,
            precision="int8",
            calibration_data=calibration_data,
            max_batch_size=1
        )

        assert engine is not None
        assert engine.precision == "int8"

    def test_tensorrt_engine_serialization(self):
        """Test TensorRT engine serialization and deserialization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert and serialize
        engine = optimizer.to_tensorrt(model, precision="fp16")
        engine_path = Path("/tmp/test_tensorrt_engine.trt")

        engine.serialize(engine_path)

        # Deserialize
        loaded_engine = optimizer.load_tensorrt_engine(engine_path)

        assert loaded_engine is not None
        assert loaded_engine.precision == engine.precision

        # Cleanup
        if engine_path.exists():
            engine_path.unlink()

    def test_tensorrt_dynamic_shapes(self):
        """Test TensorRT conversion with dynamic shapes"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Configure dynamic shapes for variable-length inputs
        engine = optimizer.to_tensorrt(
            model,
            precision="fp16",
            dynamic_shapes={
                "input_ids": (1, 64, 512),  # (min, opt, max)
                "attention_mask": (1, 64, 512)
            }
        )

        assert engine is not None
        assert engine.supports_dynamic_shapes

    def test_tensorrt_batch_optimization(self):
        """Test TensorRT batch size 1 optimization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Optimize for batch size 1 (low-latency use case)
        engine = optimizer.to_tensorrt(
            model,
            precision="fp16",
            max_batch_size=1,
            optimization_profile="low_latency"
        )

        assert engine is not None
        assert engine.max_batch_size == 1


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not TENSORRT_AVAILABLE, reason="TensorRT not available")
class TestTensorRTInferenceAccuracy:
    """Test TensorRT inference accuracy compared to baseline models"""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_tensorrt_inference_accuracy_fp16(self):
        """Validate TensorRT FP16 inference accuracy (<1% loss)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Baseline output
        test_sentences = ["test sentence for accuracy validation"]
        baseline_output = model.encode(test_sentences)

        # TensorRT FP16 output
        engine = optimizer.to_tensorrt(model, precision="fp16")
        trt_output = engine.encode(test_sentences)

        # Check cosine similarity > 0.99 (<1% accuracy loss)
        similarity = self.cosine_similarity(baseline_output[0], trt_output[0])
        assert similarity > 0.99, f"FP16 accuracy loss: {1 - similarity:.4f}"

    def test_tensorrt_inference_accuracy_int8(self):
        """Validate TensorRT INT8 inference accuracy (<5% loss)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Baseline output
        test_sentences = ["test sentence for int8 accuracy"]
        baseline_output = model.encode(test_sentences)

        # TensorRT INT8 output (requires calibration)
        calibration_data = ["calibration sentence"] * 100
        engine = optimizer.to_tensorrt(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
        trt_output = engine.encode(test_sentences)

        # Check cosine similarity > 0.95 (<5% accuracy loss)
        similarity = self.cosine_similarity(baseline_output[0], trt_output[0])
        assert similarity > 0.95, f"INT8 accuracy loss: {1 - similarity:.4f}"

    def test_tensorrt_batch_inference_accuracy(self):
        """Test TensorRT batch inference accuracy"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Batch of sentences
        test_sentences = [
            "first test sentence",
            "second test sentence",
            "third test sentence"
        ]

        # Baseline
        baseline_output = model.encode(test_sentences)

        # TensorRT
        engine = optimizer.to_tensorrt(model, precision="fp16")
        trt_output = engine.encode(test_sentences)

        # Check all samples maintain accuracy
        for i in range(len(test_sentences)):
            similarity = self.cosine_similarity(baseline_output[i], trt_output[i])
            assert similarity > 0.99, f"Batch sample {i} accuracy loss: {1 - similarity:.4f}"

    def test_tensorrt_variable_length_accuracy(self):
        """Test TensorRT accuracy with variable-length inputs"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Variable-length sentences
        test_sentences = [
            "short",
            "medium length sentence here",
            "this is a much longer sentence that tests the model's ability to handle variable input lengths"
        ]

        # Baseline
        baseline_output = model.encode(test_sentences)

        # TensorRT with dynamic shapes
        engine = optimizer.to_tensorrt(
            model,
            precision="fp16",
            dynamic_shapes={
                "input_ids": (1, 64, 512)
            }
        )
        trt_output = engine.encode(test_sentences)

        # Check all lengths maintain accuracy
        for i in range(len(test_sentences)):
            similarity = self.cosine_similarity(baseline_output[i], trt_output[i])
            assert similarity > 0.99, f"Length {len(test_sentences[i])} accuracy loss: {1 - similarity:.4f}"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not TENSORRT_AVAILABLE, reason="TensorRT not available")
class TestTensorRTPerformance:
    """Test TensorRT performance improvements over baseline"""

    def benchmark(self, func, iterations: int = 100) -> float:
        """Benchmark inference function"""
        start_time = time.time()
        for _ in range(iterations):
            func()
        end_time = time.time()
        return (end_time - start_time) / iterations

    def test_tensorrt_fp16_speedup(self):
        """Validate TensorRT FP16 speedup (>2× expected)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Benchmark baseline
        test_text = "test sentence for performance benchmarking"
        baseline_time = self.benchmark(lambda: model.encode([test_text]), 100)

        # Benchmark TensorRT FP16
        engine = optimizer.to_tensorrt(model, precision="fp16")
        trt_time = self.benchmark(lambda: engine.encode([test_text]), 100)

        # Check speedup > 2×
        speedup = baseline_time / trt_time
        assert speedup > 2.0, f"TensorRT FP16 speedup {speedup:.2f}× is below 2.0× target"

        print(f"TensorRT FP16 Speedup: {speedup:.2f}×")
        print(f"  Baseline: {baseline_time*1000:.3f}ms per inference")
        print(f"  TensorRT: {trt_time*1000:.3f}ms per inference")

    def test_tensorrt_int8_speedup(self):
        """Validate TensorRT INT8 speedup (>3× expected)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Benchmark baseline
        test_text = "test sentence for int8 benchmarking"
        baseline_time = self.benchmark(lambda: model.encode([test_text]), 100)

        # Benchmark TensorRT INT8
        calibration_data = ["calibration"] * 100
        engine = optimizer.to_tensorrt(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
        int8_time = self.benchmark(lambda: engine.encode([test_text]), 100)

        # Check speedup > 3×
        speedup = baseline_time / int8_time
        assert speedup > 3.0, f"TensorRT INT8 speedup {speedup:.2f}× is below 3.0× target"

        print(f"TensorRT INT8 Speedup: {speedup:.2f}×")

    def test_tensorrt_batch_throughput(self):
        """Test TensorRT batch inference throughput"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Single sentence baseline
        test_text = "throughput test sentence"
        baseline_time = self.benchmark(lambda: model.encode([test_text]), 100)

        # Batch inference
        engine = optimizer.to_tensorrt(model, precision="fp16", max_batch_size=8)
        batch_texts = [test_text] * 8
        batch_time = self.benchmark(lambda: engine.encode(batch_texts), 100) / 8

        # Batch should be more efficient
        batch_speedup = baseline_time / batch_time
        print(f"Batch Throughput Speedup: {batch_speedup:.2f}×")

    def test_tensorrt_memory_usage(self):
        """Test TensorRT memory footprint reduction"""
        import tracemalloc

        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Measure baseline memory
        tracemalloc.start()
        model.encode(["test"])
        baseline_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Measure TensorRT memory
        tracemalloc.start()
        engine = optimizer.to_tensorrt(model, precision="fp16")
        engine.encode(["test"])
        trt_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # TensorRT should use less memory (FP16 = 2× reduction)
        memory_reduction = baseline_memory / trt_memory
        print(f"Memory Reduction: {memory_reduction:.2f}×")
        assert memory_reduction > 1.5, "Memory reduction is below 1.5×"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not TENSORRT_AVAILABLE, reason="TensorRT not available")
class TestTensorRTFeatures:
    """Test advanced TensorRT features"""

    def test_tensorrt_layer_fusion(self):
        """Test TensorRT layer fusion optimization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        engine = optimizer.to_tensorrt(
            model,
            precision="fp16",
            enable_layer_fusion=True
        )

        # Verify layer fusion is enabled
        assert engine.layer_fusion_enabled

    def test_tensorrt_kernel_autotuning(self):
        """Test TensorRT kernel auto-tuning"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        engine = optimizer.to_tensorrt(
            model,
            precision="fp16",
            enable_kernel_autotuning=True
        )

        # Auto-tuning should select optimal kernels
        assert engine.kernel_autotuning_enabled

    def test_tensorrt_workspace_size(self):
        """Test TensorRT workspace memory configuration"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Configure workspace size (default: 1GB)
        engine = optimizer.to_tensorrt(
            model,
            precision="fp16",
            max_workspace_size=1 << 30  # 1GB
        )

        assert engine.max_workspace_size == 1 << 30

    def test_tensorrt_dla_support(self):
        """Test TensorRT DLA (Deep Learning Accelerator) support"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Check if DLA is available
        if optimizer.dla_available:
            engine = optimizer.to_tensorrt(
                model,
                precision="fp16",
                device_type="dla",
                dla_core=0
            )
            assert engine.device_type == "dla"
        else:
            pytest.skip("DLA not available on this system")
