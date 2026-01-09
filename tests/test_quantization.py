"""
Quantization Tests

Tests for model quantization (FP32 -> FP16/INT8/INT4) including accuracy validation
and performance improvements.
Based on TensorRT and ONNX quantization capabilities.
"""

import pytest
import numpy as np
from typing import List
import time

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestFP16Quantization:
    """Test FP32 -> FP16 quantization"""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_fp32_to_fp16(self):
        """Test FP32 -> FP16 quantization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Quantize to FP16
        fp16_model = optimizer.quantize(model, precision="fp16")

        assert fp16_model is not None
        assert fp16_model.precision == "fp16"

    def test_fp16_accuracy(self):
        """Test FP16 quantization accuracy (<1% loss expected)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Baseline output
        test_sentences = ["test sentence for fp16 accuracy"]
        original_output = model.encode(test_sentences)

        # FP16 output
        fp16_model = optimizer.quantize(model, precision="fp16")
        fp16_output = fp16_model.encode(test_sentences)

        # Check cosine similarity > 0.99 (<1% accuracy loss)
        similarity = self.cosine_similarity(original_output[0], fp16_output[0])
        assert similarity > 0.99, f"FP16 accuracy loss: {1 - similarity:.4f}"

    def test_fp16_batch_accuracy(self):
        """Test FP16 accuracy with batch inputs"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_sentences = [
            "first sentence",
            "second sentence",
            "third sentence"
        ]

        # Baseline
        original_output = model.encode(test_sentences)

        # FP16
        fp16_model = optimizer.quantize(model, precision="fp16")
        fp16_output = fp16_model.encode(test_sentences)

        # Check all samples
        for i in range(len(test_sentences)):
            similarity = self.cosine_similarity(original_output[i], fp16_output[i])
            assert similarity > 0.99, f"FP16 batch[{i}] accuracy loss: {1 - similarity:.4f}"

    def test_fp16_speedup(self):
        """Test FP16 quantization speedup (>1.5× expected)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_text = "speedup test sentence"

        # Benchmark FP32
        start = time.time()
        for _ in range(100):
            model.encode([test_text])
        fp32_time = (time.time() - start) / 100

        # Benchmark FP16
        fp16_model = optimizer.quantize(model, precision="fp16")
        start = time.time()
        for _ in range(100):
            fp16_model.encode([test_text])
        fp16_time = (time.time() - start) / 100

        # Check speedup > 1.5×
        speedup = fp32_time / fp16_time
        assert speedup > 1.5, f"FP16 speedup {speedup:.2f}× is below 1.5× target"

        print(f"FP16 Quantization Speedup: {speedup:.2f}×")
        print(f"  FP32: {fp32_time*1000:.3f}ms")
        print(f"  FP16: {fp16_time*1000:.3f}ms")

    def test_fp16_memory_reduction(self):
        """Test FP16 memory reduction (2× expected)"""
        import tracemalloc

        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Measure FP32 memory
        tracemalloc.start()
        model.encode(["test"])
        fp32_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Measure FP16 memory
        fp16_model = optimizer.quantize(model, precision="fp16")
        tracemalloc.start()
        fp16_model.encode(["test"])
        fp16_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # FP16 should use ~2× less memory
        reduction = fp32_memory / fp16_memory
        print(f"FP16 Memory Reduction: {reduction:.2f}×")
        assert reduction > 1.8, f"Memory reduction {reduction:.2f}× is below 1.8× target"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestINT8Quantization:
    """Test FP32 -> INT8 quantization"""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_fp32_to_int8(self):
        """Test FP32 -> INT8 quantization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Quantize to INT8 (requires calibration)
        calibration_data = ["calibration sentence"] * 100
        int8_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=calibration_data
        )

        assert int8_model is not None
        assert int8_model.precision == "int8"

    def test_int8_accuracy(self):
        """Test INT8 quantization accuracy (<5% loss expected)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Baseline output
        test_sentences = ["test sentence for int8 accuracy"]
        original_output = model.encode(test_sentences)

        # INT8 output
        calibration_data = ["calibration"] * 100
        int8_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
        int8_output = int8_model.encode(test_sentences)

        # Check cosine similarity > 0.95 (<5% accuracy loss)
        similarity = self.cosine_similarity(original_output[0], int8_output[0])
        assert similarity > 0.95, f"INT8 accuracy loss: {1 - similarity:.4f}"

    def test_int8_batch_accuracy(self):
        """Test INT8 accuracy with batch inputs"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_sentences = [
            "first test",
            "second test",
            "third test"
        ]

        # Baseline
        original_output = model.encode(test_sentences)

        # INT8
        calibration_data = ["calibration"] * 100
        int8_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
        int8_output = int8_model.encode(test_sentences)

        # Check all samples
        for i in range(len(test_sentences)):
            similarity = self.cosine_similarity(original_output[i], int8_output[i])
            assert similarity > 0.95, f"INT8 batch[{i}] accuracy loss: {1 - similarity:.4f}"

    def test_int8_speedup(self):
        """Test INT8 quantization speedup (>2× expected)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_text = "int8 speedup test"

        # Benchmark FP32
        start = time.time()
        for _ in range(100):
            model.encode([test_text])
        fp32_time = (time.time() - start) / 100

        # Benchmark INT8
        calibration_data = ["calibration"] * 100
        int8_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
        start = time.time()
        for _ in range(100):
            int8_model.encode([test_text])
        int8_time = (time.time() - start) / 100

        # Check speedup > 2×
        speedup = fp32_time / int8_time
        assert speedup > 2.0, f"INT8 speedup {speedup:.2f}× is below 2.0× target"

        print(f"INT8 Quantization Speedup: {speedup:.2f}×")
        print(f"  FP32: {fp32_time*1000:.3f}ms")
        print(f"  INT8: {int8_time*1000:.3f}ms")

    def test_int8_memory_reduction(self):
        """Test INT8 memory reduction (4× expected)"""
        import tracemalloc

        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Measure FP32 memory
        tracemalloc.start()
        model.encode(["test"])
        fp32_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Measure INT8 memory
        calibration_data = ["calibration"] * 100
        int8_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=calibration_data
        )
        tracemalloc.start()
        int8_model.encode(["test"])
        int8_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # INT8 should use ~4× less memory
        reduction = fp32_memory / int8_memory
        print(f"INT8 Memory Reduction: {reduction:.2f}×")
        assert reduction > 3.5, f"Memory reduction {reduction:.2f}× is below 3.5× target"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestINT4Quantization:
    """Test FP32 -> INT4 quantization (experimental)"""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_fp32_to_int4(self):
        """Test FP32 -> INT4 quantization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Check if INT4 is supported
        if not optimizer.supports_int4:
            pytest.skip("INT4 quantization not supported")

        # Quantize to INT4
        calibration_data = ["calibration"] * 200
        int4_model = optimizer.quantize(
            model,
            precision="int4",
            calibration_data=calibration_data
        )

        assert int4_model is not None
        assert int4_model.precision == "int4"

    def test_int4_accuracy(self):
        """Test INT4 quantization accuracy (<10% loss acceptable)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        if not optimizer.supports_int4:
            pytest.skip("INT4 quantization not supported")

        # Baseline
        test_sentences = ["test sentence for int4 accuracy"]
        original_output = model.encode(test_sentences)

        # INT4
        calibration_data = ["calibration"] * 200
        int4_model = optimizer.quantize(
            model,
            precision="int4",
            calibration_data=calibration_data
        )
        int4_output = int4_model.encode(test_sentences)

        # Check cosine similarity > 0.90 (<10% accuracy loss)
        similarity = self.cosine_similarity(original_output[0], int4_output[0])
        assert similarity > 0.90, f"INT4 accuracy loss: {1 - similarity:.4f}"

    def test_int4_speedup(self):
        """Test INT4 quantization speedup (>3× expected)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        if not optimizer.supports_int4:
            pytest.skip("INT4 quantization not supported")

        test_text = "int4 speedup test"

        # Benchmark FP32
        start = time.time()
        for _ in range(100):
            model.encode([test_text])
        fp32_time = (time.time() - start) / 100

        # Benchmark INT4
        calibration_data = ["calibration"] * 200
        int4_model = optimizer.quantize(
            model,
            precision="int4",
            calibration_data=calibration_data
        )
        start = time.time()
        for _ in range(100):
            int4_model.encode([test_text])
        int4_time = (time.time() - start) / 100

        # Check speedup > 3×
        speedup = fp32_time / int4_time
        assert speedup > 3.0, f"INT4 speedup {speedup:.2f}× is below 3.0× target"

        print(f"INT4 Quantization Speedup: {speedup:.2f}×")


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestMixedPrecision:
    """Test mixed precision quantization strategies"""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_mixed_precision_fp16_int8(self):
        """Test mixed precision: FP16 for embeddings, INT8 for attention"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Configure mixed precision
        mixed_model = optimizer.quantize(
            model,
            precision="mixed",
            layer_precision={
                "embedding": "fp16",
                "encoder": "int8",
                "pooling": "fp16"
            },
            calibration_data=["calibration"] * 100
        )

        assert mixed_model is not None

    def test_mixed_precision_accuracy(self):
        """Test mixed precision accuracy"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Baseline
        test_sentences = ["mixed precision test"]
        original_output = model.encode(test_sentences)

        # Mixed precision
        mixed_model = optimizer.quantize(
            model,
            precision="mixed",
            layer_precision={
                "embedding": "fp16",
                "encoder": "int8",
                "pooling": "fp16"
            },
            calibration_data=["calibration"] * 100
        )
        mixed_output = mixed_model.encode(test_sentences)

        # Check accuracy
        similarity = self.cosine_similarity(original_output[0], mixed_output[0])
        assert similarity > 0.97, f"Mixed precision accuracy loss: {1 - similarity:.4f}"

    def test_per_channel_quantization(self):
        """Test per-channel quantization (better accuracy)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Per-channel INT8 quantization
        calibration_data = ["calibration"] * 100
        per_channel_model = optimizer.quantize(
            model,
            precision="int8",
            per_channel=True,
            calibration_data=calibration_data
        )

        assert per_channel_model is not None
        assert per_channel_model.per_channel_enabled

    def test_dynamic_quantization(self):
        """Test dynamic quantization (no calibration needed)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Dynamic quantization
        dynamic_model = optimizer.quantize(
            model,
            precision="int8",
            quantization_mode="dynamic"
        )

        assert dynamic_model is not None
        assert dynamic_model.quantization_mode == "dynamic"

        # Should work without calibration
        output = dynamic_model.encode(["test"])
        assert output is not None


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestQuantizationCalibration:
    """Test quantization calibration strategies"""

    def test_calibration_data_size(self):
        """Test effect of calibration data size on INT8 accuracy"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_sentences = ["calibration size test"]
        baseline_output = model.encode(test_sentences)

        # Test different calibration sizes
        calibration_sizes = [50, 100, 200, 500]
        accuracies = []

        for size in calibration_sizes:
            calibration_data = ["calibration sentence"] * size
            int8_model = optimizer.quantize(
                model,
                precision="int8",
                calibration_data=calibration_data
            )
            int8_output = int8_model.encode(test_sentences)

            similarity = np.dot(baseline_output[0], int8_output[0]) / \
                        (np.linalg.norm(baseline_output[0]) * np.linalg.norm(int8_output[0]))
            accuracies.append(similarity)

        # Accuracy should improve with more calibration data
        print(f"Calibration Size vs Accuracy:")
        for size, acc in zip(calibration_sizes, accuracies):
            print(f"  {size} samples: {acc:.4f}")

        # 100 samples should achieve >0.95 accuracy
        assert accuracies[1] > 0.95, "100 calibration samples insufficient"

    def test_calibration_data_diversity(self):
        """Test effect of calibration data diversity on INT8 accuracy"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_sentences = ["diversity test"]
        baseline_output = model.encode(test_sentences)

        # Repetitive calibration data
        repetitive_data = ["same sentence"] * 100
        repetitive_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=repetitive_data
        )
        repetitive_output = repetitive_model.encode(test_sentences)

        # Diverse calibration data
        diverse_data = [f"sentence {i}" for i in range(100)]
        diverse_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=diverse_data
        )
        diverse_output = diverse_model.encode(test_sentences)

        # Diverse data should produce better accuracy
        repetitive_sim = np.dot(baseline_output[0], repetitive_output[0]) / \
                        (np.linalg.norm(baseline_output[0]) * np.linalg.norm(repetitive_output[0]))
        diverse_sim = np.dot(baseline_output[0], diverse_output[0]) / \
                     (np.linalg.norm(baseline_output[0]) * np.linalg.norm(diverse_output[0]))

        print(f"Repetitive calibration accuracy: {repetitive_sim:.4f}")
        print(f"Diverse calibration accuracy: {diverse_sim:.4f}")

        assert diverse_sim >= repetitive_sim, "Diverse calibration should be better or equal"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestQuantizationEdgeCases:
    """Test quantization edge cases and error handling"""

    def test_quantization_unsupported_precision(self):
        """Test error handling for unsupported precision"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        with pytest.raises(ValueError, match="Unsupported precision"):
            optimizer.quantize(model, precision="fp64")

    def test_quantization_insufficient_calibration(self):
        """Test error handling for insufficient calibration data"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        with pytest.raises(ValueError, match="Insufficient calibration data"):
            optimizer.quantize(
                model,
                precision="int8",
                calibration_data=["only one"]
            )

    def test_quantization_model_compatibility(self):
        """Test quantization compatibility across model types"""
        models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
        ]

        for model_name in models:
            model = SentenceTransformer(model_name)
            optimizer = InferenceOptimizer()

            # Should not raise errors
            fp16_model = optimizer.quantize(model, precision="fp16")
            assert fp16_model is not None

    def test_quantization_with_custom_config(self):
        """Test quantization with custom configuration"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Custom quantization config
        custom_model = optimizer.quantize(
            model,
            precision="int8",
            calibration_data=["calibration"] * 100,
            quantization_config={
                "per_channel": True,
                "reduce_range": False,
                "use_symmetric_quantization": True
            }
        )

        assert custom_model is not None
