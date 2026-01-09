"""
ONNX Optimization Tests

Tests for ONNX model conversion, optimization, and inference.
Based on ONNX Runtime and TensorRT-LLM ONNX capabilities.
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
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
class TestONNXConversion:
    """Test ONNX model conversion"""

    def test_to_onnx_basic(self):
        """Convert model to ONNX format"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert to ONNX
        onnx_model = optimizer.to_onnx(
            model,
            output_path="/tmp/test_model.onnx"
        )

        assert onnx_model is not None
        assert Path("/tmp/test_model.onnx").exists()

        # Verify ONNX model is valid
        onnx.checker.check_model(onnx_model)

    def test_to_onnx_with_opset(self):
        """Convert to ONNX with specific opset version"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert with opset 17
        onnx_model = optimizer.to_onnx(
            model,
            opset_version=17,
            output_path="/tmp/test_model_opset17.onnx"
        )

        assert onnx_model is not None
        assert onnx_model.opset_import[0].version == 17

    def test_to_onnx_dynamic_shapes(self):
        """Convert to ONNX with dynamic shapes"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Configure dynamic axes
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"}
        }

        onnx_model = optimizer.to_onnx(
            model,
            dynamic_axes=dynamic_axes,
            output_path="/tmp/test_model_dynamic.onnx"
        )

        assert onnx_model is not None

    def test_onnx_model_simplification(self):
        """Test ONNX model simplification (graph optimization)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert and simplify
        onnx_model = optimizer.to_onnx(
            model,
            simplify=True,
            output_path="/tmp/test_model_simplified.onnx"
        )

        # Simplified model should have fewer nodes
        assert onnx_model is not None

        # Count nodes before and after simplification
        original_model = optimizer.to_onnx(model, simplify=False)
        simplified_nodes = len(onnx_model.graph.node)
        original_nodes = len(original_model.graph.node)

        assert simplified_nodes <= original_nodes, \
            f"Simplification failed: {simplified_nodes} > {original_nodes} nodes"

    def test_onnx_model_serialization(self):
        """Test ONNX model serialization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Serialize to file
        onnx_model = optimizer.to_onnx(
            model,
            output_path="/tmp/test_serialized.onnx"
        )

        # Load from file
        loaded_model = optimizer.load_onnx_model("/tmp/test_serialized.onnx")

        assert loaded_model is not None
        assert loaded_model.graph.ByteSize() == onnx_model.graph.ByteSize()


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
class TestONNXInference:
    """Test ONNX inference"""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_onnx_inference_basic(self):
        """Test basic ONNX inference"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Convert to ONNX
        onnx_model = optimizer.to_onnx(model)

        # Run inference
        test_sentences = ["test sentence"]
        output = onnx_model.encode(test_sentences)

        assert output is not None
        assert output.shape == (1, 384)  # all-MiniLM-L6-v2 dimension

    def test_onnx_inference_accuracy(self):
        """Test ONNX inference accuracy compared to baseline"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Baseline output
        test_sentences = ["test sentence for accuracy"]
        baseline_output = model.encode(test_sentences)

        # ONNX output
        onnx_model = optimizer.to_onnx(model)
        onnx_output = onnx_model.encode(test_sentences)

        # Check cosine similarity
        similarity = self.cosine_similarity(baseline_output[0], onnx_output[0])
        assert similarity > 0.99, f"ONNX accuracy loss: {1 - similarity:.4f}"

    def test_onnx_batch_inference(self):
        """Test ONNX batch inference"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Batch input
        test_sentences = [
            "first sentence",
            "second sentence",
            "third sentence"
        ]

        # ONNX inference
        onnx_model = optimizer.to_onnx(model)
        output = onnx_model.encode(test_sentences)

        assert output.shape == (3, 384)

    def test_onnx_variable_length_inputs(self):
        """Test ONNX inference with variable-length inputs"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Variable-length sentences
        test_sentences = [
            "short",
            "medium length sentence",
            "this is a much longer sentence that should test variable length handling"
        ]

        # ONNX inference with dynamic shapes
        onnx_model = optimizer.to_onnx(
            model,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"}
            }
        )

        output = onnx_model.encode(test_sentences)

        assert output.shape == (3, 384)

    def test_onnx_inference_determinism(self):
        """Test ONNX inference determinism (same input = same output)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        onnx_model = optimizer.to_onnx(model)
        test_sentence = ["determinism test"]

        # Run inference twice
        output1 = onnx_model.encode(test_sentence)
        output2 = onnx_model.encode(test_sentence)

        # Outputs should be identical
        assert np.allclose(output1, output2, atol=1e-6)


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
class TestONNXOptimization:
    """Test ONNX graph optimization"""

    def benchmark(self, func, iterations: int = 100) -> float:
        """Benchmark inference function"""
        start_time = time.time()
        for _ in range(iterations):
            func()
        end_time = time.time()
        return (end_time - start_time) / iterations

    def test_onnx_optimization_level(self):
        """Test different ONNX optimization levels"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_text = "optimization test"

        # Test all optimization levels
        for opt_level in ["none", "basic", "extended", "all"]:
            onnx_model = optimizer.to_onnx(
                model,
                optimization_level=opt_level
            )

            # Should be able to run inference
            output = onnx_model.encode([test_text])
            assert output is not None

    def test_onnx_graph_optimization_speedup(self):
        """Test ONNX graph optimization speedup (>1.2× expected)"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_text = "graph optimization test"

        # Baseline (no optimization)
        baseline_model = optimizer.to_onnx(model, optimization_level="none")
        baseline_time = self.benchmark(lambda: baseline_model.encode([test_text]), 100)

        # Optimized
        optimized_model = optimizer.to_onnx(model, optimization_level="all")
        optimized_time = self.benchmark(lambda: optimized_model.encode([test_text]), 100)

        # Check speedup > 1.2×
        speedup = baseline_time / optimized_time
        assert speedup > 1.2, f"ONNX optimization speedup {speedup:.2f}× is below 1.2× target"

        print(f"ONNX Graph Optimization Speedup: {speedup:.2f}×")

    def test_onnx_constant_folding(self):
        """Test ONNX constant folding optimization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Enable constant folding
        onnx_model = optimizer.to_onnx(
            model,
            enable_constant_folding=True
        )

        # Model should be valid
        onnx.checker.check_model(onnx_model)

    def test_onnx_dead_code_elimination(self):
        """Test ONNX dead code elimination"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Enable dead code elimination
        onnx_model = optimizer.to_onnx(
            model,
            enable_dead_code_elimination=True
        )

        # Model should be valid
        onnx.checker.check_model(onnx_model)

    def test_onnx_operator_fusion(self):
        """Test ONNX operator fusion"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Enable operator fusion
        onnx_model = optimizer.to_onnx(
            model,
            enable_operator_fusion=True
        )

        # Model should be valid and faster
        onnx.checker.check_model(onnx_model)


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
class TestONNXRuntime:
    """Test ONNX Runtime execution providers"""

    def test_onnx_cpu_provider(self):
        """Test ONNX Runtime with CPU execution provider"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Use CPU provider
        onnx_model = optimizer.to_onnx(
            model,
            execution_provider="CPUExecutionProvider"
        )

        output = onnx_model.encode(["test"])
        assert output is not None

    def test_onnx_cuda_provider(self):
        """Test ONNX Runtime with CUDA execution provider"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Check if CUDA is available
        if "CUDAExecutionProvider" in ort.get_available_providers():
            onnx_model = optimizer.to_onnx(
                model,
                execution_provider="CUDAExecutionProvider"
            )

            output = onnx_model.encode(["test"])
            assert output is not None
        else:
            pytest.skip("CUDA execution provider not available")

    def test_onnx_tensorrt_provider(self):
        """Test ONNX Runtime with TensorRT execution provider"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Check if TensorRT is available
        available_providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available_providers:
            onnx_model = optimizer.to_onnx(
                model,
                execution_provider="TensorrtExecutionProvider"
            )

            output = onnx_model.encode(["test"])
            assert output is not None
        else:
            pytest.skip("TensorRT execution provider not available")

    def test_onnx_multiple_providers(self):
        """Test ONNX Runtime with multiple execution providers"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Configure fallback providers: CUDA -> CPU
        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        onnx_model = optimizer.to_onnx(
            model,
            execution_providers=providers
        )

        output = onnx_model.encode(["test"])
        assert output is not None


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
class TestONNXQuantization:
    """Test ONNX quantization"""

    def test_onnx_dynamic_quantization(self):
        """Test ONNX dynamic quantization"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Apply dynamic quantization (int8)
        quantized_model = optimizer.quantize_onnx(
            model,
            quantization_mode="dynamic",
            per_channel=False
        )

        # Should be able to run inference
        output = quantized_model.encode(["test"])
        assert output is not None

    def test_onnx_static_quantization(self):
        """Test ONNX static quantization with calibration"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Calibration dataset
        calibration_data = ["calibration sentence"] * 100

        # Apply static quantization
        quantized_model = optimizer.quantize_onnx(
            model,
            quantization_mode="static",
            calibration_data=calibration_data
        )

        # Should be able to run inference
        output = quantized_model.encode(["test"])
        assert output is not None

    def test_onnx_quantization_accuracy(self):
        """Test ONNX quantization accuracy"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        # Baseline
        test_sentences = ["quantization accuracy test"]
        baseline_output = model.encode(test_sentences)

        # Quantized model
        quantized_model = optimizer.quantize_onnx(
            model,
            quantization_mode="dynamic"
        )
        quantized_output = quantized_model.encode(test_sentences)

        # Check accuracy
        similarity = self.cosine_similarity(baseline_output[0], quantized_output[0])
        assert similarity > 0.95, f"Quantization accuracy loss: {1 - similarity:.4f}"

    def test_onnx_quantization_performance(self):
        """Test ONNX quantization performance improvement"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        optimizer = InferenceOptimizer()

        test_text = "performance test"

        # Baseline ONNX
        baseline_model = optimizer.to_onnx(model)
        baseline_time = self.benchmark(lambda: baseline_model.encode([test_text]), 100)

        # Quantized ONNX
        quantized_model = optimizer.quantize_onnx(
            model,
            quantization_mode="dynamic"
        )
        quantized_time = self.benchmark(lambda: quantized_model.encode([test_text]), 100)

        # Quantization should be faster
        speedup = baseline_time / quantized_time
        print(f"ONNX Quantization Speedup: {speedup:.2f}×")
        assert speedup > 1.1, f"Quantization speedup {speedup:.2f}× is below 1.1× target"
