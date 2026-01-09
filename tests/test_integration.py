"""
Integration Tests

Tests for integration with embeddings-engine and end-to-end optimization pipeline.
Validates compatibility with gpu-accelerator and vector store systems.
"""

import pytest
import numpy as np
from typing import List, Dict
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = True

try:
    from embeddings_engine import EmbeddingsEngine
    EMBEDDINGS_ENGINE_AVAILABLE = True
except ImportError:
    EMBEDDINGS_ENGINE_AVAILABLE = False

try:
    from vector_navigator import VectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

from inference_optimizer import InferenceOptimizer


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not EMBEDDINGS_ENGINE_AVAILABLE, reason="embeddings-engine not available")
class TestEmbeddingsEngineIntegration:
    """Test integration with embeddings-engine"""

    def test_embeddings_engine_tensorrt_optimization(self):
        """Test EmbeddingsEngine with TensorRT optimization"""
        optimizer = InferenceOptimizer()

        # Create optimized embeddings engine
        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="tensorrt fp16",
        )

        # Encode text
        embedding = engine.encode("test sentence")

        assert embedding is not None
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
        assert np.allclose(np.linalg.norm(embedding), 1.0)  # L2 normalized

    def test_embeddings_engine_fp16_optimization(self):
        """Test EmbeddingsEngine with FP16 quantization"""
        optimizer = InferenceOptimizer()

        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="fp16",
        )

        embedding = engine.encode("fp16 optimization test")

        assert embedding is not None
        assert len(embedding) == 384

    def test_embeddings_engine_int8_optimization(self):
        """Test EmbeddingsEngine with INT8 quantization"""
        optimizer = InferenceOptimizer()

        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="int8",
            calibration_data=["calibration"] * 100
        )

        embedding = engine.encode("int8 optimization test")

        assert embedding is not None
        assert len(embedding) == 384

    def test_embeddings_engine_batch_encoding(self):
        """Test EmbeddingsEngine batch encoding with optimization"""
        optimizer = InferenceOptimizer()

        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="tensorrt fp16",
            batch_size=8
        )

        # Batch encode
        texts = [
            "first sentence",
            "second sentence",
            "third sentence"
        ]

        embeddings = engine.encode_batch(texts)

        assert embeddings is not None
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_embeddings_engine_optimization_comparison(self):
        """Compare different optimization strategies in EmbeddingsEngine"""
        optimizer = InferenceOptimizer()

        optimizations = ["none", "fp16", "tensorrt fp16"]
        results = {}

        test_text = "comparison test"

        for opt in optimizations:
            engine = EmbeddingsEngine(
                model="all-MiniLM-L6-v2",
                optimizer=optimizer,
                optimization=opt
            )

            # Encode and measure
            import time
            start = time.time()
            embedding = engine.encode(test_text)
            elapsed = time.time() - start

            results[opt] = {
                "embedding": embedding,
                "time": elapsed
            }

        # All embeddings should be similar
        baseline_embedding = results["none"]["embedding"]

        for opt in optimizations:
            if opt == "none":
                continue
            embedding = results[opt]["embedding"]
            similarity = np.dot(baseline_embedding, embedding) / \
                        (np.linalg.norm(baseline_embedding) * np.linalg.norm(embedding))

            assert similarity > 0.95, f"{opt} embedding deviates too much from baseline"

            print(f"{opt}: {results[opt]['time']*1000:.3f}ms")


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not EMBEDDINGS_ENGINE_AVAILABLE, reason="embeddings-engine not available")
@pytest.mark.skipif(not VECTOR_STORE_AVAILABLE, reason="vector-navigator not available")
class TestVectorStoreIntegration:
    """Test integration with vector store and optimization"""

    def test_optimized_embeddings_vector_storage(self):
        """Test storing optimized embeddings in vector store"""
        optimizer = InferenceOptimizer()

        # Create optimized embeddings engine
        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="tensorrt fp16"
        )

        # Create vector store
        store = VectorStore(dimension=384)

        # Index documents
        documents = [
            {"id": 1, "text": "first document about machine learning"},
            {"id": 2, "text": "second document about neural networks"},
            {"id": 3, "text": "third document about deep learning"}
        ]

        for doc in documents:
            embedding = engine.encode(doc["text"])
            store.insert(embedding, doc)

        # Search
        query_embedding = engine.encode("machine learning research")
        results = store.search(query_embedding, k=3)

        assert len(results) == 3
        assert results[0]["id"] == 1  # Most relevant

    def test_end_to_end_rag_pipeline(self):
        """Test end-to-end RAG pipeline with optimization"""
        optimizer = InferenceOptimizer()

        # Create optimized embeddings engine
        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="fp16"
        )

        # Create vector store
        store = VectorStore(dimension=384)

        # Index knowledge base
        knowledge_base = [
            "Equilibrium Tokens is a conversational AI architecture",
            "The system uses rate, context, and sentiment equilibrium surfaces",
            "TensorRT optimization enables real-time inference",
            "Quantization reduces memory footprint while maintaining accuracy"
        ]

        for text in knowledge_base:
            embedding = engine.encode(text)
            store.insert(embedding, {"text": text})

        # Query
        query = "How does TensorRT help?"
        query_embedding = engine.encode(query)
        results = store.search(query_embedding, k=2)

        assert len(results) == 2
        assert "TensorRT" in results[0]["text"] or "TensorRT" in results[1]["text"]

    def test_batch_vector_storage(self):
        """Test batch vector storage with optimization"""
        optimizer = InferenceOptimizer()

        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="int8",
            calibration_data=["calibration"] * 100
        )

        store = VectorStore(dimension=384)

        # Batch encode and store
        texts = [f"document {i}" for i in range(100)]
        embeddings = engine.encode_batch(texts)

        for i, embedding in enumerate(embeddings):
            store.insert(embedding, {"id": i, "text": texts[i]})

        # Verify storage
        assert store.count == 100

        # Search
        query_embedding = engine.encode("document 5")
        results = store.search(query_embedding, k=5)

        assert len(results) == 5


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not EMBEDDINGS_ENGINE_AVAILABLE, reason="embeddings-engine not available")
class TestMultiModelOptimization:
    """Test optimization across multiple embedding models"""

    def test_multiple_models_optimization(self):
        """Test optimization of multiple embedding models"""
        optimizer = InferenceOptimizer()

        models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2"
        ]

        for model_name in models:
            engine = EmbeddingsEngine(
                model=model_name,
                optimizer=optimizer,
                optimization="fp16"
            )

            embedding = engine.encode("test")

            assert embedding is not None
            print(f"{model_name}: dimension={len(embedding)}")

    def test_model_switching_optimization(self):
        """Test switching between optimized models"""
        optimizer = InferenceOptimizer()

        # Create first optimized engine
        engine1 = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="tensorrt fp16"
        )

        # Create second optimized engine
        engine2 = EmbeddingsEngine(
            model="all-mpnet-base-v2",
            optimizer=optimizer,
            optimization="fp16"
        )

        # Use both engines
        text = "model switching test"
        embedding1 = engine1.encode(text)
        embedding2 = engine2.encode(text)

        assert embedding1 is not None
        assert embedding2 is not None
        assert len(embedding1) != len(embedding2)  # Different dimensions


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not EMBEDDINGS_ENGINE_AVAILABLE, reason="embeddings-engine not available")
class TestPerformanceIntegration:
    """Test performance improvements in integrated system"""

    def test_embeddings_engine_throughput(self):
        """Test EmbeddingsEngine throughput with optimization"""
        import time

        optimizer = InferenceOptimizer()

        # Baseline (no optimization)
        baseline_engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="none"
        )

        start = time.time()
        for _ in range(100):
            baseline_engine.encode("throughput test")
        baseline_time = time.time() - start

        # Optimized (TensorRT FP16)
        optimized_engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="tensorrt fp16"
        )

        start = time.time()
        for _ in range(100):
            optimized_engine.encode("throughput test")
        optimized_time = time.time() - start

        speedup = baseline_time / optimized_time
        print(f"EmbeddingsEngine Speedup: {speedup:.2f}×")
        assert speedup > 1.5, f"Speedup {speedup:.2f}× is below 1.5× target"

    def test_vector_search_performance(self):
        """Test vector search performance with optimized embeddings"""
        import time

        optimizer = InferenceOptimizer()

        # Create optimized engine
        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="fp16"
        )

        # Create vector store with 1000 documents
        store = VectorStore(dimension=384)

        for i in range(1000):
            text = f"document number {i} with some content"
            embedding = engine.encode(text)
            store.insert(embedding, {"id": i, "text": text})

        # Measure search time
        query = "search query"
        query_embedding = engine.encode(query)

        start = time.time()
        for _ in range(100):
            results = store.search(query_embedding, k=10)
        search_time = (time.time() - start) / 100

        print(f"Average Search Time: {search_time*1000:.3f}ms")
        assert search_time < 0.1, f"Search too slow: {search_time*1000:.3f}ms"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not EMBEDDINGS_ENGINE_AVAILABLE, reason="embeddings-engine not available")
class TestErrorHandlingIntegration:
    """Test error handling in integrated system"""

    def test_incompatible_optimization(self):
        """Test error handling for incompatible optimization"""
        optimizer = InferenceOptimizer()

        with pytest.raises(ValueError, match="Unsupported optimization"):
            EmbeddingsEngine(
                model="all-MiniLM-L6-v2",
                optimizer=optimizer,
                optimization="invalid_optimization"
            )

    def test_missing_calibration_data(self):
        """Test error handling for missing calibration data"""
        optimizer = InferenceOptimizer()

        with pytest.raises(ValueError, match="calibration data"):
            EmbeddingsEngine(
                model="all-MiniLM-L6-v2",
                optimizer=optimizer,
                optimization="int8"
                # Missing calibration_data
            )

    def test_model_not_found(self):
        """Test error handling for model not found"""
        optimizer = InferenceOptimizer()

        with pytest.raises(ValueError, match="model not found"):
            EmbeddingsEngine(
                model="nonexistent-model",
                optimizer=optimizer,
                optimization="fp16"
            )


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not EMBEDDINGS_ENGINE_AVAILABLE, reason="embeddings-engine not available")
class TestCudaIntegration:
    """Test CUDA/GPU integration with optimization"""

    def test_gpu_acceleration(self):
        """Test GPU acceleration with optimization"""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        optimizer = InferenceOptimizer()

        # Create GPU-accelerated engine
        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="tensorrt fp16",
            device="cuda"
        )

        embedding = engine.encode("gpu acceleration test")

        assert embedding is not None
        assert len(embedding) == 384

    def test_multi_gpu_optimization(self):
        """Test multi-GPU optimization (if available)"""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")

        optimizer = InferenceOptimizer()

        # Create engines on different GPUs
        engine0 = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="fp16",
            device="cuda:0"
        )

        engine1 = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="fp16",
            device="cuda:1"
        )

        # Both should work
        embedding0 = engine0.encode("gpu 0 test")
        embedding1 = engine1.encode("gpu 1 test")

        assert embedding0 is not None
        assert embedding1 is not None


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
@pytest.mark.skipif(not EMBEDDINGS_ENGINE_AVAILABLE, reason="embeddings-engine not available")
class TestProductionScenarios:
    """Test production-like scenarios"""

    def test_high_volume_batch_processing(self):
        """Test high-volume batch processing with optimization"""
        optimizer = InferenceOptimizer()

        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="tensorrt fp16",
            batch_size=32
        )

        # Process large batch
        texts = [f"document {i}" for i in range(1000)]

        import time
        start = time.time()
        embeddings = engine.encode_batch(texts)
        elapsed = time.time() - start

        throughput = len(texts) / elapsed
        print(f"Batch Processing Throughput: {throughput:.1f} docs/sec")

        assert len(embeddings) == 1000
        assert throughput > 100, f"Throughput {throughput:.1f} is below 100 docs/sec"

    def test_concurrent_encoding(self):
        """Test concurrent encoding with optimized engine"""
        import concurrent.futures

        optimizer = InferenceOptimizer()

        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="fp16"
        )

        # Concurrent encoding tasks
        def encode_task(text):
            return engine.encode(text)

        texts = [f"concurrent task {i}" for i in range(50)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(encode_task, text) for text in texts]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 50
        assert all(emb is not None for emb in results)

    def test_memory_efficient_large_batch(self):
        """Test memory-efficient large batch processing"""
        optimizer = InferenceOptimizer()

        # Use INT8 for memory efficiency
        engine = EmbeddingsEngine(
            model="all-MiniLM-L6-v2",
            optimizer=optimizer,
            optimization="int8",
            calibration_data=["calibration"] * 100,
            batch_size=64
        )

        # Process very large batch
        texts = [f"large batch document {i}" for i in range(500)]

        # Should not run out of memory
        embeddings = engine.encode_batch(texts)

        assert len(embeddings) == 500
