# Testing Strategy for inference-optimizer

## Overview

This document outlines the comprehensive testing strategy for **inference-optimizer**, a Python/Rust library providing TensorRT, ONNX, and quantization optimization for ML models.

## Testing Philosophy

Our testing approach prioritizes:

1. **Accuracy First**: Optimizations must maintain model accuracy within defined thresholds
2. **Performance Validation**: Speedup and memory reduction targets must be met
3. **Production Readiness**: Tests simulate real-world usage patterns
4. **Integration Coverage**: Validate compatibility with dependencies (gpu-accelerator, embeddings-engine)

## Test Categories

### 1. Unit Tests

**Location**: `tests/test_*.py`

**Purpose**: Test individual optimization methods in isolation

#### TensorRT Tests (`test_tensorrt.py`)

- **Conversion Tests**
  - SentenceTransformers to TensorRT (FP16, FP32, INT8)
  - Engine serialization/deserialization
  - Dynamic shapes configuration
  - Batch size 1 optimization

- **Accuracy Tests**
  - FP16: >99% cosine similarity vs baseline
  - INT8: >95% cosine similarity vs baseline
  - Batch inference accuracy
  - Variable-length input accuracy

- **Performance Tests**
  - FP16 speedup: >2× vs baseline
  - INT8 speedup: >3× vs baseline
  - Memory reduction: >2× for FP16, >4× for INT8

#### ONNX Tests (`test_onnx.py`)

- **Conversion Tests**
  - Model to ONNX format
  - Opset version support
  - Dynamic shapes
  - Model simplification

- **Optimization Tests**
  - Optimization levels (none, basic, extended, all)
  - Graph optimization speedup: >1.2×
  - Constant folding, dead code elimination, operator fusion

- **Runtime Tests**
  - CPU execution provider
  - CUDA execution provider (if available)
  - TensorRT execution provider (if available)
  - Multiple providers with fallback

#### Quantization Tests (`test_quantization.py`)

- **FP16 Quantization**
  - Accuracy: >99% cosine similarity
  - Speedup: >1.5×
  - Memory reduction: >1.8×

- **INT8 Quantization**
  - Accuracy: >95% cosine similarity
  - Speedup: >2×
  - Memory reduction: >3.5×

- **INT4 Quantization** (experimental)
  - Accuracy: >90% cosine similarity
  - Speedup: >3×

- **Mixed Precision**
  - Per-layer precision configuration
  - Accuracy: >97% cosine similarity

- **Calibration Strategies**
  - Calibration data size (50, 100, 200, 500 samples)
  - Data diversity impact on accuracy

### 2. Integration Tests

**Location**: `tests/test_integration.py`

**Purpose**: Test integration with dependencies and end-to-end workflows

#### EmbeddingsEngine Integration

- TensorRT optimization with EmbeddingsEngine
- FP16/INT8 quantization with EmbeddingsEngine
- Batch encoding with optimization
- Optimization comparison across strategies

#### VectorStore Integration

- Storing optimized embeddings in vector store
- End-to-end RAG pipeline with optimization
- Batch vector storage
- Vector search performance with optimized embeddings

#### Multi-Model Optimization

- Multiple embedding models optimization
- Model switching between optimized engines

#### Performance Integration

- EmbeddingsEngine throughput with optimization
- Vector search performance
- High-volume batch processing (1000 docs)
- Concurrent encoding with ThreadPoolExecutor

#### CUDA Integration

- GPU acceleration with optimization
- Multi-GPU optimization (if available)

### 3. Benchmark Tests

**Location**: `tests/benches/*.py`

**Purpose**: Comprehensive performance benchmarking

#### Conversion Time Benchmarks (`benches/conversion_time.py`)

**Metrics**:
- Conversion time for each optimization type
- Model size impact on conversion time

**Models**:
- all-MiniLM-L6-v2 (small, 384 dim)
- all-mpnet-base-v2 (medium, 768 dim)

**Conversions**:
- TensorRT FP16/INT8
- ONNX (basic/optimized)
- FP16/INT8/INT4 quantization

**Output**: JSON file with conversion times

#### Inference Latency Benchmarks (`benches/inference_latency.py`)

**Metrics**:
- Mean, median, standard deviation
- P50, P90, P95, P99, P99.9 percentiles
- Latency consistency (P99/P50 ratio)

**Iterations**: 1000 per optimization

**Output**: JSON file with latency statistics

#### Throughput Benchmarks (`benches/throughput.py`)

**Scenarios**:
1. **Single-threaded throughput** (samples/second)
2. **Batch throughput** (batch sizes: 1, 4, 8, 16)
3. **Concurrent throughput** (workers: 1, 2, 4)

**Metrics**:
- Samples per second
- Batch efficiency (throughput per sample)
- Scaling efficiency (multi-worker vs single-worker)

**Output**: JSON file with throughput statistics

#### Memory Usage Profiling (`benches/memory_usage.py`)

**Metrics**:
1. **RAM Usage**
   - Current memory
   - Peak memory during inference

2. **GPU Memory** (if CUDA available)
   - Current GPU memory
   - Peak GPU memory

3. **Conversion Overhead**
   - Peak memory during model conversion

**Output**: JSON file with memory profiles

## Accuracy Validation Methodology

### Cosine Similarity Testing

**Formula**: `cosine_sim = (A · B) / (||A|| × ||B||)`

**Thresholds**:
- FP16: >0.99 ( <1% accuracy loss)
- INT8: >0.95 ( <5% accuracy loss)
- INT4: >0.90 ( <10% accuracy loss)
- Mixed precision: >0.97 ( <3% accuracy loss)

### Test Datasets

**Validation Dataset**:
- Single sentence: "test sentence for accuracy"
- Batch: 3 sentences of varying complexity
- Variable-length: Short (1 word), medium (10 words), long (20+ words)

**Calibration Dataset** (for INT8/INT4):
- Minimum 100 samples
- Diverse sentences representing production data
- Tested sizes: 50, 100, 200, 500 samples

### Accuracy Regression Testing

Each optimization is tested against:
1. **Baseline model** (original FP32)
2. **Previous optimization** (e.g., FP16 before INT8)
3. **Production embeddings** (if available)

## Performance Benchmarking Strategy

### Benchmarking Environment

**Requirements**:
- Minimal system load (close other applications)
- Consistent hardware (no thermal throttling)
- Warmup iterations (10) before benchmarking
- Multiple iterations (100-1000) for statistical significance

### Metric Targets

#### Inference Latency

| Optimization | Target Speedup | Max P99 Latency |
|--------------|----------------|-----------------|
| FP16 | >1.5× | <10ms |
| TensorRT FP16 | >2.0× | <5ms |
| INT8 | >2.0× | <5ms |
| TensorRT INT8 | >3.0× | <3ms |

#### Memory Reduction

| Optimization | Target Reduction |
|--------------|------------------|
| FP16 | >1.8× |
| INT8 | >3.5× |
| TensorRT FP16 | >2.0× (RAM), >1.3× (GPU) |
| TensorRT INT8 | >4.0× (RAM), >2.5× (GPU) |

#### Throughput

| Scenario | Target |
|----------|--------|
| Single-threaded | >100 samples/sec |
| Batch (8) | >500 samples/sec |
| Concurrent (4 workers) | >300 samples/sec |

### Benchmark Execution

**Local Development**:
```bash
# Run all benchmarks
pytest tests/benches/ -v

# Run specific benchmark
python tests/benches/inference_latency.py

# Quick validation (reduced iterations)
pytest tests/benches/inference_latency.py::test_inference_latency_comparison -v
```

**CI/CD**:
- Run subset of benchmarks (1 model, fewer iterations)
- Compare against baseline metrics
- Fail on regression >10%

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Test inference-optimizer

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov

    - name: Run unit tests
      run: pytest tests/ -v --cov=inference_optimizer

    - name: Run integration tests
      run: pytest tests/test_integration.py -v

    - name: Run benchmarks (CI mode)
      run: pytest tests/benches/ -v -m "ci"
      env:
        CI: true

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Markers

```python
# Unit tests (fast, run in CI)
@pytest.mark.unit

# Integration tests (require dependencies)
@pytest.mark.integration

# Benchmark tests (slow, CI mode only)
@pytest.mark.benchmark
@pytest.mark.ci  # Run in CI with reduced iterations

# GPU tests (require CUDA)
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available())
```

### CI Test Configuration

**Unit Tests** (all commits):
```bash
pytest tests/test_tensorrt.py -v -m "not gpu" --maxfail=3
pytest tests/test_onnx.py -v -m "not gpu" --maxfail=3
pytest tests/test_quantization.py -v --maxfail=3
```

**Integration Tests** (all commits):
```bash
pytest tests/test_integration.py -v -m "not gpu" --maxfail=5
```

**Benchmarks** (nightly, PR merges):
```bash
# CI mode (reduced iterations)
ITERATIONS=100 pytest tests/benches/ -v

# Full mode (nightly)
ITERATIONS=1000 pytest tests/benches/ -v
```

## Test Execution

### Local Development

**Run all tests**:
```bash
pytest tests/ -v
```

**Run specific test file**:
```bash
pytest tests/test_tensorrt.py -v
```

**Run specific test class**:
```bash
pytest tests/test_tensorrt.py::TestTensorRTConversion -v
```

**Run specific test**:
```bash
pytest tests/test_tensorrt.py::TestTensorRTConversion::test_sentencetransformers_to_tensorrt -v
```

**Run with coverage**:
```bash
pytest tests/ --cov=inference_optimizer --cov-report=html
```

### Skip Conditions

Tests are skipped if dependencies are unavailable:
- `SENTENCE_TRANSFORMERS_AVAILABLE`: sentence-transformers not installed
- `TENSORRT_AVAILABLE`: TensorRT not installed
- `ONNX_AVAILABLE`: ONNX not installed
- `EMBEDDINGS_ENGINE_AVAILABLE`: embeddings-engine not installed
- `VECTOR_STORE_AVAILABLE`: vector-navigator not installed
- `CUDA_AVAILABLE`: CUDA not available

## Continuous Improvement

### Performance Regression Detection

**Baseline Metrics** (stored in `tests/baselines.json`):
```json
{
  "tensorrt_fp16_latency_p50_ms": 2.5,
  "tensorrt_fp16_throughput_samples_per_sec": 400,
  "fp16_memory_reduction_mb": 150
}
```

**Regression Threshold**: 10% degradation from baseline

### Accuracy Regression Detection

**Minimum Accuracy Thresholds**:
- FP16: 0.99 cosine similarity
- INT8: 0.95 cosine similarity
- ONNX: 0.99 cosine similarity

### Test Maintenance

**Monthly**:
- Review test execution time
- Update baseline metrics
- Add new test cases for edge cases

**Quarterly**:
- Audit test coverage
- Remove obsolete tests
- Add benchmarks for new optimizations

## Success Criteria

### Test Coverage

- Unit test coverage: >80%
- Integration test coverage: >70%
- Critical paths: 100% coverage

### Performance Validation

All targets met:
- Accuracy thresholds maintained
- Speedup targets achieved
- Memory reduction targets achieved

### CI/CD Pass Rate

- Unit tests: >95% pass rate
- Integration tests: >90% pass rate
- Benchmarks: No regressions >10%

## Documentation

All tests include:
- Clear docstrings
- Parameter descriptions
- Expected outputs
- Error conditions

Benchmarks produce:
- JSON output files
- Human-readable summaries
- Comparison tables

---

**Last Updated**: 2026-01-08
**Maintained By**: inference-optimizer team
