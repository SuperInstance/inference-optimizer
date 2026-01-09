# inference-optimizer Test Suite

Comprehensive test suite for model inference optimization including TensorRT, ONNX, and quantization.

## Test Structure

```
tests/
├── test_tensorrt.py          # TensorRT conversion and inference tests
├── test_onnx.py              # ONNX optimization tests
├── test_quantization.py      # Quantization accuracy tests
├── test_benchmarks.py        # Performance benchmarks
├── test_integration.py       # Integration with embeddings-engine
└── benches/                  # Performance benchmarking scripts
    ├── conversion_time.py    # Model conversion benchmarks
    ├── inference_latency.py  # Inference latency benchmarks
    ├── throughput.py         # Throughput benchmarks
    └── memory_usage.py       # Memory profiling
```

## Quick Start

### Install Dependencies

```bash
# Core dependencies
pip install sentence-transformers
pip install pytest pytest-cov

# Optional (for specific test suites)
pip install tensorrt onnxruntime-gpu  # TensorRT/ONNX tests
pip install torch torchvision         # GPU tests
```

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=inference_optimizer --cov-report=html

# Run unit tests only (fast)
pytest tests/ -m "unit" -v

# Run integration tests
pytest tests/ -m "integration" -v
```

### Run Specific Test Files

```bash
# TensorRT tests
pytest tests/test_tensorrt.py -v

# ONNX tests
pytest tests/test_onnx.py -v

# Quantization tests
pytest tests/test_quantization.py -v
```

### Run Benchmarks

```bash
# All benchmarks (fast mode)
pytest tests/test_benchmarks.py -v

# Standalone benchmark scripts
python tests/benches/conversion_time.py
python tests/benches/inference_latency.py
python tests/benches/throughput.py
python tests/benches/memory_usage.py
```

## Test Categories

### 1. Unit Tests (`tests/test_*.py`)

**Purpose**: Test individual optimization methods

**Tests**:
- **TensorRT** (200+ tests):
  - Conversion to TensorRT (FP16, FP32, INT8)
  - Accuracy validation (>99% FP16, >95% INT8)
  - Performance validation (>2× speedup)
  - Engine serialization

- **ONNX** (150+ tests):
  - ONNX conversion and optimization
  - Graph optimization speedup (>1.2×)
  - Execution providers (CPU, CUDA, TensorRT)
  - Model simplification

- **Quantization** (100+ tests):
  - FP16/INT8/INT4 quantization
  - Accuracy thresholds
  - Calibration strategies
  - Mixed precision

### 2. Integration Tests (`tests/test_integration.py`)

**Purpose**: Test integration with dependencies

**Tests**:
- EmbeddingsEngine optimization
- VectorStore integration
- End-to-end RAG pipeline
- Multi-model optimization
- GPU acceleration
- Production scenarios

### 3. Benchmarks (`tests/test_benchmarks.py`, `tests/benches/*.py`)

**Purpose**: Performance validation

**Benchmarks**:
- **Conversion Time**: Time to convert models
- **Inference Latency**: P50, P95, P99 latencies
- **Throughput**: Samples per second
- **Memory Usage**: RAM and GPU memory

## Test Markers

```bash
# Run only unit tests
pytest tests/ -m "unit" -v

# Run only integration tests
pytest tests/ -m "integration" -v

# Run only GPU tests
pytest tests/ -m "gpu" -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Run CI-specific tests (reduced iterations)
pytest tests/ -m "ci" -v
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Install dependencies
      run: |
        pip install sentence-transformers pytest pytest-cov
    - name: Run unit tests
      run: pytest tests/ -m "unit" --cov
    - name: Run integration tests
      run: pytest tests/test_integration.py -v
```

## Test Configuration

### Environment Variables

```bash
# CI mode (reduced iterations)
export CI=true

# Skip GPU tests
export SKIP_GPU_TESTS=true

# Benchmark iterations (default: 1000)
export BENCHMARK_ITERATIONS=100

# Model download directory
export SENTENCE_TRANSFORMERS_HOME=/tmp/models
```

### Pytest Configuration

See `pytest.ini` for configuration:
- Test discovery patterns
- Output formatting
- Coverage settings
- Custom markers

## Accuracy Thresholds

| Optimization | Cosine Similarity | Accuracy Loss |
|--------------|-------------------|---------------|
| Baseline     | 1.0000            | 0%            |
| FP16         | >0.99             | <1%           |
| INT8         | >0.95             | <5%           |
| INT4         | >0.90             | <10%          |
| ONNX         | >0.99             | <1%           |

## Performance Targets

### Inference Latency

| Optimization    | Target Speedup | Max P99 Latency |
|-----------------|----------------|-----------------|
| FP16            | >1.5×          | <10ms           |
| TensorRT FP16   | >2.0×          | <5ms            |
| INT8            | >2.0×          | <5ms            |
| TensorRT INT8   | >3.0×          | <3ms            |

### Memory Reduction

| Optimization    | RAM Reduction  | GPU Reduction |
|-----------------|----------------|---------------|
| FP16            | >1.8×         | >1.3×         |
| INT8            | >3.5×         | >2.5×         |
| TensorRT FP16   | >2.0×         | >2.0×         |
| TensorRT INT8   | >4.0×         | >3.0×         |

## Debugging Tests

### Verbose Output

```bash
pytest tests/test_tensorrt.py::TestTensorRTConversion::test_sentencetransformers_to_tensorrt -vv
```

### Stop on First Failure

```bash
pytest tests/ -x -v
```

### Enter Debugger on Failure

```bash
pytest tests/ -x -v --pdb
```

### Show Local Variables on Failure

```bash
pytest tests/ -x -v -l
```

## Skip Conditions

Tests are automatically skipped if dependencies are unavailable:

- `SENTENCE_TRANSFORMERS_AVAILABLE`: sentence-transformers not installed
- `TENSORRT_AVAILABLE`: TensorRT not installed
- `ONNX_AVAILABLE`: ONNX not installed
- `EMBEDDINGS_ENGINE_AVAILABLE`: embeddings-engine not installed
- `VECTOR_STORE_AVAILABLE`: vector-navigator not installed
- `CUDA_AVAILABLE`: CUDA not available

## Benchmark Results

Benchmark results are saved to `/tmp/`:

- `/tmp/conversion_benchmarks.json`
- `/tmp/inference_latency_benchmarks.json`
- `/tmp/throughput_benchmarks.json`
- `/tmp/memory_usage_profiles.json`

## Contributing

When adding new tests:

1. **Unit Tests**: Add to appropriate `test_*.py` file
2. **Integration Tests**: Add to `test_integration.py`
3. **Benchmarks**: Add to `test_benchmarks.py` or create new bench script

**Test Template**:

```python
import pytest

@pytest.mark.unit
class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic functionality"""
        assert True

    def test_edge_case(self):
        """Test edge case"""
        with pytest.raises(ValueError):
            # Code that should raise ValueError
            pass
```

## Troubleshooting

### Import Errors

```bash
# Error: ImportError: No module named 'sentence_transformers'
pip install sentence-transformers

# Error: ImportError: No module named 'tensorrt'
pip install tensorrt nvidia-tensorrt
```

### CUDA Out of Memory

```bash
# Reduce batch size in test
export TEST_BATCH_SIZE=1

# Or skip GPU tests
pytest tests/ -m "not gpu" -v
```

### Model Download Issues

```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Or set custom cache directory
export SENTENCE_TRANSFORMERS_HOME=/path/to/cache
```

## Resources

- **Testing Strategy**: `docs/TESTING_STRATEGY.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **NVIDIA Research**: `/tmp/nvidia_tech_research.md`

---

**Last Updated**: 2026-01-08
**Maintained By**: inference-optimizer team
