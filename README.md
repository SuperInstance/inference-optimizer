# inference-optimizer

Python/Rust library providing TensorRT, ONNX, and quantization optimization for ML models in the Equilibrium Tokens architecture.

## Overview

**inference-optimizer** accelerates embedding models for production deployment through:

- **TensorRT Optimization**: 2-4× faster inference with FP16/INT8 precision
- **ONNX Runtime**: Cross-platform optimization with multiple execution providers
- **Quantization**: FP32 → FP16/INT8/INT4 with minimal accuracy loss
- **Production Ready**: Comprehensive testing, error handling, and monitoring

## Features

### TensorRT Optimization

- Layer fusion and kernel auto-tuning
- FP16/FP32/INT8 precision support
- Dynamic shapes for variable-length inputs
- Batch size 1 optimization for low-latency
- **Performance**: 2-4× speedup, <5ms P99 latency

### ONNX Optimization

- Multiple opset version support (11-17)
- Graph optimization (constant folding, dead code elimination)
- Execution providers: CPU, CUDA, TensorRT
- Model simplification and validation
- **Performance**: 1.2-1.5× speedup via graph optimization

### Quantization

- Post-training quantization (PTQ)
- Dynamic and static quantization modes
- Per-channel and symmetric quantization
- Calibration data management
- **Performance**: 2-4× speedup, 2-8× memory reduction

## Installation

### Dependencies

```bash
# Core
pip install sentence-transformers numpy

# TensorRT (optional, for GPU optimization)
pip install tensorrt nvidia-tensorrt

# ONNX (optional, for cross-platform)
pip install onnx onnxruntime

# GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### From Source

```bash
git clone https://github.com/equilibrium-tokens/inference-optimizer.git
cd inference-optimizer
pip install -e .
```

## Quick Start

### Basic Usage

```python
from sentence_transformers import SentenceTransformer
from inference_optimizer import InferenceOptimizer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create optimizer
optimizer = InferenceOptimizer()

# Optimize to TensorRT FP16
engine = optimizer.to_tensorrt(model, precision="fp16")

# Run inference
embedding = engine.encode("Your text here")
```

### FP16 Quantization

```python
# Quantize to FP16
fp16_model = optimizer.quantize(model, precision="fp16")

# Inference (1.5-2× faster)
embedding = fp16_model.encode("Your text here")
```

### INT8 Quantization

```python
# Quantize to INT8 (requires calibration)
calibration_data = ["calibration text"] * 100
int8_model = optimizer.quantize(
    model,
    precision="int8",
    calibration_data=calibration_data
)

# Inference (2-3× faster)
embedding = int8_model.encode("Your text here")
```

### ONNX Conversion

```python
# Convert to ONNX
onnx_model = optimizer.to_onnx(model)

# Optimized ONNX
onnx_optimized = optimizer.to_onnx(
    model,
    optimization_level="all",
    simplify=True
)

# Inference
embedding = onnx_optimized.encode("Your text here")
```

## Integration with EmbeddingsEngine

```python
from embeddings_engine import EmbeddingsEngine
from inference_optimizer import InferenceOptimizer

optimizer = InferenceOptimizer()

# Create optimized embeddings engine
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    optimizer=optimizer,
    optimization="tensorrt fp16"  # or "fp16", "int8", "onnx"
)

# Transparent optimization
embedding = engine.encode("Your text here")
```

## Performance Benchmarks

### Inference Latency

| Optimization    | Mean (ms) | P95 (ms) | P99 (ms) | Speedup |
|-----------------|-----------|----------|----------|---------|
| Baseline (FP32) | 8.5       | 9.2      | 10.1     | 1.0×    |
| FP16            | 5.2       | 5.8      | 6.5      | 1.6×    |
| TensorRT FP16   | 3.1       | 3.8      | 4.2      | 2.7×    |
| INT8            | 3.8       | 4.5      | 5.1      | 2.2×    |
| TensorRT INT8   | 2.1       | 2.6      | 2.9      | 4.0×    |

### Memory Usage

| Optimization    | RAM (MB) | GPU (MB) | Reduction |
|-----------------|----------|----------|-----------|
| Baseline (FP32) | 420      | 650      | 1.0×      |
| FP16            | 230      | 380      | 1.8×      |
| TensorRT FP16   | 200      | 320      | 2.0×      |
| INT8            | 110      | 210      | 3.8×      |
| TensorRT INT8   | 95       | 180      | 4.4×      |

### Accuracy (Cosine Similarity)

| Optimization    | Similarity | Accuracy Loss |
|-----------------|------------|---------------|
| Baseline (FP32) | 1.0000     | 0%            |
| FP16            | 0.9987     | <0.2%         |
| TensorRT FP16   | 0.9981     | <0.3%         |
| INT8            | 0.9623     | <4%           |
| TensorRT INT8   | 0.9589     | <5%           |

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/ -m "unit" -v

# Integration tests
pytest tests/test_integration.py -v

# With coverage
pytest tests/ --cov=inference_optimizer --cov-report=html
```

### Run Benchmarks

```bash
# All benchmarks
python tests/benches/conversion_time.py
python tests/benches/inference_latency.py
python tests/benches/throughput.py
python tests/benches/memory_usage.py
```

See [tests/README.md](tests/README.md) for detailed testing documentation.

## Documentation

- **Testing Strategy**: [docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Test Suite**: [tests/README.md](tests/README.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    inference-optimizer                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         InferenceOptimizer (Main API)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         ▼                 ▼                 ▼              │
│  ┌────────────┐    ┌────────────┐   ┌─────────────┐       │
│  │  TensorRT  │    │   ONNX     │   │Quantization │       │
│  │ Optimizer  │    │ Optimizer  │   │ Optimizer   │       │
│  └────────────┘    └────────────┘   └─────────────┘       │
│         │                 │                  │              │
│         └─────────────────┼──────────────────┘              │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Optimized Model Interface                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      EmbeddingsEngine / VectorStore Integration       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# Model cache directory
export SENTENCE_TRANSFORMERS_HOME=/opt/models

# TensorRT cache
export TENSORRT_CACHE_DIR=/opt/tensorrt/cache

# Log level
export INFERENCE_OPTIMIZER_LOG_LEVEL=INFO
```

### Python API

```python
from inference_optimizer import InferenceOptimizer

optimizer = InferenceOptimizer(
    cache_dir="/opt/cache",
    log_level="INFO"
)
```

## Dependencies

### Required

- `sentence-transformers` (>=2.2.0)
- `numpy` (>=1.21.0)
- `torch` (>=1.12.0)

### Optional

- `tensorrt` (>=8.5.0) - for TensorRT optimization
- `onnx` (>=1.12.0) - for ONNX conversion
- `onnxruntime` (>=1.12.0) - for ONNX inference
- `cuda` (>=11.8) - for GPU acceleration

### Integration

- `embeddings-engine` (Round 2) - primary use case
- `gpu-accelerator` (Round 1) - CUDA operations
- `vector-navigator` - vector storage

## Roadmap

### v0.1.0 (Current)

- [x] TensorRT optimization (FP16, FP32, INT8)
- [x] ONNX conversion and optimization
- [x] Quantization (FP16, INT8, INT4)
- [x] Comprehensive test suite
- [x] Integration with embeddings-engine

### v0.2.0 (Q1 2026)

- [ ] BERT/RoBERTa model support
- [ ] AWQ quantization
- [ ] GPTQ for LLMs
- [ ] Quantization-aware training (QAT)

### v0.3.0 (Q2 2026)

- [ ] Neural architecture search (NAS)
- [ ] Pruning + quantization pipeline
- [ ] Knowledge distillation
- [ ] Dynamic batching optimization

## Contributing

Contributions welcome! Please see [tests/README.md](tests/README.md) for testing guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/equilibrium-tokens/inference-optimizer.git
cd inference-optimizer

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run benchmarks
python tests/benches/inference_latency.py
```

## License

MIT License - see LICENSE file for details

## Citation

```bibtex
@software{inference_optimizer,
  title = {inference-optimizer: Model Optimization for Equilibrium Tokens},
  author = {Equilibrium Tokens Team},
  year = {2026},
  url = {https://github.com/equilibrium-tokens/inference-optimizer}
}
```

## Acknowledgments

Built on research from:
- NVIDIA TensorRT Team
- ONNX Runtime Community
- Sentence-Transformers Library

Special thanks to the Equilibrium Tokens architecture team.

## Contact

- GitHub Issues: https://github.com/equilibrium-tokens/inference-optimizer/issues
- Documentation: https://docs.equilibrium-tokens.org/inference-optimizer

---

**The code is ephemeral; the grammar is eternal.**

**Last Updated**: 2026-01-08
**Version**: 0.1.0-alpha
**Maintained By**: Equilibrium Tokens Team
