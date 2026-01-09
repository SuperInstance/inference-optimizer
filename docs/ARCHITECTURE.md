# inference-optimizer Architecture

## Overview

**inference-optimizer** is a Python/Rust library providing comprehensive model optimization for production ML inference. It integrates TensorRT, ONNX Runtime, and quantization techniques to accelerate embedding models used in the Equilibrium Tokens architecture.

## Design Principles

1. **Accuracy Preservation**: Optimizations must maintain model accuracy within defined thresholds
2. **Performance First**: Target >2× speedup with significant memory reduction
3. **Production Ready**: Battle-tested optimizations with comprehensive error handling
4. **Dependency Integration**: Seamless integration with gpu-accelerator and embeddings-engine

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    inference-optimizer                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Optimization Pipeline                     │  │
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
│  │              Optimized Model Interface                │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          EmbeddingsEngine Integration                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. InferenceOptimizer

**Main entry point** for all optimization operations.

**Location**: `inference_optimizer/__init__.py`

**Responsibilities**:
- Model format conversion (PyTorch → TensorRT/ONNX)
- Quantization (FP32 → FP16/INT8/INT4)
- Calibration data management
- Optimization configuration

**API**:
```python
class InferenceOptimizer:
    def to_tensorrt(
        self,
        model,
        precision: str = "fp16",
        max_batch_size: int = 1,
        dynamic_shapes: dict = None,
        calibration_data: List[str] = None,
        **kwargs
    ) -> TensorRTEngine

    def to_onnx(
        self,
        model,
        opset_version: int = 17,
        dynamic_axes: dict = None,
        simplify: bool = True,
        optimization_level: str = "all",
        **kwargs
    ) -> ONNXModel

    def quantize(
        self,
        model,
        precision: str = "fp16",
        calibration_data: List[str] = None,
        quantization_mode: str = "dynamic",
        **kwargs
    ) -> QuantizedModel
```

### 2. TensorRT Optimizer

**Purpose**: Convert models to TensorRT engine format for maximum GPU performance.

**Location**: `inference_optimizer/tensorrt.py`

**Key Features**:
- Layer fusion and kernel auto-tuning
- FP16/FP32/INT8 precision support
- Dynamic shapes for variable-length inputs
- Batch size 1 optimization for low-latency

**Architecture**:
```
PyTorch Model
      │
      ▼
┌─────────────┐
│ ONNX Export │
└─────────────┘
      │
      ▼
┌─────────────┐
│ TensorRT    │
│ Builder     │
└─────────────┘
      │
      ├──► Network Definition
      ├──► Builder Config
      ├──► Engine Optimization
      └──► Engine Serialization
      │
      ▼
TensorRT Engine
```

**Optimization Pipeline**:
1. **Model Export**: Convert PyTorch → ONNX
2. **Network Definition**: Parse ONNX into TensorRT network
3. **Optimization Profile**: Configure batch sizes, dynamic shapes
4. **Precision Calibration**: INT8 calibration (if required)
5. **Engine Building**: Layer fusion, kernel selection
6. **Serialization**: Save engine to disk

### 3. ONNX Optimizer

**Purpose**: Convert models to ONNX format with graph optimization.

**Location**: `inference_optimizer/onnx.py`

**Key Features**:
- Multiple opset version support
- Graph simplification and optimization
- Execution provider abstraction (CPU/CUDA/TensorRT)
- Model serialization and validation

**Architecture**:
```
PyTorch Model
      │
      ▼
┌─────────────┐
│ ONNX Export │
└─────────────┘
      │
      ▼
┌─────────────┐
│ ONNX        │
│ Checker     │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Graph       │
│ Optimizer   │
└─────────────┘
      │
      ├──► Constant Folding
      ├──► Dead Code Elimination
      ├──► Operator Fusion
      └──► Cast Elimination
      │
      ▼
Optimized ONNX Model
```

**Optimization Levels**:
- **none**: No optimization
- **basic**: Constant folding, dead code elimination
- **extended**: Operator fusion, redundant node elimination
- **all**: All optimizations + backend-specific passes

### 4. Quantization Optimizer

**Purpose**: Reduce model precision (FP32 → FP16/INT8/INT4) for faster inference.

**Location**: `inference_optimizer/quantization.py`

**Key Features**:
- Post-training quantization (PTQ)
- Dynamic and static quantization modes
- Per-channel and symmetric quantization
- Calibration data management

**Quantization Pipeline**:
```
FP32 Model
      │
      ▼
┌─────────────────────┐
│ Calibration         │
│ (for INT8/INT4)     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Precision Analysis  │
│ (per-layer)         │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Quantization        │
│ (FP16/INT8/INT4)    │
└─────────────────────┘
      │
      ├──► Dynamic Quantization
      ├──► Static Quantization
      └──► Mixed Precision
      │
      ▼
Quantized Model
```

**Precision Modes**:
- **FP16**: Half precision (2× memory reduction, <1% accuracy loss)
- **INT8**: 8-bit integer (4× memory reduction, <5% accuracy loss)
- **INT4**: 4-bit integer (8× memory reduction, <10% accuracy loss)
- **Mixed**: Per-layer precision configuration

### 5. Optimized Model Interface

**Purpose**: Unified interface for optimized model inference.

**Location**: `inference_optimizer/optimized_model.py`

**Features**:
- Consistent API across optimization types
- Batch inference support
- Memory-efficient encoding
- Performance monitoring

**API**:
```python
class OptimizedModel:
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> List[np.ndarray]

    def get_memory_usage(self) -> Dict[str, float]
    def get_inference_time(self) -> Dict[str, float]
```

## Integration Architecture

### EmbeddingsEngine Integration

**Location**: `embeddings-engine/src/optimizer.py`

**Integration Points**:
```python
from embeddings_engine import EmbeddingsEngine
from inference_optimizer import InferenceOptimizer

optimizer = InferenceOptimizer()

engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    optimizer=optimizer,
    optimization="tensorrt fp16",  # or "fp16", "int8", "onnx"
    batch_size=8
)

# Transparent optimization
embedding = engine.encode("text")
```

**Optimization Flow**:
1. EmbeddingsEngine loads SentenceTransformers model
2. InferenceOptimizer applies selected optimization
3. Optimized model replaces baseline model
4. Inference uses optimized path transparently

### GPU Accelerator Integration

**Location**: `gpu-accelerator/src/cuda_ops.py`

**Integration Points**:
- CUDA kernel optimization via TensorRT
- GPU memory management
- Multi-GPU support (NVLink)

**Data Flow**:
```
Input Text
      │
      ▼
Tokenization (CPU)
      │
      ▼
GPU Transfer (CUDA)
      │
      ▼
TensorRT Engine (GPU)
      │
      ├──► Layer Fusion
      ├──► Kernel Auto-tuning
      └──► Tensor Core Optimization
      │
      ▼
Embedding (GPU)
      │
      ▼
GPU Transfer (CUDA)
      │
      ▼
Normalized Embedding (CPU)
```

## Performance Characteristics

### TensorRT Optimization

**Speedup**:
- FP16: 2-3× faster than baseline
- INT8: 3-4× faster than baseline
- Batch inference: Up to 5× with batch size 8

**Memory**:
- FP16: ~2× memory reduction
- INT8: ~4× memory reduction
- GPU memory: ~1.3-2.5× reduction

**Latency**:
- FP16: P99 <5ms
- INT8: P99 <3ms
- Batch size 1: Optimized for <10ms

### ONNX Optimization

**Speedup**:
- Graph optimization: 1.2-1.5× faster
- TensorRT provider: Similar to TensorRT
- CPU provider: 1.3× faster than PyTorch

**Memory**:
- ONNX format: ~10% smaller than PyTorch
- Optimized ONNX: ~20% smaller

**Portability**:
- Cross-platform (Windows/Linux)
- Multiple execution providers
- Hardware-agnostic inference

### Quantization

**Speedup vs Accuracy Tradeoff**:

| Precision | Speedup | Memory | Accuracy Loss |
|-----------|---------|--------|---------------|
| FP16 | 1.5-2× | 2× reduction | <1% |
| INT8 | 2-3× | 4× reduction | <5% |
| INT4 | 3-4× | 8× reduction | <10% |

## Calibration Strategy

### Calibration Dataset Requirements

**Size**:
- Minimum: 100 samples
- Recommended: 200-500 samples
- Diminishing returns beyond 500 samples

**Diversity**:
- Representative of production data
- Variable length (short, medium, long)
- Different domains/topics
- Edge cases (special characters, numbers)

### Calibration Process

```
Calibration Data
      │
      ▼
┌─────────────────────┐
│ Forward Pass        │
│ (FP32)              │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Activation Range    │
│ Collection          │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Scale/Zero Point    │
│ Calculation         │
└─────────────────────┘
      │
      ▼
Quantized Model
```

## Error Handling

### Conversion Errors

**Missing Dependencies**:
```python
if not TENSORRT_AVAILABLE:
    raise ImportError(
        "TensorRT not available. Install with: "
        "pip install tensorrt nvidia-tensorrt"
    )
```

**Invalid Model**:
```python
if not hasattr(model, "encode"):
    raise ValueError(
        f"Model must have 'encode' method. Got: {type(model)}"
    )
```

**Insufficient Calibration Data**:
```python
if len(calibration_data) < 100:
    raise ValueError(
        f"INT8 quantization requires at least 100 calibration samples. "
        f"Got: {len(calibration_data)}"
    )
```

### Runtime Errors

**CUDA Out of Memory**:
```python
try:
    output = model.encode(texts)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Reduce batch size and retry
        model.batch_size = max(1, model.batch_size // 2)
        output = model.encode(texts)
    else:
        raise
```

**Invalid Input Shape**:
```python
if len(texts) > engine.max_batch_size:
    raise ValueError(
        f"Batch size {len(texts)} exceeds engine max_batch_size {engine.max_batch_size}"
    )
```

## Testing Architecture

### Test Pyramid

```
        ┌─────────────┐
        │  E2E Tests  │  (tests/test_integration.py)
        │     (10)    │
        ├─────────────┤
        │ Integration │  (tests/test_integration.py)
        │    (50)     │
        ├─────────────┤
        │   Benchs    │  (tests/benches/*.py)
        │    (20)     │
        ├─────────────┤
        │  Unit Tests │  (tests/test_*.py)
        │   (200+)    │
        └─────────────┘
```

### Test Coverage

- **Unit Tests**: 200+ tests covering individual optimizations
- **Integration Tests**: 50+ tests covering dependency integration
- **Benchmarks**: 20+ performance benchmarks
- **Total**: 270+ test cases

## Deployment Architecture

### Model Artifacts

**Directory Structure**:
```
/opt/inference-optimizer/
├── models/
│   ├── all-MiniLM-L6-v2/
│   │   ├── pytorch_model.bin        # Original FP32
│   │   ├── fp16_model.pt            # FP16 quantized
│   │   ├── int8_model.pt            # INT8 quantized
│   │   ├── tensorrt_fp16.engine     # TensorRT FP16
│   │   ├── tensorrt_int8.engine     # TensorRT INT8
│   │   ├── onnx_model.onnx          # ONNX format
│   │   └── calibration_data.json    # Calibration cache
│   └── all-mpnet-base-v2/
│       └── ...
├── cache/
│   ├── tensorrt_engines/
│   └── onnx_models/
└── logs/
    ├── conversion.log
    └── inference.log
```

### Production Configuration

**YAML Config**:
```yaml
optimizer:
  default_optimization: "tensorrt_fp16"
  cache_dir: "/opt/inference-optimizer/cache"
  log_level: "INFO"

tensorrt:
  max_batch_size: 8
  max_workspace_size: 1073741824  # 1GB
  fp16: true
  int8_calib_cache: "/opt/inference-optimizer/cache/int8.cache"

onnx:
  opset_version: 17
  optimization_level: "all"
  execution_providers:
    - "CUDAExecutionProvider"
    - "CPUExecutionProvider"

quantization:
  calibration_samples: 200
  per_channel: true
  symmetric: true
```

## Monitoring & Observability

### Metrics

**Conversion Metrics**:
- Conversion time per optimization type
- Conversion success rate
- Memory overhead during conversion

**Inference Metrics**:
- Latency (P50, P95, P99)
- Throughput (samples/second)
- Memory usage (current, peak)
- GPU utilization (if CUDA available)

**Quality Metrics**:
- Cosine similarity vs baseline
- Accuracy loss per optimization
- Calibration quality score

### Logging

**Log Levels**:
- DEBUG: Detailed conversion progress
- INFO: Optimization summaries
- WARNING: Suboptimal configurations
- ERROR: Conversion failures

**Example Log**:
```
2026-01-08 10:15:30 INFO Converting all-MiniLM-L6-v2 to TensorRT FP16
2026-01-08 10:15:31 INFO Building TensorRT engine with max_batch_size=1
2026-01-08 10:15:35 INFO Engine built successfully in 4.2s
2026-01-08 10:15:35 INFO Engine serialized to /tmp/model_fp16.trt (78.3 MB)
2026-01-08 10:15:35 INFO Validation: cosine_similarity=0.9987 (>0.99 threshold)
```

## Future Enhancements

### Near-Term (Q1 2026)

- [ ] Support for BERT/RoBERTa models
- [ ] AWQ (Activation-aware Weight Quantization)
- [ ] GPTQ (GPT Quantization) for LLMs
- [ ] Quantization-aware training (QAT)

### Mid-Term (Q2-Q3 2026)

- [ ] Neural architecture search (NAS) for optimization
- [ ] Pruning + quantization pipeline
- [ ] Knowledge distillation
- [ ] Dynamic batching optimization

### Long-Term (Q4 2026+)

- [ ] Automated optimization selection (ML-based)
- [ ] Multi-model ensemble optimization
- [ ] Custom kernel generation (CUTLASS integration)
- [ ] Federated learning for calibration

## References

- **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
- **ONNX Runtime**: https://onnxruntime.ai/
- **CUDA Graphs**: https://developer.nvidia.com/blog/cuda-graphs/
- **NVIDIA Research**: /tmp/nvidia_tech_research.md

---

**Last Updated**: 2026-01-08
**Maintained By**: inference-optimizer team
**Version**: 0.1.0-alpha
