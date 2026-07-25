# QueryClassifier

**Goal:** A pip-installable Python library that classifies a text query into `low` / `medium` / `hard` complexity, for use as a pre-routing step in an LLM cascade (cheap model → medium model → frontier model).

**Approach:** Fine-tuned `answerdotai/ModernBERT-large` quantized into int8 ONNX format (`query-classifier-onnx`). Uses a FastEmbed-inspired architecture powered by **pure ONNXRuntime + Rust `tokenizers`** — zero PyTorch, zero `transformers` library dependencies, and zero heavy GPU requirements.

---

## ⚡ Quickstart

### Installation

```bash
pip install -e .
```

Dependencies installed are lightweight (~38MB total): `onnxruntime`, `tokenizers`, `numpy`, `huggingface-hub`, and `python-dotenv`.

---

## ⚙️ Configuration (`.env`)

Configure model paths dynamically in `.env` (refer to `.env.example`):

```env
# Hugging Face Repository ID (Default: prvn-ramesh/query-classifier-onnx)
QUERY_CLASSIFIER_HF_REPO=prvn-ramesh/query-classifier-onnx

# Optional local model directory override
QUERY_CLASSIFIER_MODEL_PATH=
```

**Resolution Priority:**
1. Explicit Python Argument (`hf_repo="..."` or `model_path="..."`)
2. Environment Variable `QUERY_CLASSIFIER_HF_REPO`
3. Environment Variable `QUERY_CLASSIFIER_MODEL_PATH`
4. Default Hugging Face Repository (`prvn-ramesh/query-classifier-onnx`)

---

## 🐍 Python API Usage

```python
from query_classifier import QueryClassifier

# Initializes model using Hugging Face Hub (prvn-ramesh/query-classifier-onnx) or .env configuration
classifier = QueryClassifier()

# 1. Single query prediction
result = classifier.predict("Can you summarize this document?")

print(result.label)        # Output: "low"
print(result.confidence)   # Output: 0.5779
print(result.latency_ms)   # Output: 29.8 ms
print(result.scores)       # Output: {'low': 0.5779, 'medium': 0.4201, 'hard': 0.0020}

# 2. Batch query prediction
queries = [
    "What is the capital of France?",
    "Calculate the integral of x^2 * sin(x) dx",
    "Write a lock-free multi-threaded queue in C++ using atomics"
]

batch_results = classifier.predict_batch(queries)
for q, res in zip(queries, batch_results):
    print(f"[{res.label.upper()}] ({res.latency_ms:.1f}ms) -> {q}")
```

---

## 💻 CLI Command Usage

```bash
# Single query classification
query-classifier "What is 2 + 2?"

# JSON output mode
query-classifier "Write a Rust lock-free SPMC queue" --json

# Batch mode from file
query-classifier --file queries.txt
```

---

## 📊 Benchmark & Accuracy Results

The underlying model [`prvn-ramesh/query-classifier-onnx`](https://huggingface.co/prvn-ramesh/query-classifier-onnx) is evaluated on an independent **held-out test set of 301 unseen queries** (balanced across `low`, `medium`, and `hard` buckets). For full details, see the [Hugging Face Model README](https://huggingface.co/prvn-ramesh/query-classifier-onnx).

### Accuracy & F1-Score Breakdown

- **Overall Accuracy**: **88.04%**
- **Macro F1-Score**: **0.8797**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **`low`** | **100.00%** | 81.00% | **0.8950** | 100 |
| **`medium`** | **82.18%** | 83.00% | **0.8259** | 100 |
| **`hard`** | **84.87%** | **100.00%** | **0.9182** | 101 |
| **Overall** | **88.04% Acc** | — | **0.8797 Macro-F1** | **301 total** |

### Latency & Throughput Profile (CPU Execution)

| Metric | Measurement |
| :--- | :--- |
| **Median Latency (p50)** | **67.68 ms** |
| **Mean Latency** | **81.44 ms** |
| **p90 Latency** | **134.69 ms** |
| **p99 Latency** | **161.74 ms** |
| **Throughput** | **12.3 queries/sec** |

---

## 🛠️ Project Roadmap Status

- [x] **Phase 0 — Environment & Scaffold**: Package directory layout (`src/query_classifier`, `tests/`, `pyproject.toml`).
- [x] **Phase 2 — Labeled Dataset**: `dataset.csv` with query samples.
- [x] **Phase 4 & 5 — Fine-Tuned ModernBERT & ONNX int8 Quantization**: Quantized model published on Hugging Face at [`prvn-ramesh/query-classifier-onnx`](https://huggingface.co/prvn-ramesh/query-classifier-onnx).
- [x] **Phase 6 — Inference Engine**: Pure ONNXRuntime + Tokenizers inference engine (`onnx_engine.py`, `classifier.py`, `cli.py`).
