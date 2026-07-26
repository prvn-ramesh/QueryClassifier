# ⚡ QueryClassifier

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Playground-success?style=for-the-badge&logo=vercel)](https://query-classifier.vercel.app/)
[![PyPI version](https://img.shields.io/pypi/v/queryclf?style=for-the-badge&color=blue)](https://pypi.org/project/queryclf/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ONNX%20Model-yellow?style=for-the-badge)](https://huggingface.co/prvn-ramesh/query-classifier-onnx)


**Goal:** A high-speed, zero-heavy-dependency Python library (`queryclf`) that classifies user queries into `low`, `medium`, or `hard` complexity. Designed as an intelligent pre-routing engine for LLM cascades (e.g., routing simple queries to cheap models and complex queries to frontier models).

**Approach:** Fine-tuned `answerdotai/ModernBERT-large` quantized into int8 ONNX format (`query-classifier-onnx`). Uses a FastEmbed-inspired architecture powered by **pure ONNXRuntime + Rust `tokenizers`** — zero PyTorch, zero `transformers` library runtime dependencies, and sub-100ms CPU latency.

---

## 🌐 Interactive Playground

Try out QueryClassifier in your browser without installing anything:
👉 **[query-classifier.vercel.app](https://query-classifier.vercel.app/)**

---

## ⚡ Quickstart

### Installation

```bash
pip install queryclf
```

Dependencies installed are lightweight (~38MB total): `onnxruntime`, `tokenizers`, `numpy`, `huggingface-hub`, and `python-dotenv`.

---

## 📓 Fine-Tuning Notebook

The complete model fine-tuning and ONNX int8 quantization workflow is available as a Jupyter Notebook:
- **Notebook File:** [`notebooks/query_classifier_training.ipynb`](notebooks/query_classifier_training.ipynb)

---

**Resolution Priority:**
1. Explicit Python Argument (`hf_repo="..."` or `model_path="..."`)
2. Environment Variable `QUERY_CLASSIFIER_HF_REPO`
3. Environment Variable `QUERY_CLASSIFIER_MODEL_PATH`
4. Default Hugging Face Repository (`prvn-ramesh/query-classifier-onnx`)

---

## 🐍 Python API Usage

```python
import asyncio
from query_classifier import QueryClassifier

# Initializes classifier with built-in LRU cache and CPU thread tuning
classifier = QueryClassifier(
    cache_size=1000,             # Max items in LRU query cache (default: 1000)
    intra_op_num_threads=4,      # Intra-node CPU parallelism
)

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

# 3. Async prediction (for FastAPI / AsyncIO frameworks)
async def main():
    async_res = await classifier.predict_async("Explain quantum computing")
    print(f"Async prediction: {async_res.label} ({async_res.confidence:.2f})")

asyncio.run(main())

# 4. Cache statistics & management
print(classifier.cache_info())  # e.g., {'hits': 1, 'misses': 4, 'maxsize': 1000, 'currsize': 4}
classifier.clear_cache()
```

---

## 💻 CLI Command Usage

```bash
# Single query classification
queryclf "What is 2 + 2?"

# JSON output mode
queryclf "Write a Rust lock-free SPMC queue" --json

# Batch mode from file
queryclf --file queries.txt
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
