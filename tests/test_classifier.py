import sys
from pathlib import Path
import pytest

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query_classifier import QueryClassifier, PredictionResult


LOCAL_MODEL_DIR = Path(__file__).parent.parent.parent / "query-classifier-onnx"


def get_test_classifier(**kwargs):
    if "model_path" not in kwargs and LOCAL_MODEL_DIR.exists():
        kwargs["model_path"] = LOCAL_MODEL_DIR
    return QueryClassifier(**kwargs)


@pytest.fixture(scope="module")
def classifier():
    return get_test_classifier()


def test_single_prediction_low(classifier):
    res = classifier.predict("What is the capital of France?")
    assert isinstance(res, PredictionResult)
    assert res.label in ["low", "medium", "hard"]
    assert res.label == "low"
    assert 0.0 <= res.confidence <= 1.0
    assert res.latency_ms > 0.0
    assert "low" in res.scores


def test_single_prediction_hard(classifier):
    res = classifier.predict("Write a multi-threaded C++ lock-free SPMC queue implementation with atomic CAS operations.")
    assert isinstance(res, PredictionResult)
    assert res.label in ["low", "medium", "hard"]
    assert 0.0 <= res.confidence <= 1.0


def test_empty_query(classifier):
    res = classifier.predict("")
    assert res.label == "low"
    assert res.confidence == 0.0
    assert res.latency_ms == 0.0


def test_batch_prediction(classifier):
    queries = [
        "What color is the sky?",
        "Explain quantum entanglement and its implications on cryptography.",
        "Calculate the derivative of f(x) = x^3 * sin(x).",
    ]
    results = classifier.predict_batch(queries)
    assert len(results) == len(queries)
    for r in results:
        assert isinstance(r, PredictionResult)
        assert r.label in ["low", "medium", "hard"]
        assert 0.0 <= r.confidence <= 1.0


def test_to_dict(classifier):
    res = classifier.predict("Sample query")
    d = res.to_dict()
    assert isinstance(d, dict)
    assert "label" in d
    assert "confidence" in d
    assert "latency_ms" in d
    assert "scores" in d


def test_lru_cache(classifier):
    classifier.clear_cache()
    query = "Unique test query for LRU cache verification"

    # First call -> cache miss
    res1 = classifier.predict(query)
    info1 = classifier.cache_info()
    assert info1["hits"] == 0
    assert info1["misses"] >= 1

    # Second call -> cache hit
    res2 = classifier.predict(query)
    info2 = classifier.cache_info()
    assert info2["hits"] == 1
    assert res2.label == res1.label
    assert res2.latency_ms == 0.0

    # Clear cache
    classifier.clear_cache()
    info3 = classifier.cache_info()
    assert info3["hits"] == 0
    assert info3["misses"] == 0
    assert info3["currsize"] == 0


def test_predict_async(classifier):
    import asyncio
    res = asyncio.run(classifier.predict_async("Explain photosynthesis"))
    assert isinstance(res, PredictionResult)
    assert res.label in ["low", "medium", "hard"]


def test_predict_batch_async(classifier):
    import asyncio
    queries = ["What is HTML?", "How does TCP handshake work?"]
    results = asyncio.run(classifier.predict_batch_async(queries))
    assert len(results) == 2
    for r in results:
        assert isinstance(r, PredictionResult)


def test_thread_options():
    c = get_test_classifier(intra_op_num_threads=2, inter_op_num_threads=1, cache_size=100)
    res = c.predict("Testing thread configuration")
    assert isinstance(res, PredictionResult)

