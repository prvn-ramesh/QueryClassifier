import os
import sys
from pathlib import Path
import pytest

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query_classifier import QueryClassifier, PredictionResult


@pytest.fixture(scope="module")
def classifier():
    return QueryClassifier()


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
