from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

from query_classifier.onnx_engine import ONNXClassifierEngine


@dataclass
class PredictionResult:
    """Class representing prediction outcome for a single query."""
    label: str
    confidence: float
    latency_ms: float
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 2),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


class QueryClassifier:
    """
    High-level FastEmbed-style Query Classifier.
    
    Provides fast, low-memory ONNX inference to predict query difficulty 
    ('low', 'medium', 'hard') for LLM cascading & pre-routing.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        hf_repo: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize the QueryClassifier engine.
        
        Args:
            model_path: Optional path to local directory containing model.onnx, tokenizer.json, config.json.
            hf_repo: Optional HuggingFace hub repo ID (defaults to 'prvn-ramesh/query-classifier-onnx').
            providers: List of ONNXRuntime execution providers (default: ['CPUExecutionProvider']).
        """
        self.engine = ONNXClassifierEngine(
            model_path=model_path,
            hf_repo=hf_repo,
            providers=providers,
        )

    def predict(self, text: str) -> PredictionResult:
        """
        Classifies a single query into difficulty bucket ('low', 'medium', 'hard').

        Args:
            text: Input user query text.

        Returns:
            PredictionResult object containing label, confidence, latency_ms, and raw scores.
        """
        if not text or not text.strip():
            # Return low confidence default for empty query
            return PredictionResult(
                label="low",
                confidence=0.0,
                latency_ms=0.0,
                scores={"low": 1.0, "medium": 0.0, "hard": 0.0},
            )

        res = self.engine.predict_one(text)
        return PredictionResult(
            label=res["label"],
            confidence=res["confidence"],
            latency_ms=res["latency_ms"],
            scores=res["scores"],
        )

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """
        Classifies a batch of queries efficiently.

        Args:
            texts: List of query strings.

        Returns:
            List of PredictionResult objects.
        """
        if not texts:
            return []

        batch_res = self.engine.predict_batch(texts)
        return [
            PredictionResult(
                label=r["label"],
                confidence=r["confidence"],
                latency_ms=r["latency_ms"],
                scores=r["scores"],
            )
            for r in batch_res
        ]
