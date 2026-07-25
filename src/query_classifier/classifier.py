import asyncio
from collections import OrderedDict
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
        cache_size: Optional[int] = 1000,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ):
        """
        Initialize the QueryClassifier engine.
        
        Args:
            model_path: Optional path to local directory containing model.onnx, tokenizer.json, config.json.
            hf_repo: Optional HuggingFace hub repo ID (defaults to 'prvn-ramesh/query-classifier-onnx').
            providers: List of ONNXRuntime execution providers (default: ['CPUExecutionProvider']).
            cache_size: Max entries for LRU query cache (set to 0 or None to disable caching).
            intra_op_num_threads: Number of threads used to parallelize execution within nodes.
            inter_op_num_threads: Number of threads used to parallelize execution of different nodes.
        """
        self.engine = ONNXClassifierEngine(
            model_path=model_path,
            hf_repo=hf_repo,
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )

        self.cache_size = cache_size if (cache_size is not None and cache_size > 0) else 0
        self._cache: OrderedDict[str, PredictionResult] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def clear_cache(self) -> None:
        """Clears the in-memory LRU prediction cache and resets hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cache_info(self) -> Dict[str, Any]:
        """Returns statistics about the prediction cache."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "maxsize": self.cache_size,
            "currsize": len(self._cache),
        }

    def _get_from_cache(self, text: str) -> Optional[PredictionResult]:
        if not self.cache_size:
            return None
        if text in self._cache:
            self._cache.move_to_end(text)
            self._hits += 1
            cached = self._cache[text]
            return PredictionResult(
                label=cached.label,
                confidence=cached.confidence,
                latency_ms=0.0,
                scores=dict(cached.scores),
            )
        self._misses += 1
        return None

    def _put_in_cache(self, text: str, result: PredictionResult) -> None:
        if not self.cache_size:
            return
        if text in self._cache:
            self._cache.move_to_end(text)
        self._cache[text] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def predict(self, text: str) -> PredictionResult:
        """
        Classifies a single query into difficulty bucket ('low', 'medium', 'hard').

        Args:
            text: Input user query text.

        Returns:
            PredictionResult object containing label, confidence, latency_ms, and raw scores.
        """
        if not text or not text.strip():
            return PredictionResult(
                label="low",
                confidence=0.0,
                latency_ms=0.0,
                scores={"low": 1.0, "medium": 0.0, "hard": 0.0},
            )

        cached_res = self._get_from_cache(text)
        if cached_res is not None:
            return cached_res

        res = self.engine.predict_one(text)
        pred = PredictionResult(
            label=res["label"],
            confidence=res["confidence"],
            latency_ms=res["latency_ms"],
            scores=res["scores"],
        )
        self._put_in_cache(text, pred)
        return pred

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

        results: List[Optional[PredictionResult]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = PredictionResult(
                    label="low",
                    confidence=0.0,
                    latency_ms=0.0,
                    scores={"low": 1.0, "medium": 0.0, "hard": 0.0},
                )
                continue

            cached_res = self._get_from_cache(text)
            if cached_res is not None:
                results[i] = cached_res
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            batch_res = self.engine.predict_batch(missing_texts)
            for idx, text, r in zip(missing_indices, missing_texts, batch_res):
                pred = PredictionResult(
                    label=r["label"],
                    confidence=r["confidence"],
                    latency_ms=r["latency_ms"],
                    scores=r["scores"],
                )
                self._put_in_cache(text, pred)
                results[idx] = pred

        return [r for r in results if r is not None]

    async def predict_async(self, text: str) -> PredictionResult:
        """
        Asynchronously classifies a single query without blocking the asyncio event loop.
        """
        return await asyncio.to_thread(self.predict, text)

    async def predict_batch_async(self, texts: List[str]) -> List[PredictionResult]:
        """
        Asynchronously classifies a batch of queries without blocking the asyncio event loop.
        """
        return await asyncio.to_thread(self.predict_batch, texts)
