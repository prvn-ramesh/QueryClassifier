"""
QueryClassifier: Ultra-fast ONNX query difficulty classifier for pre-routing in LLM cascades.
"""

from query_classifier.classifier import QueryClassifier, PredictionResult

__version__ = "0.1.0"
__all__ = ["QueryClassifier", "PredictionResult", "__version__"]
