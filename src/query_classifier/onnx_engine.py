import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, try_to_load_from_cache

# Load environment variables automatically if .env exists
load_dotenv()

DEFAULT_HF_REPO = "prvn-ramesh/query-classifier-onnx"
DEFAULT_ID2LABEL = {"0": "low", "1": "medium", "2": "hard"}


def _get_hf_file(repo_id: str, filename: str) -> Path:
    cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    if cached is None or isinstance(cached, Exception):
        print(f"Downloading model artifact '{filename}' from Hugging Face Hub ('{repo_id}')...", file=sys.stderr, flush=True)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def resolve_model_files(
    model_path: Optional[Union[str, Path]] = None,
    hf_repo: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Resolves the locations of model.onnx, tokenizer.json, and config.json.
    Resolution priority:
      1. Explicit argument (`hf_repo` or `model_path`)
      2. Environment variable `QUERY_CLASSIFIER_HF_REPO`
      3. Environment variable `QUERY_CLASSIFIER_MODEL_PATH`
      4. Default Hugging Face repository (`prvn-ramesh/query-classifier-onnx`)
    """
    repo = hf_repo or os.getenv("QUERY_CLASSIFIER_HF_REPO")
    local_dir = model_path or os.getenv("QUERY_CLASSIFIER_MODEL_PATH")

    if repo and repo.strip():
        repo_id = repo.strip()
        model_file = _get_hf_file(repo_id, "model.onnx")
        tok_file = _get_hf_file(repo_id, "tokenizer.json")
        cfg_file = _get_hf_file(repo_id, "config.json")
        return model_file, tok_file, cfg_file

    if local_dir and str(local_dir).strip():
        dir_path = Path(local_dir).expanduser().resolve()
        if not dir_path.exists():
            raise FileNotFoundError(
                f"Model directory '{dir_path}' specified in QUERY_CLASSIFIER_MODEL_PATH or model_path does not exist."
            )
        model_file = dir_path / "model.onnx"
        tok_file = dir_path / "tokenizer.json"
        cfg_file = dir_path / "config.json"

        for file_path, name in [(model_file, "model.onnx"), (tok_file, "tokenizer.json")]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file '{name}' missing in '{dir_path}'")

        return model_file, tok_file, cfg_file

    # Default fallback: Hugging Face hub repository
    repo_id = DEFAULT_HF_REPO
    model_file = _get_hf_file(repo_id, "model.onnx")
    tok_file = _get_hf_file(repo_id, "tokenizer.json")
    cfg_file = _get_hf_file(repo_id, "config.json")
    return model_file, tok_file, cfg_file


class ONNXClassifierEngine:
    """
    Ultra-lightweight ONNX runtime engine using HuggingFace tokenizers + ONNXRuntime.
    No PyTorch or Transformers library dependencies.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        hf_repo: Optional[str] = None,
        providers: Optional[List[str]] = None,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
        max_length: Optional[int] = 512,
    ):
        model_file, tok_file, cfg_file = resolve_model_files(model_path=model_path, hf_repo=hf_repo)

        self.tokenizer = Tokenizer.from_file(str(tok_file))
        if max_length is not None and max_length > 0:
            self.tokenizer.enable_truncation(max_length=max_length)

        providers = providers or ["CPUExecutionProvider"]
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # Suppress ONNX runtime warnings

        if intra_op_num_threads is not None and intra_op_num_threads > 0:
            opts.intra_op_num_threads = intra_op_num_threads
        if inter_op_num_threads is not None and inter_op_num_threads > 0:
            opts.inter_op_num_threads = inter_op_num_threads

        self.session = ort.InferenceSession(str(model_file), sess_options=opts, providers=providers)

        # Inspect ONNX input requirement names
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        # Load label mapping from config.json if present
        self.id2label = DEFAULT_ID2LABEL
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    if "id2label" in cfg_data:
                        raw_map = cfg_data["id2label"]
                        self.id2label = {str(k): str(v) for k, v in raw_map.items()}
            except Exception:
                pass

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict_one(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        encoded = self.tokenizer.encode(text)

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, inputs)
        t1 = time.perf_counter()

        logits = outputs[0][0]
        probs = self._softmax(logits)
        pred_idx = int(np.argmax(probs))
        label = self.id2label.get(str(pred_idx), str(pred_idx))
        score = float(probs[pred_idx])

        all_scores = {
            self.id2label.get(str(idx), str(idx)): float(probs[idx])
            for idx in range(len(probs))
        }

        return {
            "label": label,
            "confidence": score,
            "latency_ms": (t1 - t0) * 1000.0,
            "scores": all_scores,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []

        t0 = time.perf_counter()
        enc_list = self.tokenizer.encode_batch(texts)

        max_len = max(len(enc.ids) for enc in enc_list)

        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, enc in enumerate(enc_list):
            input_ids[i, : len(enc.ids)] = enc.ids
            attention_mask[i, : len(enc.attention_mask)] = enc.attention_mask

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, inputs)
        t1 = time.perf_counter()

        batch_logits = outputs[0]
        batch_probs = self._softmax(batch_logits)
        batch_latency = (t1 - t0) * 1000.0 / len(texts)

        results = []
        for i in range(len(texts)):
            probs = batch_probs[i]
            pred_idx = int(np.argmax(probs))
            label = self.id2label.get(str(pred_idx), str(pred_idx))
            score = float(probs[pred_idx])
            all_scores = {
                self.id2label.get(str(idx), str(idx)): float(probs[idx])
                for idx in range(len(probs))
            }

            results.append(
                {
                    "label": label,
                    "confidence": score,
                    "latency_ms": batch_latency,
                    "scores": all_scores,
                }
            )

        return results
