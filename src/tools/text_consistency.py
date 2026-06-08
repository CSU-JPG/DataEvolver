"""Text-consistency checker via Sentence-BERT with topic/subtopic prompt comparison."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import torch

__all__ = ["TextConsistencyChecker"]

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except Exception:
    _HAS_ST = False

_MODEL_SINGLETON: Any = None
_MODEL_LOCK = threading.Lock()
_PROMPT_CACHE: Dict[str, torch.Tensor] = {}
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str) -> Any:
    """Load (or retrieve cached) Sentence-Transformer model."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON
    with _MODEL_LOCK:
        if _MODEL_SINGLETON is None:
            _MODEL_SINGLETON = SentenceTransformer(model_name, device=_DEVICE)
            if _DEVICE.startswith("cuda"):
                try:
                    _MODEL_SINGLETON = _MODEL_SINGLETON.half()
                except Exception:
                    pass
    return _MODEL_SINGLETON


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scalar cosine similarity between two tensors."""
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).item())


class TextConsistencyChecker:
    """Semantic text-consistency checker for OCR + topic/subtopic pairs."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg or {}
        self.tc_cfg = self.cfg.get("text_consistency") or {}

        self.enabled = bool(self.tc_cfg.get("enabled", True)) and _HAS_ST
        self.model_name = self.tc_cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.top_k = int(self.tc_cfg.get("top_k_ocr", 10))
        self.th_topic = float(self.tc_cfg.get("min_topic_similarity", 0.55))
        self.th_subtopic = float(self.tc_cfg.get("min_subtopic_similarity", 0.60))
        self.th_combined = float(self.tc_cfg.get("min_combined_similarity", 0.58))
        self.device = _DEVICE
        self.cache_prompts = True

        if not _HAS_ST:
            print(
                "[TextConsistency] sentence-transformers not installed; "
                "feature disabled.  pip install sentence-transformers"
            )

    def is_available(self) -> bool:
        """Return ``True`` when the checker is ready."""
        return self.enabled

    def refresh_thresholds(self, override: Optional[Dict[str, Any]] = None) -> None:
        """Update thresholds dynamically from a policy override dict."""
        src = override or (self.cfg.get("text_consistency") or {}) or {}
        self.th_topic = float(src.get("min_topic_similarity", self.th_topic))
        self.th_subtopic = float(src.get("min_subtopic_similarity", self.th_subtopic))
        self.th_combined = float(src.get("min_combined_similarity", self.th_combined))

    def check(
        self,
        ocr_result: Dict[str, Any],
        topic: str,
        subtopic: str,
    ) -> Dict[str, Any]:
        """Run text-consistency check and return a structured dict."""
        self.refresh_thresholds()

        thresholds = {
            "topic": self.th_topic,
            "subtopic": self.th_subtopic,
            "combined": self.th_combined,
        }

        if not self.enabled:
            return {
                "passed": True,
                "topic_similarity": 0.0,
                "subtopic_similarity": 0.0,
                "combined_similarity": 0.0,
                "disabled": True,
                "thresholds": thresholds,
                "ocr_lines_used": [],
            }

        lines = (ocr_result or {}).get("texts") or []
        confs = (ocr_result or {}).get("confs") or []

        if not lines:
            return {
                "passed": False,
                "reason": "text_consistency_no_ocr_text",
                "topic_similarity": 0.0,
                "subtopic_similarity": 0.0,
                "combined_similarity": 0.0,
                "thresholds": thresholds,
                "ocr_lines_used": [],
            }

        ranked = list(zip(lines, confs)) if confs else [(t, 0.0) for t in lines]
        ranked.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in ranked[: self.top_k]]
        ocr_concat = " ".join(selected)[:2000]

        prompts = [
            f"An image about {topic }",
            f"A photo related to {topic }",
            f"A picture showing {topic }",
            f"An image about {subtopic }",
            f"A photo related to {subtopic }",
            f"A picture showing {subtopic }",
            f"A picture of {subtopic } in {topic }",
            f"A {topic } image about {subtopic }",
        ]

        try:
            model = _load_model(self.model_name)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                enabled=self.device.startswith("cuda"),
            ):
                prompt_embeds = self._get_prompt_embeds(model, prompts)
                ocr_embed = model.encode(
                    ocr_concat,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
        except Exception as e:
            return {
                "passed": False,
                "reason": f"text_consistency_model_error:{e }",
                "topic_similarity": 0.0,
                "subtopic_similarity": 0.0,
                "combined_similarity": 0.0,
                "thresholds": thresholds,
                "ocr_lines_used": selected,
            }

        sims = []
        for i in range(len(prompts)):
            sims.append(
                _cosine(
                    ocr_embed.unsqueeze(0),
                    prompt_embeds[i].unsqueeze(0),
                )
            )

        topic_sims = sims[:3]
        subtopic_sims = sims[3:6]
        combined_sims = sims[6:]

        topic_max = max(topic_sims) if topic_sims else 0.0
        subtopic_max = max(subtopic_sims) if subtopic_sims else 0.0
        combined_max = max(combined_sims) if combined_sims else 0.0

        failed_dims = []
        if topic_max < self.th_topic:
            failed_dims.append("topic")
        if subtopic_max < self.th_subtopic:
            failed_dims.append("subtopic")
        if combined_max < self.th_combined:
            failed_dims.append("combined")

        passed = len(failed_dims) == 0
        reasons = [f"text_{d }_fail" for d in failed_dims]

        return {
            "passed": bool(passed),
            "topic_similarity": float(topic_max),
            "subtopic_similarity": float(subtopic_max),
            "combined_similarity": float(combined_max),
            "failed_dims": failed_dims,
            "rejection_reasons": reasons if not passed else [],
            "thresholds": thresholds,
        }

    def _get_prompt_embeds(self, model: Any, prompts: List[str]) -> torch.Tensor:
        """Encode prompts (cached by concatenation key)."""
        key = "|".join(prompts)
        if self.cache_prompts and key in _PROMPT_CACHE:
            return _PROMPT_CACHE[key]
        embeds = model.encode(
            prompts, convert_to_tensor=True, normalize_embeddings=True
        )
        if self.cache_prompts:
            _PROMPT_CACHE[key] = embeds
        return embeds
