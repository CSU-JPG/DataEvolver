"""CLIP-based semantic relevance checker with JSON-lines logging."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

__all__ = ["SemanticChecker"]


class SemanticChecker:
    """Semantic relevance checker powered by CLIP."""
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        semantic_config = config.get("semantic", {})

        self.enabled = semantic_config.get("enabled", True)
        self.device = semantic_config.get("device", "cpu")
        self.verbose = semantic_config.get("verbose", False)
        self.model_name = semantic_config.get(
            "model_name", "openai/clip-vit-base-patch32"
        )

        thresholds = semantic_config.get("thresholds", {})
        self.min_similarity = float(thresholds.get("min_similarity", 0.4))
        self.min_topic_similarity = float(thresholds.get("min_topic_similarity", 0.4))
        self.min_subtopic_similarity = float(
            thresholds.get("min_subtopic_similarity", 0.45)
        )
        self.min_prompt_similarity = float(thresholds.get("min_prompt_similarity", 0.4))
        self.min_combined_similarity = float(
            thresholds.get("min_combined_similarity", 0.4)
        )

        log_dir = config.get("paths", {}).get("log_dir", "/tmp")
        self.log_path = os.path.join(log_dir, "clip.json")

        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None

        if self.enabled:
            try:
                if self.verbose:
                    print(f"[SemanticChecker] Loading model: " f"{self .model_name }")
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                if self.verbose:
                    print("[SemanticChecker] Model loaded successfully.")
            except Exception as e:
                print(f"[SemanticChecker] Failed to load model: {e }")
                self.enabled = False

    def is_available(self) -> bool:
        """Return ``True`` if the checker is ready to use."""
        return self.enabled and self.model is not None

    def compute_similarities(
        self, image_path: str, topic: str, subtopic: str
    ) -> Dict[str, Any]:
        """Compute raw similarity scores without making accept/reject call."""
        try:
            image = Image.open(image_path).convert("RGB")

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

            similarities = self._compute_similarity(image, prompts)

            min_similarity = float(similarities.min())
            topic_similarities = [similarities[i] for i in range(0, 3)]
            subtopic_similarities = [similarities[i] for i in range(3, 6)]
            combined_similarities = [similarities[i] for i in range(6, 8)]

            topic_avg = float(sum(topic_similarities) / len(topic_similarities))
            subtopic_avg = float(
                sum(subtopic_similarities) / len(subtopic_similarities)
            )
            combined_sim = float(
                sum(combined_similarities) / len(combined_similarities)
            )

            return {
                "topic_avg": topic_avg,
                "subtopic_avg": subtopic_avg,
                "combined": combined_sim,
                "min_similarity": min_similarity,
                "similarities": {p: float(s) for p, s in zip(prompts, similarities)},
            }

        except Exception as e:
            print(
                f"[ERROR] Semantic similarity computation failed for "
                f"{image_path }: {e }"
            )
            traceback.print_exc()
            return {"error": str(e), "min_similarity": 0.0}

    def check_relevance(
        self,
        image_path: str,
        topic: str,
        subtopic: str,
        gen_prompts: Optional[List[str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check whether an image is semantically relevant."""
        self.refresh_thresholds()

        if not self.is_available():
            return True, {
                "min_similarity": 1.0,
                "best_prompt": "semantic_checker_unavailable",
            }

        try:
            image = Image.open(image_path).convert("RGB")

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
            if gen_prompts:
                prompts.extend(gen_prompts)

            similarities = self._compute_similarity(image, prompts)
            min_similarity = float(similarities.min())

            topic_similarities = [similarities[i] for i in range(0, 3)]
            subtopic_similarities = [similarities[i] for i in range(3, 6)]
            combined_similarities = [similarities[i] for i in range(6, 8)]

            gen_prompt_sim = None
            if gen_prompts:
                gen_vals = [similarities[i] for i in range(9, 10)]
                gen_prompt_sim = float(sum(gen_vals) / len(gen_vals))

            topic_avg = float(sum(topic_similarities) / len(topic_similarities))
            subtopic_avg = float(
                sum(subtopic_similarities) / len(subtopic_similarities)
            )
            combined_sim = float(
                sum(combined_similarities) / len(combined_similarities)
            )

            sim_dict: Dict[str, Any] = {
                "topic_avg": topic_avg,
                "subtopic_avg": subtopic_avg,
                "combined": combined_sim,
                "min_similarity": min_similarity,
                "similarities": {p: float(s) for p, s in zip(prompts, similarities)},
            }
            if gen_prompt_sim is not None:
                sim_dict["gen_prompt_similarities"] = gen_prompt_sim

            rejection_reasons: List[str] = []
            if topic_avg < self.min_topic_similarity:
                rejection_reasons.append("semantic_topic_fail")
            if subtopic_avg < self.min_subtopic_similarity:
                rejection_reasons.append("semantic_subtopic_fail")
            if combined_sim < self.min_combined_similarity:
                rejection_reasons.append("semantic_combined_fail")

            is_relevant = len(rejection_reasons) == 0
            sim_dict["rejection_reasons"] = rejection_reasons

            self._log_result(image_path, topic, subtopic, sim_dict, is_relevant)

            return is_relevant, sim_dict

        except Exception as e:
            print(f"[ERROR] Semantic check failed for {image_path }: {e }")
            traceback.print_exc()
            return False, {
                "rejection_reasons": ["semantic_error"],
                "error": str(e),
            }

    def refresh_thresholds(self, override: Optional[Dict[str, Any]] = None) -> None:
        """Update thresholds dynamically (e.g. from policy overrides)."""
        src = (
            override or (self.cfg.get("semantic", {}) or {}).get("thresholds", {}) or {}
        )
        self.min_similarity = float(src.get("min_similarity", self.min_similarity))
        self.min_topic_similarity = float(
            src.get("min_topic_similarity", self.min_topic_similarity)
        )
        self.min_subtopic_similarity = float(
            src.get("min_subtopic_similarity", self.min_subtopic_similarity)
        )
        self.min_combined_similarity = float(
            src.get("min_combined_similarity", self.min_combined_similarity)
        )
        self.min_prompt_similarity = float(
            src.get("min_prompt_similarity", self.min_prompt_similarity)
        )

    def _compute_similarity(
        self,
        image: Image.Image,
        texts: List[str],
    ) -> Any:
        """Compute cosine similarity between image and text embeddings."""
        with torch.no_grad():
            img_inputs = self.processor(images=image, return_tensors="pt").to(
                self.device
            )
            img_feat = self.model.get_image_features(**img_inputs)

            txt_inputs = self.processor(
                text=texts, return_tensors="pt", padding=True
            ).to(self.device)
            txt_feat = self.model.get_text_features(**txt_inputs)

            img_feat = F.normalize(img_feat, p=2, dim=-1)
            txt_feat = F.normalize(txt_feat, p=2, dim=-1)

            sims = (img_feat @ txt_feat.T).squeeze(0)
            sims = ((sims + 1) / 2).clamp(0, 1)

            return sims.cpu().numpy()

    def _log_result(
        self,
        image_path: str,
        topic: str,
        subtopic: str,
        results: Dict[str, Any],
        is_relevant: bool,
    ) -> None:
        """Append a JSON-lines entry to the CLIP log file."""
        try:
            log_entry = {
                "image_path": image_path,
                "topic": topic,
                "subtopic": subtopic,
                "is_relevant": is_relevant,
                "clip_results": results,
                "timestamp": time.time(),
            }
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARNING] Failed to log CLIP result: {e }")
