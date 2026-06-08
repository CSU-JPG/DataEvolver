"""Rejection statistics tracking and async rate limiting."""

from __future__ import annotations

import asyncio
import json
import pathlib
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.critic import Critic
from src.utils.common import ensure_dir


class RejectTracker:
    """Lightweight in-memory rejection tracker for pipeline use."""

    def __init__(self):
        self.by_reason: Counter = Counter()
        self.examples: defaultdict = defaultdict(list)
        self.total: int = 0
        self.accepted: int = 0

    def add(self, image_path: str, reasons: List[str]) -> None:
        """Record an image's acceptance or rejection."""
        self.total += 1
        if not reasons:
            self.accepted += 1
            return
        for r in reasons:
            self.by_reason[r] += 1
            if len(self.examples[r]) < 6:
                self.examples[r].append(image_path)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics for checked, accepted, and rejected items."""
        rej = self.total - self.accepted
        return {
            "checked": self.total,
            "accepted": self.accepted,
            "rejected": rej,
            "accept_rate": (self.accepted / self.total) if self.total else 0.0,
            "top_reasons": self.by_reason.most_common(10),
        }


class RejectionVectorRoundTracker:
    """Accumulates per-round rejection vectors and averages for critic feedback."""

    QUALITY_METRICS = [
        "short_edge",
        "sobel_density",
        "blank_ratio",
        "coverage",
        "legibility",
        "words",
        "sharpness",
        "contrast",
        "char_h_mean",
        "char_h_cv",
        "line_angle_std",
        "clutter_index",
        "centrality",
        "border_ratio",
        "char_density",
        "text_density",
    ]
    QUALITY_CANONICAL = [
        "quality_error",
        "quality_resolution_lt",
        "quality_complexity_lt",
        "quality_blank_ratio_gt",
        "quality_coverage_lt",
        "quality_legibility_lt",
        "quality_words_lt",
        "quality_sharpness_lt",
        "quality_char_h_mean_lt",
        "quality_char_h_cv_gt",
        "quality_line_angle_std_gt",
        "quality_centrality_lt",
        "quality_border_ratio_gt",
        "quality_char_density_lt",
        "quality_text_density_lt",
        "quality_clutter_gt",
        "quality_contrast_lt",
        "quality_lang_not_allowed",
    ]

    def __init__(
        self,
        cfg: Dict[str, Any],
        output_dir: pathlib.Path,
        critic: Optional[Critic] = None,
        event: Optional[asyncio.Event] = None,
    ):
        self.enabled = bool(cfg.get("enabled", True))
        self.round_size = max(int(cfg.get("round_size", 1000)), 1)
        self.output_dir = ensure_dir(output_dir)
        self.round_index = 1
        self.critic = critic
        self.round_completed_event = event
        self._last_payload: Optional[Dict[str, Any]] = None
        self._last_path: Optional[pathlib.Path] = None
        self.last_round_details: Dict[str, Any] = {}
        self.vector_labels = (
            ["ocr_failure", "dedup_failure"]
            + self.QUALITY_CANONICAL
            + [
                "semantic_topic_fail",
                "semantic_subtopic_fail",
                "semantic_combined_fail",
                "semantic_error",
            ]
            + [
                "text_topic_fail",
                "text_subtopic_fail",
                "text_combined_fail",
                "text_error",
            ]
            + ["total_rejection_reason_count"]
        )
        self._label_to_index = {
            label: idx for idx, label in enumerate(self.vector_labels[:-1])
        }
        self._quality_reason_map = {
            "quality_error": "quality_error",
            "legacy_reject": "quality_error",
            "quality_reject": "quality_error",
            "resolution_lt": "quality_resolution_lt",
            "complexity_lt": "quality_complexity_lt",
            "blank_ratio_gt": "quality_blank_ratio_gt",
            "coverage_lt": "quality_coverage_lt",
            "legibility_lt": "quality_legibility_lt",
            "words_lt": "quality_words_lt",
            "sharpness_lt": "quality_sharpness_lt",
            "char_h_mean_lt": "quality_char_h_mean_lt",
            "char_h_cv_gt": "quality_char_h_cv_gt",
            "line_angle_std_gt": "quality_line_angle_std_gt",
            "centrality_lt": "quality_centrality_lt",
            "border_ratio_gt": "quality_border_ratio_gt",
            "char_density_lt": "quality_char_density_lt",
            "text_density_lt": "quality_text_density_lt",
            "clutter_gt": "quality_clutter_gt",
            "contrast_lt": "quality_contrast_lt",
            "lang_not_allowed": "quality_lang_not_allowed",
        }
        self.reset()

    def reset(self) -> None:
        """Reset per-round accumulation counters."""
        self.processed = 0
        self.accepted = 0
        self.rejected = 0
        self._rejection_lists: List[List[str]] = []
        self._reason_total = 0
        self._reason_counts: Counter = Counter()
        self.ocr_conf_sum = 0.0
        self.ocr_conf_count = 0
        self.quality_sums = {k: 0.0 for k in self.QUALITY_METRICS}
        self.quality_counts = {k: 0 for k in self.QUALITY_METRICS}
        self.semantic_sums = {"topic_avg": 0.0, "subtopic_avg": 0.0, "combined": 0.0}
        self.semantic_counts = {k: 0 for k in self.semantic_sums}
        self.text_sums = {
            "topic_similarity": 0.0,
            "subtopic_similarity": 0.0,
            "combined_similarity": 0.0,
        }
        self.text_counts = {k: 0 for k in self.text_sums}
        self.accepted_details: List[Dict[str, Any]] = []
        self.rejected_details: List[Dict[str, Any]] = []

    def record(self, sample: Dict[str, Any], reasons: List[str]) -> None:
        """Record a single sample outcome, accumulating metrics and reasons."""
        if not self.enabled:
            return
        reasons = list(reasons or [])
        self.processed += 1
        if reasons:
            self.rejected += 1
            canonical = [
                r for r in (self._canonicalize_reason(rn) for rn in reasons) if r
            ]
            self._rejection_lists.append(canonical or ["quality_error"])
            self._reason_total += len(canonical or ["quality_error"])
            for label in canonical or ["quality_error"]:
                self._reason_counts[label] += 1
            self.rejected_details.append(
                {
                    "topic": sample.get("topic", ""),
                    "subtopic": sample.get("subtopic", ""),
                    "query": sample.get("query", ""),
                    "reasons": reasons,
                }
            )
        else:
            self.accepted += 1
            self.accepted_details.append(
                {
                    "topic": sample.get("topic", ""),
                    "subtopic": sample.get("subtopic", ""),
                    "query": sample.get("query", ""),
                }
            )
        self._accumulate_metrics(sample or {})
        if self.processed >= self.round_size:
            self._persist_round()

    def finalize(self) -> None:
        """Persist the current round if any samples have been recorded."""
        if not self.enabled:
            return
        if self.processed:
            self._persist_round()

    def _accumulate_metrics(self, sample: Dict[str, Any]) -> None:
        """Sum per-sample metrics for later averaging."""
        ocr = sample.get("ocr") or {}
        avg_conf = ocr.get("avg_conf")
        if isinstance(avg_conf, (int, float)):
            self.ocr_conf_sum += float(avg_conf)
            self.ocr_conf_count += 1
        q = sample.get("quality") or {}
        for name in self.QUALITY_METRICS:
            if name == "density":
                val = q.get("char_density")
                if val is None:
                    val = q.get("text_density")
            else:
                val = q.get(name)
            if isinstance(val, (int, float)):
                self.quality_sums[name] += float(val)
                self.quality_counts[name] += 1
        sem = sample.get("semantic") or {}
        for key in self.semantic_sums:
            val = sem.get(key)
            if isinstance(val, (int, float)):
                self.semantic_sums[key] += float(val)
                self.semantic_counts[key] += 1
        tc = sample.get("text_consistency") or {}
        for key in self.text_sums:
            val = tc.get(key)
            if isinstance(val, (int, float)):
                self.text_sums[key] += float(val)
                self.text_counts[key] += 1

    def _canonicalize_reason(self, reason: str) -> Optional[str]:
        """Map a raw rejection reason to its canonical label."""
        if not reason:
            return None
        if reason.startswith("ocr"):
            return "ocr_failure"
        if reason.startswith("dedup"):
            return "dedup_failure"
        if reason in self._quality_reason_map:
            return self._quality_reason_map[reason]
        if reason.startswith("quality"):
            return "quality_error"
        if reason.startswith("semantic_topic"):
            return "semantic_topic_fail"
        if reason.startswith("semantic_subtopic"):
            return "semantic_subtopic_fail"
        if reason.startswith("semantic_combined"):
            return "semantic_combined_fail"
        if reason.startswith("semantic"):
            return "semantic_error"
        if reason.startswith("text_topic"):
            return "text_topic_fail"
        if reason.startswith("text_subtopic"):
            return "text_subtopic_fail"
        if reason.startswith("text_combined"):
            return "text_combined_fail"
        if reason.startswith("text"):
            return "text_error"
        return None

    def _build_vector(self) -> List[float]:
        """Build the per-label rejection vector for this round."""
        slots = [0.0 for _ in range(len(self.vector_labels) - 1)]
        if self.rejected > 0:
            base = 100.0 / self.rejected
            for rlist in self._rejection_lists:
                count = max(len(rlist), 1)
                weight = base / count
                for reason in rlist:
                    idx = self._label_to_index.get(reason)
                    if idx is None:
                        continue
                    slots[idx] += weight
        return [round(v, 6) for v in slots]

    def _metrics_payload(self) -> Dict[str, Any]:
        """Compute average metrics across all samples in this round."""
        quality_avg = {}
        for name in self.QUALITY_METRICS:
            cnt = self.quality_counts[name]
            quality_avg[name] = round(self.quality_sums[name] / cnt, 6) if cnt else 0.0
        semantic_avg = {}
        for key in self.semantic_sums:
            cnt = self.semantic_counts[key]
            semantic_avg[key] = round(self.semantic_sums[key] / cnt, 6) if cnt else 0.0
        text_avg = {}
        for key in self.text_sums:
            cnt = self.text_counts[key]
            text_avg[key] = round(self.text_sums[key] / cnt, 6) if cnt else 0.0
        return {
            "avg_ocr_conf": (
                round(self.ocr_conf_sum / self.ocr_conf_count, 6)
                if self.ocr_conf_count
                else 0.0
            ),
            "quality": quality_avg,
            "semantic": semantic_avg,
            "text_consistency": text_avg,
        }

    def _persist_round(self) -> None:
        """Finalise the current round: build vector, call critic, persist JSON."""
        vector = self._build_vector()
        vector.append(self._reason_total)
        pass_rate = (self.accepted / self.processed) if self.processed else 0.0

        current_topic = "unknown"
        current_subtopic = "unknown"
        all_details = self.accepted_details + self.rejected_details
        if all_details:
            first = all_details[0]
            current_topic = first.get("topic", "unknown")
            current_subtopic = first.get("subtopic", "unknown")

        payload = {
            "round_id": self.round_index,
            "round_size": self.round_size,
            "processed": self.processed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "pass_rate": round(pass_rate, 6),
            "rejection_vector": vector,
            "vector_labels": self.vector_labels,
            "metrics": self._metrics_payload(),
            "timestamp": time.time(),
            "rejection_reason_counts": dict(self._reason_counts),
            "accepted_queries": sorted(
                list(set(x["query"] for x in self.accepted_details if x.get("query")))
            ),
            "rejected_queries": sorted(
                list(set(x["query"] for x in self.rejected_details if x.get("query")))
            ),
        }
        payload["round_details"] = {
            "accepted_samples": list(self.accepted_details),
            "rejected_samples": list(self.rejected_details),
        }
        self.last_round_details = {
            "round_index": self.round_index,
            "accepted": self.accepted_details,
            "rejected": self.rejected_details,
        }

        if self.critic:
            try:
                self.critic.on_round(payload, self._last_payload, None, self._last_path)
            except Exception as e:
                print(f"[Critic] on_round error: {e }")

        try:
            reason_counts = payload.get("rejection_reason_counts", {}) or {}
            reason_top = sorted(
                reason_counts.items(), key=lambda x: x[1], reverse=True
            )[:7]
            print("====================================")
            print("= Top-7 Rejection Reasons:")
            for name, cnt in reason_top:
                print(f"= {name }: {cnt }")
            print("====================================")
        except Exception as exc:
            print(f"[RejectionVector] emphasis log failed: {exc }")

        ts = time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime(payload["timestamp"]))
        filename = self.output_dir / f"{self .round_index }_{ts }.json"
        with open(filename, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False, indent=2)
        print(
            f"[RejectionVector] saved {filename .name } processed={self .processed } rejected={self .rejected }"
        )

        if self.round_completed_event:
            self.round_completed_event.set()

        self._last_payload = payload
        self._last_path = filename
        self.round_index += 1
        self.reset()


class AsyncTokenBucket:
    """Async token bucket for QPS rate limiting."""
    def __init__(self, rate: float, capacity: Optional[float] = None):
        self.rate = max(rate, 0.0)
        self.capacity = capacity if capacity is not None else self.rate
        self.tokens = self.capacity
        self.updated = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, n: float = 1.0) -> None:
        """Acquire *n* tokens, waiting if necessary."""
        if self.rate <= 0:
            return
        while True:
            async with self._lock:
                now = time.time()
                elapsed = now - self.updated
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    self.updated = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait_time = (n - self.tokens) / self.rate if self.rate > 0 else 0.05
            await asyncio.sleep(min(wait_time, 1.0))
