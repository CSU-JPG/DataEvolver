"""Memory-augmented critic for semantic advantage analysis and strategy planning."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

Number = (int, float)


def _ensure_dir(p: Path) -> Path:
    """Create directory if missing and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


class Critic:
    """Memory-augmented critic producing semantic advantages and next-round strategies."""

    QUALITY_KEYS = [
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
    CLIP_KEYS = ["topic_avg", "subtopic_avg", "combined"]
    TEXT_KEYS = ["topic_similarity", "subtopic_similarity", "combined_similarity"]

    def __init__(
        self,
        cfg: Dict[str, Any],
        log_dir: Path,
        agents: Optional[Dict[str, Any]] = None,
        qa=None,
        semantic=None,
        text_checker=None,
    ):
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.agents = agents or {}
        self.qa = qa
        self.semantic = semantic
        self.text_checker = text_checker
        self.pending_subtopics: List[str] = []
        critic_cfg = cfg.get("critic", {})

        strategy_cfg = critic_cfg.get("strategy", {}) or {}
        self.queries_enabled = bool(
            strategy_cfg.get("queries_enabled", strategy_cfg.get("enabled", True))
        )

        experience_cfg = critic_cfg.get("experience", {}) or {}
        self.strategy_warmup_rounds = max(
            int(strategy_cfg.get("warmup_rounds", 3) or 3), 0
        )
        self.history_limit = max(int(strategy_cfg.get("history_limit", 30) or 30), 1)
        self.semantic_window = max(
            int(experience_cfg.get("semantic_window", 5) or 5), 1
        )
        self.top_query_limit = max(
            int(experience_cfg.get("topk", experience_cfg.get("top_k", 10) or 10)), 1
        )
        self.prompt_window = max(int(experience_cfg.get("prompt_window", 3) or 3), 1)
        self.strategy_agent = self.agents.get("StrategyPlannerAgent")
        self.experience_fallback_mode: str = str(
            experience_cfg.get("fallback_mode", "merge") or "merge"
        ).lower()
        self.query_agent = self.agents.get("QueryPlannerAgent")
        self.semantic_agent = self.agents.get("SemanticAdvantageAgent")
        self.experience_agent = self.agents.get("ExperienceLibrarianAgent")
        self.use_experience_agent = bool(
            (critic_cfg.get("experience") or {}).get("use_agent", True)
        )

        self.round_dir = _ensure_dir(self.log_dir / "RoundSnapshots")
        self.adv_dir = _ensure_dir(self.log_dir / "SemanticAdvantages")
        self.strategy_dir = _ensure_dir(self.log_dir / "Strategies")
        self.experience_dir = _ensure_dir(self.log_dir / "Experience")

        self.experience_file = self.experience_dir / "experience_library.json"
        self.strategy_history_file = self.strategy_dir / "history.json"

        self.experience = self._load_experience()
        self.round_history: List[Dict[str, Any]] = []
        self.strategy_history: List[Dict[str, Any]] = []
        self.latest_advantage: Optional[Dict[str, Any]] = None
        self.latest_strategy: Optional[Dict[str, Any]] = None
        self.completed_rounds: int = 0

        gen_cfg = cfg.get("generation", {}) or {}
        self.prompt_critic_enabled = bool(
            gen_cfg.get("prompt_critic_enabled", False)
        )
        self.max_regeneration_rounds = max(
            int(gen_cfg.get("max_regeneration_rounds", 1)), 1
        )
        self.prompt_critic_agent = self.agents.get("PromptCriticAgent")
        self.prompt_critic_log_path: Optional[Path] = None

    def init_prompt_critic_log(self, log_dir: Path) -> None:
        """Initialize the PromptCritic log directory and JSONL file path."""
        if not self.prompt_critic_enabled:
            return
        critic_log_dir = _ensure_dir(Path(log_dir) / "PromptCritic")
        self.prompt_critic_log_path = critic_log_dir / "regeneration_log.jsonl"
        print(
            f"[Critic] PromptCritic log: {self.prompt_critic_log_path}"
        )

    def optimize_failed_prompts(
        self, failures: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze each failed image, optimize its prompt."""
        if not self.prompt_critic_enabled or not self.prompt_critic_agent:
            return []

        optimized: List[Dict[str, Any]] = []
        for f in failures:
            topic = f.get("topic", "unknown")
            sub = f.get("subtopic", "unknown")
            original_prompt = f.get("prompt", "")
            if not original_prompt:
                continue

            context = {
                "original_prompt": original_prompt,
                "topic": topic,
                "subtopic": sub,
                "failure_stage": f.get("failure_stage", "unknown"),
                "failure_reasons": json.dumps(
                    f.get("failure_reasons", []), ensure_ascii=False
                ),
                "quality_details": json.dumps(
                    f.get("quality_details") or {}, ensure_ascii=False
                ),
                "semantic_details": json.dumps(
                    f.get("semantic_details") or {}, ensure_ascii=False
                ),
                "text_details": json.dumps(
                    f.get("text_details") or {}, ensure_ascii=False
                ),
            }

            optimized_prompt: Optional[str] = None
            try:
                resp = self.prompt_critic_agent.run_template(context)
                data = self._parse_json_object(resp)
                if isinstance(data, dict) and isinstance(
                    data.get("optimized_prompt"), str
                ):
                    optimized_prompt = data["optimized_prompt"].strip()
                    if optimized_prompt:
                        print(
                            f"[Critic] Prompt optimized: "
                            f"stage={f.get('failure_stage')} "
                            f"reasons={f.get('failure_reasons')}"
                        )
            except Exception as exc:
                print(f"[Critic] PromptCriticAgent failed: {exc}")

            log_entry = {
                "image_path": f.get("image_path", "unknown"),
                "topic": topic,
                "subtopic": sub,
                "original_prompt": original_prompt,
                "failure_stage": f.get("failure_stage"),
                "failure_reasons": f.get("failure_reasons", []),
                "quality_details": f.get("quality_details"),
                "semantic_details": f.get("semantic_details"),
                "text_details": f.get("text_details"),
                "optimized_prompt": optimized_prompt,
                "timestamp": time.time(),
            }
            if self.prompt_critic_log_path:
                try:
                    with open(
                        self.prompt_critic_log_path, "a", encoding="utf-8"
                    ) as lf:
                        lf.write(
                            json.dumps(log_entry, ensure_ascii=False) + "\n"
                        )
                except Exception as exc:
                    print(f"[Critic] Failed to write prompt-critic log: {exc}")

            if optimized_prompt:
                optimized.append(
                    {
                        "topic": topic,
                        "subtopic": sub,
                        "prompt": optimized_prompt,
                        "original_prompt": original_prompt,
                        "failure_reasons": f.get("failure_reasons", []),
                    }
                )

        print(
            f"[Critic] optimize_failed_prompts: "
            f"input={len(failures)} output={len(optimized)}"
        )
        return optimized

    def on_round(
        self,
        current_payload: Dict[str, Any],
        previous_payload: Optional[Dict[str, Any]] = None,
        current_path: Optional[Path] = None,
        previous_path: Optional[Path] = None,
    ):
        """Process one round of stats: generate semantic advantage, update experience library, plan strategy, and persist."""
        if not current_payload:
            return
        stats = self._extract_stats(current_payload, current_path)
        prev_stats = (
            self._extract_stats(previous_payload, previous_path)
            if previous_payload
            else None
        )

        round_record = {
            "round": stats.get("round"),
            "timestamp": stats.get("timestamp"),
            "rejection_counts": stats.get("rejection_counts") or {},
        }
        self.round_history.append(round_record)
        self.round_history = self.round_history[-self.history_limit :]

        a_text = self._generate_semantic_advantage(stats, prev_stats)
        adv_entry = {
            "round": stats.get("round"),
            "text": a_text,
            "timestamp": stats.get("timestamp"),
            "metrics": {
                "pass_rate": stats.get("pass_rate"),
                "ocr_conf": stats.get("ocr_conf"),
                "clip": stats.get("clip_avg"),
                "text": stats.get("text_avg"),
                "quality": stats.get("quality_avg"),
            },
        }
        self.latest_advantage = adv_entry
        self._persist_advantage(adv_entry)

        candidates = self._candidate_queries_from_stats(stats)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_exp = executor.submit(
                self._update_experience_library, adv_entry, candidates
            )
            future_strat = executor.submit(
                self._plan_strategy, adv_entry
            )
            self.experience = future_exp.result()
            strategy = future_strat.result()

        self._persist_experience()
        if strategy:
            self.latest_strategy = {
                "round": stats.get("round"),
                "timestamp": stats.get("timestamp"),
                "advantage": adv_entry,
                "strategy": strategy,
            }
            self.strategy_history.append(self.latest_strategy)
            self.strategy_history = self.strategy_history[-self.history_limit :]
            self._persist_strategy(self.latest_strategy)

        self.strategy_history = self.strategy_history[-self.prompt_window :]
        self.completed_rounds += 1

    def should_use_new_strategy(self) -> bool:
        """Return True once warmup rounds completed and a strategy exists."""
        return self.completed_rounds >= self.strategy_warmup_rounds and bool(
            self.latest_strategy
        )

    def should_use_query_strategy(self) -> bool:
        if not self.queries_enabled:
            return False
        return self.completed_rounds >= self.strategy_warmup_rounds and bool(
            self.latest_strategy
        )

    def get_strategy_context(self) -> Dict[str, Any]:
        return {
            "round_history": self.round_history[-self.history_limit :],
            "experience": self.experience,
            "strategy_history": self.strategy_history[-self.history_limit :],
            "latest": self.latest_strategy,
        }

    def _extract_stats(
        self, payload: Optional[Dict[str, Any]], path: Optional[Path]
    ) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        metrics = payload.get("metrics") or {}
        stats = {
            "round": payload.get("round_id"),
            "topic": payload.get("topic"),
            "subtopic": payload.get("subtopic"),
            "timestamp": payload.get("timestamp") or time.time(),
            "pass_rate": payload.get("pass_rate"),
            "processed": payload.get("processed"),
            "accepted": payload.get("accepted"),
            "rejected": payload.get("rejected"),
            "ocr_conf": metrics.get("avg_ocr_conf"),
            "quality_avg": metrics.get("quality") or {},
            "clip_avg": metrics.get("semantic") or {},
            "text_avg": metrics.get("text_consistency") or {},
            "rejection_vector": payload.get("rejection_vector") or [],
            "vector_labels": payload.get("vector_labels") or [],
            "rejection_counts": payload.get("rejection_reason_counts") or {},
            "queries": {
                "accepted": payload.get("accepted_queries") or [],
                "rejected": payload.get("rejected_queries") or [],
            },
            "round_details": payload.get("round_details") or {},
            "payload_path": str(path) if path else None,
        }
        return stats

    def _generate_semantic_advantage(
        self,
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
    ) -> str:
        context = {
            "current": self._compress_stats_for_prompt(current),
            "previous": self._compress_stats_for_prompt(previous) if previous else None,
        }
        if not self.semantic_agent:
            return self._fallback_advantage(context)

        current_counts = current.get("rejection_counts") or {}
        prev_counts = (previous.get("rejection_counts") or {}) if previous else {}
        kw_analysis = current.get("queries") or {}

        inject_data = {
            "current_rejection_counts": json.dumps(current_counts, ensure_ascii=False),
            "prev_rejection_counts": json.dumps(prev_counts, ensure_ascii=False),
            "keyword_analysis": json.dumps(kw_analysis, ensure_ascii=False),
        }

        try:
            text = self.semantic_agent.run_template(inject_data).strip()

            return text or self._fallback_advantage(context)
        except Exception as exc:
            print(f"[Critic] SemanticAdvantageAgent failed: {exc }")
            return self._fallback_advantage(context)

    def _update_experience_library(
        self, adv_entry: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        state = {
            "recent_advantages": (self.experience.get("recent_advantages") or [])
            + [self._library_advantage_entry(adv_entry)],
            "top_queries": self.experience.get("top_queries") or [],
        }
        state["recent_advantages"] = state["recent_advantages"][-self.semantic_window :]

        if not self.use_experience_agent:
            return self._experience_fallback_update(
                self.experience, adv_entry, candidates, state
            )

        latest_advantage_text = adv_entry.get("text")
        accepted_candidates = []
        for c in candidates:
            if c.get("status") == "accepted" and isinstance(c.get("query"), str):
                accepted_candidates.append(
                    {
                        "query": c.get("query").strip(),
                        "score": c.get("score"),
                    }
                )
        payload = {
            "latest_advantage_text": latest_advantage_text,
            "experience": state,
            "accepted_candidates": accepted_candidates,
        }
        if not self.experience_agent:
            state["top_queries"] = self._merge_top_queries(
                state["top_queries"], candidates
            )
            return state

        inject_data = {
            "current_a_text": latest_advantage_text or "",
            "recent_advantages": json.dumps(
                state["recent_advantages"], ensure_ascii=False
            ),
            "top_queries": json.dumps(state["top_queries"], ensure_ascii=False),
            "K": self.semantic_window,
        }

        try:
            resp = self.experience_agent.run_template(inject_data)

            data = self._parse_json_object(resp)

        except Exception as exc:
            print(f"[Critic] ExperienceLibrarianAgent failed: {exc }")
            data = None
        if not isinstance(data, dict):
            state["top_queries"] = self._merge_top_queries(
                state["top_queries"], candidates
            )
            return state
        recent = data.get("updated_recent_advantages") or state["recent_advantages"]
        top_q_raw = data.get("updated_top_queries")

        if not top_q_raw:

            fallback_result = self._experience_fallback_update(
                self.experience,
                adv_entry,
                candidates,
                {
                    "recent_advantages": recent[-self.semantic_window :],
                    "top_queries": state.get("top_queries", []),
                },
            )
            return fallback_result

        return {
            "recent_advantages": recent[-self.semantic_window :],
            "top_queries": self._trim_top_queries(top_q_raw),
        }

    def _experience_fallback_update(
        self,
        current_experience: Dict[str, Any],
        adv_entry: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        state_view: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return updated experience lib view per fallback mode (merge/keep/hardcoded)."""
        mode = self.experience_fallback_mode
        mode = mode if mode in ("merge", "keep", "hardcoded") else "merge"
        recent = state_view.get("recent_advantages") or (
            current_experience.get("recent_advantages") or []
        )
        recent = recent[-self.semantic_window :]
        if mode == "keep":
            top_q = state_view.get("top_queries") or (
                current_experience.get("top_queries") or []
            )
            return {
                "recent_advantages": recent,
                "top_queries": self._trim_top_queries(top_q),
            }
        elif mode == "hardcoded":
            try:
                from .critic_experience import HardcodedExperienceMaintainer

                maint = HardcodedExperienceMaintainer()
                per_query_pass: Dict[str, Any] = {}
                for c in candidates:
                    q = c.get("query")
                    if not isinstance(q, str) or not q.strip():
                        continue
                    status = c.get("status")
                    if q not in per_query_pass:
                        per_query_pass[q] = [0, 0]
                    if status == "accepted":
                        per_query_pass[q][0] += 1
                        per_query_pass[q][1] += 1
                    elif status == "rejected":
                        per_query_pass[q][1] += 1
                limits = {
                    "semantic_window": self.semantic_window,
                    "top_k": self.top_query_limit,
                }
                updated = maint.update(
                    current_experience, adv_entry, per_query_pass, limits
                )
                print(
                    f"[Critic] Fallback hardcoded maintainer used. queries_in={len (per_query_pass )}"
                )
                return {
                    "recent_advantages": (updated.get("recent_advantages") or recent)[
                        -self.semantic_window :
                    ],
                    "top_queries": self._trim_top_queries(
                        updated.get("top_queries") or []
                    ),
                }
            except Exception as exc:
                print(f"[Critic] HardcodedExperienceMaintainer fallback failed: {exc }")

                mode = "merge"

        merged = self._merge_top_queries(
            state_view.get("top_queries")
            or (current_experience.get("top_queries") or []),
            candidates,
        )
        return {
            "recent_advantages": recent,
            "top_queries": self._trim_top_queries(merged),
        }

    def _plan_strategy(
        self, adv_entry: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:

        trimmed_experience = self._slice_experience_for_prompt()

        queries_out: List[str] = []

        if self.query_agent:
            inject_data = {
                "pending_subtopics": json.dumps(
                    self.pending_subtopics[:5], ensure_ascii=False
                ),
                "top_queries": json.dumps(
                    trimmed_experience.get("top_queries"), ensure_ascii=False
                ),
                "recent_a_text": adv_entry.get("text") or "",
            }
            try:
                resp_q = self.query_agent.run_template(inject_data)

                data_q = self._parse_json_object(resp_q)

                if isinstance(data_q, dict) and isinstance(data_q.get("queries"), list):
                    queries_out = [
                        q for q in data_q.get("queries") if isinstance(q, str)
                    ]
            except Exception as exc:
                print(f"[Critic] QueryPlannerAgent failed: {exc }")

        if not queries_out:
            queries_out = [
                c.get("query")
                for c in self.experience.get("top_queries", [])[:3]
                if c.get("query")
            ]

        return {
            "queries": queries_out,
            "rationale": "QueryPlannerAgent for query generation.",
        }

    def _slice_experience_for_prompt(self) -> Dict[str, Any]:
        """Return a trimmed experience window for prompt construction."""
        exp = self.experience or {}
        recent = (exp.get("recent_advantages") or [])[-2:]
        top_q = (exp.get("top_queries") or [])[:7]

        return {
            "recent_advantages": recent,
            "top_queries": top_q[: self.top_query_limit],
        }

    def set_next_subtopics(self, subtopics: List[str]):
        """Cache the list of subtopics the pipeline is about to process."""
        if not isinstance(subtopics, list):
            return
        cleaned = [s for s in subtopics if isinstance(s, str) and s.strip()]
        self.pending_subtopics = cleaned[:5]

    def generate_queries_for(
        self, topic: str, subtopic: str
    ) -> List[str]:
        """Generate optimized search queries for a (topic, subtopic) pair."""
        if not self.should_use_new_strategy():
            return []

        base_text = (self.latest_advantage or {}).get(
            "text"
        ) or "No prior semantic summary available."

        current_pending = self.pending_subtopics
        if subtopic not in current_pending:
            current_pending = [subtopic] + current_pending

        payload_q = {
            "target_subtopic": subtopic,
            "target_topic": topic,
            "A_text": base_text,
            "experience": self.experience,
            "pending_subtopics": current_pending[:10],
        }

        queries_out: List[str] = []

        if self.query_agent:
            inject_data = {
                "pending_subtopics": json.dumps(
                    current_pending[:10], ensure_ascii=False
                ),
                "top_queries": json.dumps(
                    self.experience.get("top_queries", [])[: self.top_query_limit],
                    ensure_ascii=False,
                ),
                "recent_a_text": base_text,
            }
            try:
                resp_q = self.query_agent.run_template(inject_data)
                data_q = self._parse_json_object(resp_q)
                if isinstance(data_q, dict) and isinstance(data_q.get("queries"), list):
                    queries_out = [
                        q for q in data_q.get("queries") if isinstance(q, str)
                    ]
            except Exception as exc:
                print(f"[Critic] QueryPlannerAgent (adhoc) failed: {exc }")

        if not queries_out:
            queries_out = [
                c.get("query")
                for c in self.experience.get("top_queries", [])[:3]
                if c.get("query")
            ]

        if queries_out:
            uniq = []
            seen = set()
            for q in queries_out:
                if not isinstance(q, str):
                    continue
                qq = q.strip()
                if not qq or qq in seen:
                    continue
                seen.add(qq)
                uniq.append(qq)
            return uniq
        return []

    def _persist_round_snapshot(self, record: Dict[str, Any]):
        round_id = record.get("round") or "unknown"
        ts = record.get("timestamp") or time.time()
        ts_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts))
        path = self.round_dir / f"round_{round_id }_{ts_str }.json"
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(record, fw, ensure_ascii=False, indent=2)

    def _persist_advantage(self, entry: Dict[str, Any]):
        round_id = entry.get("round") or "unknown"
        path = self.adv_dir / f"A_text_round_{round_id }.json"
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(entry, fw, ensure_ascii=False, indent=2)

    def _persist_strategy(self, entry: Dict[str, Any]):
        round_id = entry.get("round") or "unknown"
        path = self.strategy_dir / f"strategy_round_{round_id }.json"
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(entry, fw, ensure_ascii=False, indent=2)
        with open(self.strategy_history_file, "w", encoding="utf-8") as fw:
            json.dump(self.strategy_history, fw, ensure_ascii=False, indent=2)

    def _persist_experience(self):
        payload = {
            "updated_ts": time.time(),
            "semantic_window": self.semantic_window,
            "top_query_limit": self.top_query_limit,
            "experience": self._slice_experience_for_prompt(),
        }
        with open(self.experience_file, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False, indent=2)

    def _load_experience(self) -> Dict[str, Any]:
        if not self.experience_file.exists():
            return {"recent_advantages": [], "top_queries": []}
        try:
            with open(self.experience_file, "r", encoding="utf-8") as fr:
                data = json.load(fr)
            exp = data.get("experience") or {}
            exp.setdefault("recent_advantages", [])
            exp.setdefault("top_queries", [])
            return exp
        except Exception as exc:
            print(f"[Critic] Failed to load experience file: {exc }")
            return {"recent_advantages": [], "top_queries": []}

    def _compress_stats_for_prompt(
        self, stats: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not stats:
            return None
        return {
            "round": stats.get("round"),
            "pass_rate": stats.get("pass_rate"),
            "ocr_conf": stats.get("ocr_conf"),
            "quality": {
                k: stats.get("quality_avg", {}).get(k)
                for k in self.QUALITY_KEYS
                if stats.get("quality_avg", {}).get(k) is not None
            },
            "clip": {
                k: stats.get("clip_avg", {}).get(k)
                for k in self.CLIP_KEYS
                if stats.get("clip_avg", {}).get(k) is not None
            },
            "text": {
                k: stats.get("text_avg", {}).get(k)
                for k in self.TEXT_KEYS
                if stats.get("text_avg", {}).get(k) is not None
            },
            "rejection_counts": stats.get("rejection_counts"),
            "queries": stats.get("queries"),
        }

    def _fallback_advantage(self, context: Dict[str, Any]) -> str:
        cur = context.get("current") or {}
        prev = context.get("previous") or {}
        pr_cur = cur.get("pass_rate")
        pr_prev = prev.get("pass_rate")
        trend = ""
        if isinstance(pr_cur, Number) and isinstance(pr_prev, Number):
            delta = pr_cur - pr_prev
            sign = "up" if delta >= 0 else "down"
            trend = f"Pass rate {sign } {abs (delta ):.3f}; "
        return (
            f"{trend }OCR avg={cur .get ('ocr_conf')}; CLIP={cur .get ('clip')}; "
            f"Text consistency={cur .get ('text')}. "
            "Suggestion: raise clarity thresholds, emphasise high resolution, "
            "topic keywords, and readable text."
        )

    def _library_advantage_entry(self, adv_entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "round": adv_entry.get("round"),
            "text": adv_entry.get("text"),
            "pass_rate": adv_entry.get("metrics", {}).get("pass_rate"),
            "timestamp": adv_entry.get("timestamp"),
        }

    def _candidate_queries_from_stats(
        self, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        pass_rate = stats.get("pass_rate")
        for q in stats.get("queries", {}).get("accepted", []):
            candidates.append(
                {
                    "query": q,
                    "round": stats.get("round"),
                    "score": pass_rate,
                    "status": "accepted",
                }
            )
        for q in stats.get("queries", {}).get("rejected", []):
            candidates.append(
                {
                    "query": q,
                    "round": stats.get("round"),
                    "score": pass_rate,
                    "status": "rejected",
                }
            )
        return candidates

    def _merge_top_queries(
        self, existing: List[Dict[str, Any]], candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        cache: Dict[str, Dict[str, Any]] = {}
        for item in existing:
            if isinstance(item, dict) and item.get("query"):
                cache[item["query"]] = dict(item)
        for cand in candidates:
            q = cand.get("query")
            if not q or not isinstance(q, str):
                continue
            entry = cache.get(q, {"query": q})
            score = cand.get("score")
            if isinstance(score, Number):
                prev = entry.get("score")
                if prev is None or score > prev:
                    entry["score"] = score
            entry["round"] = cand.get("round")
            status = cand.get("status")
            if status:
                entry["status"] = status
            cache[q] = entry
        return list(cache.values())

    def _trim_top_queries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean, deduplicate, sort by score, and truncate to top_k."""
        valid_entries: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for e in entries:
            if not isinstance(e, dict):
                continue
            q = e.get("query")
            if not isinstance(q, str):
                continue
            q_clean = q.strip()
            if not q_clean or q_clean in seen:
                continue
            seen.add(q_clean)
            if not isinstance(e.get("score"), Number):
                e["score"] = 0.0
            valid_entries.append(e)
        valid_entries.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return valid_entries[: self.top_query_limit]

    def _parse_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    return None
            return None


__all__ = ["Critic"]
