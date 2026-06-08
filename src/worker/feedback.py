"""Monitors round completion and dynamically generates queries for remaining subtopics."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING

from src.worker.query_gen import generate_queries

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def feedback_loop(pipeline: OrchestratorPipeline) -> None:
    """Listen for round-completion events and dynamically generate queries for unprocessed subtopics."""
    critic_strategy = (pipeline.cfg.get("critic", {}) or {}).get("strategy", {}) or {}
    warmup_rounds = int(critic_strategy.get("warmup_rounds", 2) or 2)
    fb_cfg = pipeline.cfg.get("feedback_query", {}) or {}
    per_round_limit = int(fb_cfg.get("max_llm_calls_per_loop", 5) or 5)
    dq_cfg = pipeline.cfg.get("dynamic_query", {}) or {}

    print(
        f"[FeedbackLoop] started (warmup_rounds={warmup_rounds }, per_round_limit={per_round_limit })"
    )
    while not pipeline.stop_event.is_set():
        try:
            await asyncio.wait_for(pipeline.round_completed_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            remaining = pipeline._remaining_subtopics_count()
            if remaining == 0 and pipeline._all_stage_queues_empty():
                print(
                    "[FeedbackLoop] All subtopics completed & queues empty; signaling stop_event."
                )
                pipeline.stop_event.set()
                break
            continue
        pipeline.round_completed_event.clear()

        try:
            if pipeline.rejection_vector_tracker and getattr(
                pipeline.rejection_vector_tracker, "_last_payload", None
            ):
                pl = pipeline.rejection_vector_tracker._last_payload or {}
                round_id = pl.get("round_id")
                pass_rate = pl.get("pass_rate")
                acc_q = pl.get("accepted_queries") or []
                rej_q = pl.get("rejected_queries") or []
                used_queries = sorted(list(set(acc_q + rej_q)))
                print(
                    json.dumps(
                        {
                            "RoundSummary": {
                                "round": round_id,
                                "pass_rate": pass_rate,
                                "used_queries": used_queries[:50],
                            }
                        },
                        ensure_ascii=False,
                    )
                )
        except Exception as e:
            print(f"[FeedbackLoop] Print round summary error: {e }")

        if pipeline.critic and pipeline.critic.should_use_new_strategy():
            try:
                latest = pipeline.critic.latest_strategy or {}
                strat = latest.get("strategy") or {}
                strat_queries = strat.get("queries") or []
                added = 0
                if strat_queries:
                    async with pipeline._enqueue_lock:
                        targets = [
                            p
                            for p in pipeline._all_subtopics
                            if p not in pipeline._processed_subtopics
                            and p not in pipeline._scheduled_subtopics
                        ]
                        if not targets:
                            targets = list(pipeline._all_subtopics)
                        for topic, sub in targets:
                            for q in strat_queries:
                                if not isinstance(q, str) or not q.strip():
                                    continue
                                if q in pipeline._generated_queries:
                                    continue
                                try:
                                    pipeline.queries_queue.put_nowait(
                                        (1, -time.time(), (topic, sub, q))
                                    )
                                    pipeline._generated_queries.add(q)
                                    added += 1
                                except asyncio.QueueFull:
                                    break
                        if added > 0:
                            print(
                                f"[FeedbackLoop] Enqueued {added } queries from latest strategy."
                            )
                            pipeline._print_queue_head()
            except Exception as e:
                print(f"[FeedbackLoop] Strategy enqueue error: {e }")

        if not pipeline.rejection_vector_tracker or not getattr(
            pipeline.rejection_vector_tracker, "last_round_details", None
        ):
            continue
        details = pipeline.rejection_vector_tracker.last_round_details
        rid = details.get("round_index", 0)
        if rid <= warmup_rounds:
            print(
                f"[FeedbackLoop] Round {rid } <= warmup ({warmup_rounds }) warmup mode: using base query generation."
            )
        async with pipeline._enqueue_lock:
            remaining = [
                p
                for p in pipeline._all_subtopics
                if p not in pipeline._processed_subtopics
                and p not in pipeline._scheduled_subtopics
            ]
            if (
                remaining
                and pipeline.critic
                and hasattr(pipeline.critic, "set_next_subtopics")
            ):
                with contextlib.suppress(Exception):
                    preview_limit = max(per_round_limit, 5)
                    preview_subs = [s for _, s in remaining[:preview_limit]]
                    pipeline.critic.set_next_subtopics(preview_subs)
                    print(f"[FeedbackLoop] set_next_subtopics preview={preview_subs }")
            if not remaining:
                if pipeline.queries_queue.empty():
                    print("[FeedbackLoop] No remaining subtopics; signaling stop.")
                    pipeline.stop_event.set()
                break

            batch = remaining[:per_round_limit]
            added_subtopics = 0
            for topic, sub in batch:
                if pipeline.stop_event.is_set():
                    break
                queries, is_strategy = await asyncio.to_thread(
                    generate_queries, pipeline, topic, sub
                )
                if pipeline._limit_queries:
                    queries = queries[: pipeline._limit_queries]
                added_any = False
                priority = 1 if is_strategy else 10
                for q in queries:
                    if q in pipeline._generated_queries:
                        continue
                    while True:
                        try:
                            pipeline.queries_queue.put_nowait(
                                (priority, -time.time(), (topic, sub, q))
                            )
                            pipeline._generated_queries.add(q)
                            added_any = True
                            break
                        except asyncio.QueueFull:
                            await asyncio.sleep(0.05)
                pipeline._processed_subtopics.add((topic, sub))
                if added_any:
                    added_subtopics += 1
                    pipeline._print_queue_head()
        print(
            f"[FeedbackLoop] Round {rid } added {added_subtopics } subtopics; "
            f"processed={len (pipeline ._processed_subtopics )}/{len (pipeline ._all_subtopics )}"
        )
    print("[FeedbackLoop] exiting.")
