"""Watches the query queue water level and triggers fallback when it drops too low."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from src.worker.query_gen import generate_queries

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def queue_monitor(pipeline: OrchestratorPipeline) -> None:
    """Continuously monitor the queries queue and refill when below the low-water mark."""
    dq_cfg = pipeline.cfg.get("dynamic_query", {}) or {}
    low_watermark = int(dq_cfg.get("low_watermark", 30))
    batch_size = int(dq_cfg.get("monitor_batch_size", 3))
    per_subtopic_limit = int(dq_cfg.get("monitor_queries_per_subtopic", 15) or 15)

    if low_watermark <= 0:
        return

    # print(
    #     f"[QueueMonitor] started low_watermark={low_watermark }, batch_size={batch_size }"
    # )

    while not pipeline.stop_event.is_set():
        try:
            current_size = pipeline.queries_queue.qsize()
            if current_size < low_watermark:
                async with pipeline._enqueue_lock:
                    remaining = [
                        p
                        for p in pipeline._all_subtopics
                        if p not in pipeline._processed_subtopics
                        and p not in pipeline._scheduled_subtopics
                    ]
                    if not remaining:
                        if (
                            pipeline.queries_queue.empty()
                            and pipeline.rejection_vector_tracker
                            and pipeline.rejection_vector_tracker.processed > 0
                        ):
                            # print(
                        #         f"[QueueMonitor] No remaining subtopics; forcing finalize "
                        #         f"round incomplete processed={pipeline .rejection_vector_tracker .processed }"
                        #     )
                            with contextlib.suppress(Exception):
                                pipeline.rejection_vector_tracker.finalize()
                    else:
                        batch = remaining[:batch_size]
                    # print(
                    #     f"[QueueMonitor] Low watermark ({current_size } < {low_watermark }). "
                    #     f"Fetching batch of {len (batch )} subtopics..."
                    # )
                        if pipeline.critic and hasattr(
                            pipeline.critic, "set_next_subtopics"
                        ):
                            with contextlib.suppress(Exception):
                                preview_len = max(batch_size, 5)
                                preview_subs = [s for _, s in remaining[:preview_len]]
                                pipeline.critic.set_next_subtopics(preview_subs)
                        for topic, sub in batch:
                            key = (topic, sub)
                            if (
                                key in pipeline._processed_subtopics
                                or key in pipeline._scheduled_subtopics
                            ):
                                continue
                            try:
                                queries, is_strategy = await asyncio.to_thread(
                                    generate_queries, pipeline, topic, sub
                                )
                                if pipeline._limit_queries:
                                    queries = queries[: pipeline._limit_queries]
                                added_count = 0
                                priority = 1 if is_strategy else 10
                                for q in queries:
                                    if q in pipeline._generated_queries:
                                        continue
                                    try:
                                        pipeline.queries_queue.put_nowait(
                                            (priority, -time.time(), (topic, sub, q))
                                        )
                                        pipeline._generated_queries.add(q)
                                        added_count += 1
                                    except asyncio.QueueFull:
                                        break
                                if added_count > 0:
                                    # print(
                                    #     f"[QueueMonitor] Added subtopic {topic }/{sub } "
                                    #     f"queries={added_count }"
                                    # )
                                    pipeline._print_queue_head()
                                else:
                                    # print(
                                    #     f"[QueueMonitor] Subtopic {topic }/{sub } yielded "
                                    #     f"0 NEW queries (dups/empty). Marked processed."
                                    # )
                                    pass
                            except Exception as e:
                                # print(
                                #     f"[QueueMonitor] Error processing subtopic {topic }/{sub }: {e }"
                                # )
                                pass
                            finally:
                                pipeline._processed_subtopics.add(key)
                                pipeline._scheduled_subtopics.add(key)

            now_size = pipeline.queries_queue.qsize()
            if now_size <= current_size:
                pipeline._dq_empty_cycles += 1
            else:
                pipeline._dq_empty_cycles = 0
            if now_size > current_size:
                pipeline._dq_last_activity_ts = time.time()

            elapsed = time.time() - pipeline._dq_last_activity_ts
            max_wait_hit = (
                pipeline._dq_max_wait_sec > 0 and elapsed >= pipeline._dq_max_wait_sec
            )
            max_cycles_hit = (
                pipeline._dq_max_empty_cycles > 0
                and pipeline._dq_empty_cycles >= pipeline._dq_max_empty_cycles
            )
            if not pipeline._dq_fallback_triggered and (max_wait_hit or max_cycles_hit):
                # print(
                #     f"[QueueMonitor] Fallback triggered "
                #     f"(elapsed={round (elapsed ,1 )}s, empty_cycles={pipeline ._dq_empty_cycles })."
                # )
                pipeline._dq_fallback_triggered = True
                await apply_dq_fallback(pipeline)
                pipeline._dq_empty_cycles = 0
                pipeline._dq_last_activity_ts = time.time()

            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # print("[QueueMonitor] cancelled")
            break
        except Exception as e:
            # print(f"[QueueMonitor] error {e }")
            await asyncio.sleep(2.0)


async def apply_dq_fallback(pipeline: OrchestratorPipeline) -> None:
    """Execute the configured fallback strategy when the query queue is starved."""
    strategy = (pipeline._dq_fallback_strategy or "static").lower()
    if strategy == "static":
        # print("[QueueMonitor] Applying static fallback: enqueue seed queries.")
        await fallback_enqueue_static(pipeline)
    elif strategy == "warmup":
        # print(
        #     "[QueueMonitor] Applying warmup fallback: generate queries via warmup logic."
        # )
        await fallback_enqueue_warmup(pipeline)
    else:
        # print(f"[QueueMonitor] Unknown fallback strategy '{strategy }', skipping.")
        pass


async def fallback_enqueue_static(pipeline: OrchestratorPipeline) -> None:
    """Enqueue seed queries from config for remaining unprocessed subtopics."""
    dq_cfg = pipeline.cfg.get("dynamic_query", {}) or {}
    per_subtopic_limit = int(dq_cfg.get("monitor_queries_per_subtopic", 15) or 15)
    batch_limit = int(dq_cfg.get("monitor_batch_size", 3) or 3)
    remaining = [
        pair
        for pair in pipeline._all_subtopics
        if pair not in pipeline._processed_subtopics
    ]
    if not remaining:
        # print("[QueueMonitor] No remaining subtopics for static fallback.")
        return
    selected = remaining[:batch_limit]
    topics_cfg = pipeline.cfg.get("topics") or []
    topic_seed_map = {
        tc.get("name"): (tc.get("seed_queries") or []) for tc in topics_cfg
    }
    added = 0
    for topic, sub in selected:
        seeds = topic_seed_map.get(topic, [])
        if not seeds:
            continue
        for q in seeds[:per_subtopic_limit]:
            key = f"{topic }::{sub }::{q }".strip()
            if key in pipeline._generated_queries:
                continue
            priority = 0
            item = (priority, -time.time(), (topic, sub, q))
            try:
                await pipeline.queries_queue.put(item)
                pipeline._generated_queries.add(key)
                added += 1
            except Exception:
                break
    # print(
    #     f"[QueueMonitor] Static fallback enqueued {added } queries across {len (selected )} subtopics."
    # )


async def fallback_enqueue_warmup(pipeline: OrchestratorPipeline) -> None:
    """Use warmup-phase query generation logic to enqueue queries for remaining subtopics."""
    dq_cfg = pipeline.cfg.get("dynamic_query", {}) or {}
    per_subtopic_limit = int(dq_cfg.get("monitor_queries_per_subtopic", 15) or 15)
    batch_limit = int(dq_cfg.get("monitor_batch_size", 3) or 3)
    remaining = [
        pair
        for pair in pipeline._all_subtopics
        if pair not in pipeline._processed_subtopics
        and pair not in pipeline._scheduled_subtopics
    ]
    if not remaining:
        # print("[QueueMonitor] No remaining subtopics for warmup fallback.")
        return
    selected = remaining[:batch_limit]
    added_total = 0
    for topic, sub in selected:
        try:
            queries, is_strategy = await asyncio.to_thread(
                generate_queries, pipeline, topic, sub
            )
            priority = 10
            added = 0
            for q in queries[:per_subtopic_limit]:
                if q in pipeline._generated_queries:
                    continue
                try:
                    await pipeline.queries_queue.put(
                        (priority, -time.time(), (topic, sub, q))
                    )
                    pipeline._generated_queries.add(q)
                    added += 1
                except asyncio.QueueFull:
                    break
            if added > 0:
                pipeline._processed_subtopics.add((topic, sub))
                pipeline._scheduled_subtopics.add((topic, sub))
                added_total += added
                pipeline._print_queue_head()
        except Exception as e:
            # print(f"[QueueMonitor] Warmup fallback error {topic }/{sub }: {e }")
            pass
    # print(
    #     f"[QueueMonitor] Warmup fallback enqueued {added_total } queries across {len (selected )} subtopics."
    # )
