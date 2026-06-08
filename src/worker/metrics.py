"""Periodic logging of pipeline throughput and resource usage."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline

try:
    import psutil

    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


async def metrics_loop(pipeline: OrchestratorPipeline) -> None:
    """Periodically print cumulative and incremental performance metrics."""
    interval = float(pipeline.metrics_cfg.get("log_interval_sec", 15))
    while True:
        await asyncio.sleep(interval)
        try:
            print_metrics_snapshot(pipeline)
        except Exception as e:
            print(f"[MetricsLoopError] {e }")


def print_metrics_snapshot(pipeline: OrchestratorPipeline, final: bool = False) -> None:
    """Print current cumulative and incremental performance metrics."""
    now = time.time()
    elapsed = now - pipeline.start_ts
    tot = pipeline.loop_totals
    checked = tot["accepted"] + tot["rejected"]
    acc_rate = (tot["accepted"] / checked) if checked else 0.0
    delta_downloaded = tot["downloaded"] - pipeline._last_metrics_snap["downloaded"]
    delta_accepted = tot["accepted"] - pipeline._last_metrics_snap["accepted"]
    delta_rejected = tot["rejected"] - pipeline._last_metrics_snap["rejected"]
    delta_t = max(now - pipeline._last_metrics_snap["t"], 1e-6)
    speed_download = delta_downloaded / delta_t
    speed_accept = delta_accepted / delta_t
    queue_sizes = {}
    if pipeline.metrics_cfg.get("enable_queue_sizes", True):
        queue_sizes = {
            "Q_queries": (
                pipeline.queries_queue.qsize() if pipeline.queries_queue else 0
            ),
            "Q_downloaded": (
                pipeline.downloaded_queue.qsize() if pipeline.downloaded_queue else 0
            ),
            "Q_ocr": pipeline.ocr_queue.qsize() if pipeline.ocr_queue else 0,
            "Q_quality": (
                pipeline.quality_queue.qsize() if pipeline.quality_queue else 0
            ),
            "Q_semantic": (
                pipeline.semantic_queue.qsize() if pipeline.semantic_queue else 0
            ),
        }
    mem_info = {}
    if _HAS_PSUTIL and pipeline.metrics_cfg.get("enable_memory", False):
        p = psutil.Process(os.getpid())
        rss = p.memory_info().rss / (1024**2)
        mem_info = {"mem_MB": round(rss, 1)}
    line = {
        "t_sec": round(elapsed, 1),
        "checked": checked,
        "accepted": tot["accepted"],
        "rejected": tot["rejected"],
        "acc_rate": round(acc_rate, 4),
        "downloaded": tot["downloaded"],
        "speeds": {
            "dl_per_sec": round(speed_download, 2),
            "acc_per_sec": round(speed_accept, 2),
            "rej_per_sec": round(delta_rejected / delta_t, 2),
        },
        "reasons": {
            "quality": tot["quality_rejected"],
            "watermark": tot["watermark_rejected"],
            "semantic": tot["semantic_rejected"],
            "text_consistency": tot["text_consistency_rejected"],
            "pHash": tot["pHash_rejected"],
            "dedup_error": tot["dedup_error"],
            "ocr_submit_error": tot["ocr_submit_error"],
        },
        **queue_sizes,
        **mem_info,
    }
    tag = "FINAL" if final else "STAT"
    print(f"[Metrics-{tag }] {json .dumps (line ,ensure_ascii =False )}")
    pipeline._last_metrics_snap.update(
        {
            "accepted": tot["accepted"],
            "rejected": tot["rejected"],
            "downloaded": tot["downloaded"],
            "t": now,
        }
    )
