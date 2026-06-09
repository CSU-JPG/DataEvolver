"""Performs pHash dedup, scores image quality and routes to the semantic queue."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def quality_worker(pipeline: OrchestratorPipeline, idx: int) -> None:
    """Perform pHash dedup, assess image quality, watermark-check, then forward to semantic queue."""
    loop = asyncio.get_running_loop()
    while True:
        meta = await pipeline.quality_queue.get()
        if meta is pipeline._sentinel:
            pipeline.quality_queue.task_done()
            break
        fp: Path = meta["path"]

        key = meta["topic"] if pipeline._dedup_scope == "topic" else ""
        try:
            async with pipeline.dedup_lock:
                is_dup, _ = pipeline.dedup.check_and_add(str(fp), key)
            if is_dup:
                pipeline._record_sample_outcome(meta, ["dedup_duplicate"])
                pipeline.loop_totals["rejected"] += 1
                pipeline.loop_totals["pHash_rejected"] += 1
                if not pipeline.keep_rejects:
                    fp.unlink(missing_ok=True)
                pipeline.quality_queue.task_done()
                continue
        except Exception as e:
            pipeline.loop_totals["dedup_error"] += 1
            pipeline.loop_totals["rejected"] += 1
            pipeline._record_sample_outcome(meta, ["dedup_error"])
            if not pipeline.keep_rejects:
                fp.unlink(missing_ok=True)
            try:
                pipeline.dedup.discard_path(str(fp), key)
            except Exception:
                pass
            pipeline.quality_queue.task_done()
            print(f"[DedupError-quality] {fp .name } {e }")
            continue

        try:
            qscore = await loop.run_in_executor(
                None, pipeline.qa.score, str(fp), meta.get("ocr", {})
            )
            meta["quality"] = qscore
            if hasattr(pipeline.qa, "check"):
                ok, reasons = pipeline.qa.check(qscore, meta.get("ocr", {}))
            else:
                ok = pipeline.qa.accept(qscore, meta.get("ocr", {}))
                reasons = [] if ok else ["legacy_reject"]
            if not ok:
                pipeline._record_sample_outcome(meta, reasons)
                pipeline.loop_totals["rejected"] += 1
                pipeline.loop_totals["quality_rejected"] += 1
                if "has_watermark" in reasons:
                    pipeline.loop_totals["watermark_rejected"] += 1
                    wm_keys = meta.get("quality", {}).get("watermark_keywords", [])
                    wm_methods = meta.get("quality", {}).get("watermark_methods", [])
                    # print(
                    #     f"[WatermarkReject] {fp .name } "
                    #     f"methods={wm_methods } keywords={wm_keys [:3 ]}"
                    # )
                if not pipeline.keep_rejects:
                    fp.unlink(missing_ok=True)
                pipeline.quality_queue.task_done()
                continue
        except Exception as e:
            print(f"[QualityError] {fp .name } {e }")
            pipeline._record_sample_outcome(meta, ["quality_error"])
            pipeline.loop_totals["rejected"] += 1
            pipeline.quality_queue.task_done()
            continue

        await pipeline.semantic_queue.put(meta)
        pipeline.quality_queue.task_done()
