"""Performs pHash dedup, checks semantic relevance, text consistency, generates captions, and buffers accepted items."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.annotation.qwen_vl import generate_image_caption
from src.utils.common import sha1_of_file

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def semantic_worker(pipeline: OrchestratorPipeline, idx: int) -> None:
    """Perform pHash dedup, semantic and text-consistency checks, generate captions, and buffer accepted items."""
    while True:
        meta = await pipeline.semantic_queue.get()
        if meta is pipeline._sentinel:
            pipeline.semantic_queue.task_done()
            break
        fp: Path = meta["path"]

        # --- pHash deduplication (moved here from quality_worker) ---
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
                pipeline.semantic_queue.task_done()
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
            pipeline.semantic_queue.task_done()
            print(f"[DedupError-semantic] {fp .name } {e }")
            continue
        # --- end pHash deduplication ---

        sem_res = {}
        try:
            sem_ok, sem_res = pipeline.semantic.check_relevance(
                str(fp), topic=meta["topic"], subtopic=meta["subtopic"]
            )
            meta["semantic"] = sem_res
            if not sem_ok:
                reasons = sem_res.get("rejection_reasons", ["semantic_error"])
                pipeline._record_sample_outcome(meta, reasons)
                pipeline.loop_totals["rejected"] += 1
                pipeline.loop_totals["semantic_rejected"] += 1
                if not pipeline.keep_rejects:
                    fp.unlink(missing_ok=True)
                try:
                    pipeline.dedup.discard_path(str(fp), key)
                except Exception:
                    pass
                pipeline.semantic_queue.task_done()
                continue
        except Exception as e:
            print(f"[SemanticError] {fp .name } {e }")
            pipeline._record_sample_outcome(meta, ["semantic_error"])
            pipeline.loop_totals["rejected"] += 1
            try:
                pipeline.dedup.discard_path(str(fp), key)
            except Exception:
                pass
            pipeline.semantic_queue.task_done()
            continue

        if not pipeline.text_checker.is_available():
            meta["text_consistency"] = None
        else:
            try:
                tc = pipeline.text_checker.check(
                    meta.get("ocr", {}), meta["topic"], meta["subtopic"]
                )
                meta["text_consistency"] = tc
                if not tc.get("passed", True):
                    reasons = tc.get("rejection_reasons") or ["text_error"]
                    pipeline._record_sample_outcome(meta, reasons)
                    pipeline.loop_totals["rejected"] += 1
                    pipeline.loop_totals["text_consistency_rejected"] += 1
                    if not pipeline.keep_rejects:
                        fp.unlink(missing_ok=True)
                    try:
                        pipeline.dedup.discard_path(str(fp), key)
                    except Exception:
                        pass
                    pipeline.semantic_queue.task_done()
                    continue
            except Exception as e:
                print(f"[TextConsistencyError] {fp .name } {e }")
                pipeline._record_sample_outcome(meta, ["text_error"])
                pipeline.loop_totals["rejected"] += 1
                try:
                    pipeline.dedup.discard_path(str(fp), key)
                except Exception:
                    pass
                pipeline.semantic_queue.task_done()
                continue

        caption = ""
        try:
            loop = asyncio.get_event_loop()
            caption = await loop.run_in_executor(
                None,
                generate_image_caption,
                str(fp),
                pipeline.cfg,
                meta["topic"],
                meta["subtopic"],
                meta.get("ocr"),
                meta.get("quality"),
            )
            if not caption:
                caption = f'{meta ["topic"]} - {meta ["subtopic"]}'
        except Exception as e:
            print(f"[CaptionError] {fp .name } {e }")
            caption = f'{meta ["topic"]} - {meta ["subtopic"]}'

        item = {
            "image_path": str(fp),
            "topic": meta["topic"],
            "subtopic": meta["subtopic"],
            "query": meta["query"],
            "caption": caption,
            "source": "bing-noapi",
            "is_generated": False,
            "ocr": meta.get("ocr"),
            "quality": meta.get("quality"),
            "semantic": sem_res,
            "text_consistency": meta.get("text_consistency"),
            "content_sha1": sha1_of_file(fp),
            "ts": time.time(),
            "shard": pipeline.shard_id,
        }
        pipeline._record_sample_outcome(meta, [])
        pipeline.loop_totals["accepted"] += 1
        pipeline._batch_buffer.append(item)
        await pipeline._flush_batch(force=True)
        pipeline.semantic_queue.task_done()
