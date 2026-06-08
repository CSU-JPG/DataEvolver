"""Submits images for OCR processing and routes results to the quality queue."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline

try:
    import torch
except ImportError:
    torch = None


async def ocr_worker(pipeline: OrchestratorPipeline, idx: int) -> None:
    """Submit images for OCR processing and forward results to the quality queue."""
    loop = asyncio.get_running_loop()
    submitted = 0
    submit_errors = 0

    while True:
        meta = await pipeline.ocr_queue.get()
        if meta is pipeline._sentinel:
            pipeline.ocr_queue.task_done()
            print(
                f"[OCR-Worker-{idx }] Exit. submitted={submitted } submit_errors={submit_errors }"
            )
            break

        fp: Path = meta["path"]

        if pipeline.ocr_multiprocess_pool:
            throttle_wait = 0
            while len(pipeline._ocr_pending) >= pipeline._max_ocr_pending:
                if pipeline._ocr_workers_done:
                    print(
                        f"[OCR-Worker-{idx }] Workers marked done, "
                        f"breaking throttle (pending={len (pipeline ._ocr_pending )})"
                    )
                    break
                throttle_wait += 0.2
                if idx == 0 and throttle_wait > 5:
                    print(
                        f"[OCRThrottle] pending={len (pipeline ._ocr_pending )} "
                        f">= {pipeline ._max_ocr_pending }, waiting..."
                    )
                    throttle_wait = 0
                await asyncio.sleep(0.2)
            job_id = uuid.uuid4().hex
            meta["job_id"] = job_id

            if not meta.get("path"):
                print(f"[OCR-Worker-{idx }] ERROR: meta missing 'path'")
                pipeline._record_sample_outcome(meta, ["ocr_meta_error"])
                pipeline.loop_totals["rejected"] += 1
                pipeline.loop_totals["ocr_submit_error"] += 1
                submit_errors += 1
                pipeline.ocr_queue.task_done()
                continue

            try:
                pipeline._ocr_pending[job_id] = meta
                pipeline.ocr_multiprocess_pool.submit_task_round_robin(
                    meta, pipeline._rr_seq
                )
                pipeline._rr_seq += 1
                submitted += 1
                if submitted % 200 == 0:
                    print(
                        f"[OCR-Submit-{idx }] submitted={submitted } "
                        f"pending={len (pipeline ._ocr_pending )} "
                        f"Q_ocr={pipeline .ocr_queue .qsize ()}"
                    )
            except Exception as e:
                submit_errors += 1
                pipeline._ocr_pending.pop(job_id, None)
                print(
                    f"[OCR-SubmitError-{idx }] {fp .name }: {type (e ).__name__ } {e }"
                )
                pipeline._record_sample_outcome(meta, ["ocr_failed"])
                pipeline.loop_totals["rejected"] += 1
                pipeline.loop_totals["ocr_submit_error"] += 1
            finally:
                pipeline.ocr_queue.task_done()
            continue

        ocr_instance = None
        rec = None
        try:
            async with pipeline._ocr_sem:
                ocr_instance = await loop.run_in_executor(
                    None, pipeline.ocr_queue_pool.get, True, 10.0
                )
                gpu_id = getattr(ocr_instance, "device_id", None)
                if gpu_id is not None and gpu_id >= 0:
                    try:
                        if torch:
                            torch.cuda.set_device(gpu_id)
                            free_bytes, _ = torch.cuda.mem_get_info()
                            free_mb = free_bytes / (1024**2)
                            if free_mb < 500:
                                print(
                                    f"[OCRMemWarn] GPU:{gpu_id } free={free_mb :.0f}MB"
                                )
                                torch.cuda.empty_cache()
                    except Exception as mem_e:
                        print(f"[OCRMemCheckError] {mem_e }")
                for attempt in range(3):
                    try:
                        rec = await asyncio.wait_for(
                            loop.run_in_executor(None, ocr_instance.run, str(fp)),
                            timeout=30.0,
                        )
                        break
                    except asyncio.TimeoutError:
                        print(f"[OCRTimeout] {fp .name } attempt={attempt +1 }")
                        if attempt == 2:
                            raise
                        await asyncio.sleep(1.0)
                    except Exception as oe:
                        print(f"[OCRError] {fp .name } attempt={attempt +1 } {oe }")
                        if attempt == 2:
                            raise
                        await asyncio.sleep(1.0)
        except Exception as e:
            print(f"[OCRFatal] {fp .name } {e }")
            pipeline._record_sample_outcome(meta, ["ocr_error"])
            pipeline.loop_totals["rejected"] += 1
            pipeline.ocr_queue.task_done()
            if ocr_instance:
                with contextlib.suppress(Exception):
                    pipeline.ocr_queue_pool.put(ocr_instance, block=False)
            continue
        finally:
            if ocr_instance:
                with contextlib.suppress(Exception):
                    pipeline.ocr_queue_pool.put(ocr_instance, block=False)

        meta["ocr"] = rec
        await pipeline.quality_queue.put(meta)
        pipeline.ocr_queue.task_done()
