"""Aggregates multi-process OCR results and routes them to the quality queue."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def ocr_result_collector(pipeline: OrchestratorPipeline) -> None:
    """Aggregate OCR results from the multi-process pool and route them to the quality queue."""
    loop = asyncio.get_running_loop()
    collected = 0
    orphans = 0
    errors = 0

    def _background_puller() -> None:
        pulled_count = 0
        thread_errors = 0
        while not pipeline._collector_stop_flag.is_set():
            try:
                r = pipeline.ocr_multiprocess_pool.result_q.get(timeout=3.0)
                loop.call_soon_threadsafe(pipeline._result_buffer_queue.put_nowait, r)
                pulled_count += 1
                if pulled_count % 1000 == 0:
                    print(
                        f"[OCRPullerThread] Pulled {pulled_count } results, "
                        f"buffer={pipeline ._result_buffer_queue .qsize ()}"
                    )
            except Exception as e:
                if "Empty" not in str(type(e).__name__):
                    thread_errors += 1
                    if thread_errors % 100 == 0:
                        print(f"[OCRPullerThreadError] {e } (total={thread_errors })")
                continue
        print(
            f"[OCRPullerThread] Stopped. pulled={pulled_count } errors={thread_errors }"
        )

    puller_thread = threading.Thread(
        target=_background_puller,
        daemon=True,
        name="OCRResultPuller",
    )
    puller_thread.start()
    pipeline._collector_puller_thread = puller_thread

    print("[OCRResultCollector] Started with background puller thread")

    batch_size = 50
    idle_timeout = 0.1

    while True:
        try:
            batch = []
            for _ in range(batch_size):
                try:
                    r = pipeline._result_buffer_queue.get_nowait()
                    batch.append(r)
                except asyncio.QueueEmpty:
                    break

            if not batch:
                if pipeline._collector_stop_event.is_set():
                    print("[OCRResultCollector] Stop event received")
                    break

                if pipeline._ocr_workers_done and not pipeline._ocr_pending:
                    print("[OCRResultCollector] All workers done and no pending tasks")
                    break

                if pipeline._ocr_workers_done and pipeline._ocr_pending:
                    max_wait = 30
                    wait_start = time.time()
                    print(
                        f"[OCRResultCollector] Waiting for {len (pipeline ._ocr_pending )} "
                        f"pending tasks to arrive..."
                    )
                    while (
                        pipeline._ocr_pending and (time.time() - wait_start) < max_wait
                    ):
                        temp_batch = []
                        for _ in range(100):
                            try:
                                r = pipeline._result_buffer_queue.get_nowait()
                                temp_batch.append(r)
                            except asyncio.QueueEmpty:
                                break
                        if temp_batch:
                            for r in temp_batch:
                                jid = r.get("job_id")
                                if not jid:
                                    errors += 1
                                    continue
                                meta = pipeline._ocr_pending.pop(jid, None)
                                if meta is None:
                                    orphans += 1
                                    continue
                                meta.update(
                                    {
                                        "ocr": r.get("ocr", {}),
                                        "ocr_time": r.get("ocr_time"),
                                        "ocr_gpu_id": r.get("ocr_gpu_id"),
                                        "success": r.get("success", False),
                                    }
                                )
                                await pipeline.quality_queue.put(meta)
                                collected += 1
                            print(
                                f"[OCRResultCollector] Drained {len (temp_batch )}, "
                                f"remaining={len (pipeline ._ocr_pending )}"
                            )
                            continue
                        if pipeline._ocr_pending:
                            remaining_time = max_wait - (time.time() - wait_start)
                            if remaining_time > 0:
                                print(
                                    f"[OCRResultCollector] Still waiting for "
                                    f"{len (pipeline ._ocr_pending )} tasks "
                                    f"(timeout in {remaining_time :.1f}s)"
                                )
                                await asyncio.sleep(0.5)
                            else:
                                break
                    if pipeline._ocr_pending:
                        print(
                            f"[OCRResultCollector] WARNING: {len (pipeline ._ocr_pending )} "
                            f"tasks still unmatched after {max_wait }s"
                        )
                        print("[OCRResultCollector] Final drain from result_q...")
                        final_drained = 0
                        for _ in range(500):
                            try:
                                r = pipeline.ocr_multiprocess_pool.result_q.get_nowait()
                                jid = r.get("job_id")
                                if jid in pipeline._ocr_pending:
                                    meta = pipeline._ocr_pending.pop(jid)
                                    meta.update(
                                        {
                                            "ocr": r.get("ocr", {}),
                                            "ocr_time": r.get("ocr_time"),
                                            "ocr_gpu_id": r.get("ocr_gpu_id"),
                                            "success": r.get("success", False),
                                        }
                                    )
                                    await pipeline.quality_queue.put(meta)
                                    final_drained += 1
                            except Exception:
                                break
                        print(
                            f"[OCRResultCollector] Final drained {final_drained }, "
                            f"still_orphaned={len (pipeline ._ocr_pending )}"
                        )
                    break

                await asyncio.sleep(idle_timeout)
                continue

            valid_count = 0
            for r in batch:
                jid = r.get("job_id")
                if not jid:
                    errors += 1
                    continue
                meta = pipeline._ocr_pending.pop(jid, None)
                if meta is None:
                    orphans += 1
                    if orphans % 100 == 0:
                        print(
                            f"[OCRResultCollector] orphan job_id={jid [:8 ]}, "
                            f"total_orphans={orphans }"
                        )
                    continue
                meta.update(
                    {
                        "ocr": r.get("ocr", {}),
                        "ocr_time": r.get("ocr_time"),
                        "ocr_gpu_id": r.get("ocr_gpu_id"),
                        "success": r.get("success", False),
                    }
                )
                valid_count += 1
                await pipeline.quality_queue.put(meta)

            collected += valid_count
            if collected % 100 == 0:
                print(
                    f"[OCRResultCollector] Collected {collected }, "
                    f"batch_size={len (batch )}, valid={valid_count }, "
                    f"pending={len (pipeline ._ocr_pending )}, "
                    f"buffer={pipeline ._result_buffer_queue .qsize ()}"
                )

        except Exception as e:
            print(f"[OCRResultCollectorError] {e }")
            import traceback

            traceback.print_exc()
            await asyncio.sleep(0.1)

    print("[OCRResultCollector] Stopping background puller thread...")
    pipeline._collector_stop_flag.set()
    if puller_thread.is_alive():
        puller_thread.join(timeout=5.0)
        if puller_thread.is_alive():
            print("[OCRResultCollector] WARNING: Puller thread did not exit cleanly")
        else:
            print("[OCRResultCollector] Puller thread exited")

    print(
        f"[OCRResultCollector] Stopped. "
        f"collected={collected } orphans={orphans } errors={errors }"
    )
