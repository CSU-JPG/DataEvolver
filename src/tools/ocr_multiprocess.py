"""Multi-GPU OCR process pool with round-robin task dispatch."""

from __future__ import annotations

import os
import time
import uuid
from multiprocessing import get_context
from queue import Empty
from typing import Any, Dict, List, Optional

import torch

__all__ = ["MultiProcessOCRPool", "ocr_worker_process"]


def ocr_worker_process(
    gpu_id: int,
    task_q: Any,
    result_q: Any,
    cfg: Dict[str, Any],
    worker_id: int,
) -> None:
    """Long-running OCR worker bound to a single GPU."""
    import paddle

    try:
        torch.cuda.set_device(gpu_id)
    except Exception as e:
        print(f"[OCRWorker-{worker_id }] set_device({gpu_id }) failed: {e }")

    print(
        f"[OCRWorker-{worker_id }] "
        f"cuda_available={torch .cuda .is_available ()} "
        f"paddle_cuda={paddle .device .is_compiled_with_cuda ()} "
        f"visible={os .environ .get ('CUDA_VISIBLE_DEVICES')}"
    )

    if torch.cuda.is_available():
        x = torch.randn(1).cuda()
        print(
            f"[OCRWorker-{worker_id }] gpu_id={gpu_id } "
            f"test_sum={float (x .sum ())}"
        )

    try:
        from src.tools.ocr import OCR

        ocr = OCR(cfg, device_id=gpu_id)
        print(f"[OCRWorker-{worker_id }] OCR instance ready")
    except Exception as e:
        print(f"[OCRWorker-{worker_id }] FATAL: OCR init failed: {e }")
        return

    processed = 0
    errors = 0

    while True:
        try:
            task = task_q.get(timeout=1.0)
        except Empty:
            continue

        if task is None:
            print(f"[OCRWorker-{worker_id }] Received stop signal")
            break

        job_id = task.get("job_id")
        if not job_id:
            errors += 1
            print(f"[OCRWorker-{worker_id }] ERROR: task missing job_id")
            result_q.put({"error": "missing_job_id"})
            continue

        target_gpu = task.get("target_gpu")
        if target_gpu is not None and target_gpu != gpu_id:
            result_q.put({"job_id": job_id, "error": "wrong_gpu"})
            continue

        img_path = task.get("path")
        if not img_path:
            result_q.put({"job_id": job_id, "error": "missing_path"})
            continue

        st = time.time()
        try:
            rec = ocr.run(str(img_path))
            task["ocr"] = rec
            task["ocr_time"] = time.time() - st
            task["success"] = True
        except Exception as e:
            errors += 1
            print(f"[OCRWorker-{worker_id }] Error processing {img_path }: {e }")
            task["ocr"] = {
                "error": str(e),
                "boxes": [],
                "texts": [],
                "confs": [],
                "avg_conf": 0.0,
            }
            task["success"] = False

        task["ocr_worker_id"] = worker_id
        task["ocr_gpu_id"] = gpu_id
        task["job_id"] = job_id

        result_q.put(task)
        processed += 1

        if processed % 20 == 0:
            print(
                f"[OCRWorker-{worker_id }] GPU:{gpu_id } "
                f"processed={processed } errors={errors }"
            )

    print(f"[OCRWorker-{worker_id }] Exit. " f"processed={processed } errors={errors }")


class MultiProcessOCRPool:
    """Manages a pool of single-GPU OCR worker processes."""
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.ctx = get_context("spawn")
        self.task_qs: Dict[int, Any] = {}
        self.result_q = self.ctx.Queue(maxsize=8000)
        self.procs: List[Any] = []
        self._started = False
        self.gpu_ids: List[int] = []

    def start(
        self,
        num_gpus: Optional[int] = None,
        instances_per_gpu: int = 1,
    ) -> None:
        """Launch worker processes, one per GPU."""
        if self._started:
            return

        if not torch.cuda.is_available():
            print("[MultiProcessOCRPool] CUDA unavailable")
            return

        avail = torch.cuda.device_count()
        if num_gpus is None:
            num_gpus = avail
        num_gpus = min(num_gpus, avail)

        if instances_per_gpu != 1:
            print("[MultiProcessOCRPool] Force instances_per_gpu=1")
            instances_per_gpu = 1

        print(f"[MultiProcessOCRPool] spawn {num_gpus } GPU(s)")

        min_free_gb = float(self.cfg.get("ocr", {}).get("min_free_gb", 2.0))

        for g in range(num_gpus):
            try:
                torch.cuda.set_device(g)
                fb, _ = torch.cuda.mem_get_info()
                if fb / (1024**3) < min_free_gb:
                    print(f"[MultiProcessOCRPool] Skip GPU:{g } low mem")
                    continue
            except Exception as e:
                print(f"[MultiProcessOCRPool] mem check GPU:{g } failed {e }")
                continue

            q = self.ctx.Queue(maxsize=8000)
            self.task_qs[g] = q
            p = self.ctx.Process(
                target=ocr_worker_process,
                args=(g, q, self.result_q, self.cfg, g),
            )
            p.start()
            self.procs.append(p)
            self.gpu_ids.append(g)
            print(f"[MultiProcessOCRPool] worker PID:{p .pid } GPU:{g }")

        self._started = True
        print(f"[MultiProcessOCRPool] Started {len (self .procs )} workers")

    def submit_task_round_robin(self, meta: Dict[str, Any], seq: int) -> None:
        """Submit one task, distributing across GPUs round-robin."""
        if "job_id" not in meta:
            meta["job_id"] = uuid.uuid4().hex
        if not self.gpu_ids:
            raise RuntimeError("No GPU workers available")
        gpu_id = self.gpu_ids[seq % len(self.gpu_ids)]
        meta["target_gpu"] = gpu_id
        self.task_qs[gpu_id].put(meta)

    def get_result(self, timeout: Optional[float] = None) -> Any:
        """Block until the next result is available."""
        if timeout:
            return self.result_q.get(timeout=timeout)
        return self.result_q.get()

    def stop(self) -> None:
        """Send stop signals and join all workers."""
        if not self._started:
            return

        for g, q in self.task_qs.items():
            q.put(None)

        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        self.procs.clear()
        self.task_qs.clear()
        self.gpu_ids.clear()
        self._started = False
        print("[MultiProcessOCRPool] stopped")
