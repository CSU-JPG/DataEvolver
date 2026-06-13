from __future__ import annotations
import asyncio
import contextlib
import json
import multiprocessing as mp
import pathlib
import queue
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import torch
except ImportError:
    torch = None

if mp:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

from src.tools.ocr import OCR
from src.tools.quality import QualityAssessor
from src.tools.semantic_checker import SemanticChecker
from src.tools.hashing import HashDeduplicator
from src.tools.text_consistency import TextConsistencyChecker
from src.critic import Critic
from src.dataset.writer import DatasetWriter
from src.utils.common import ensure_dir, sanitize_subtopic_name
from src.utils.trackers import (
    RejectTracker,
    RejectionVectorRoundTracker,
    AsyncTokenBucket,
)

from src.worker.dispatcher import download_dispatcher
from src.worker.metrics import metrics_loop, print_metrics_snapshot
from src.worker.query_gen import enqueue_all_queries, generate_queries, call_agent_list
from src.worker.query import query_worker
from src.worker.ocr import ocr_worker
from src.worker.ocr_collector import ocr_result_collector
from src.worker.quality import quality_worker
from src.worker.semantic import semantic_worker
from src.worker.generate import plan_prompts_and_generate, post_process_generated
from src.worker.feedback import feedback_loop
from src.worker.monitor import (
    queue_monitor,
    apply_dq_fallback,
    fallback_enqueue_static,
    fallback_enqueue_warmup,
)
from src.worker.writer import (
    flush_batch,
    save_accepted,
    save_generated,
    sanitize_caption,
    alloc_filename,
    content_hash_filename,
    unique_final,
    write_image_index_entry,
)


class OrchestratorPipeline:
    """Asynchronous multi-stage image acquisition and generation pipeline."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        agents: Dict[str, Any],
        shard_id: str,
        keep_rejects: bool = True,
        config_path: str = "config.yaml",
    ):
        """Initialise pipeline directories, OCR engine, tools, and async queue placeholders."""
        self.cfg = cfg
        self.agents = agents
        self.shard_id = shard_id
        self.keep_rejects = keep_rejects
        self.config_path = config_path
        self.stop_event = None
        self.round_completed_event = None
        paths = cfg.get("paths", {})
        self.images_root = pathlib.Path(
            paths.get("images_crawled", "images_crawled")
        ).resolve()
        self.accepted_root = pathlib.Path(
            paths.get("accepted_dir", "accepted")
        ).resolve()
        self.gen_dir = pathlib.Path(
            paths.get("gen_dir", paths.get("images_generated", "generated_raw"))
        ).resolve()
        self.gen_accepted_root = pathlib.Path(
            paths.get("gen_accepted_dir", str(self.accepted_root / "generated"))
        ).resolve()
        for p in [
            self.images_root,
            self.accepted_root,
            self.gen_dir,
            self.gen_accepted_root,
        ]:
            ensure_dir(p)
        self.accepted_images_dir = ensure_dir(self.accepted_root / "images")
        self.accepted_captions_dir = ensure_dir(self.accepted_root / "captions")
        self.accepted_json_dir = ensure_dir(self.accepted_root / "img_json")
        fn_cfg = cfg.setdefault("filenames", {})
        self._filename_strategy = fn_cfg.get("strategy", "seq")
        self._filename_counter = 0
        self._filename_lock_sync = threading.Lock()
        self._init_filename_counter()
        self._gen_filename_counter = 0
        self._gen_filename_lock_sync = threading.Lock()
        self._init_gen_filename_counter()
        self._image_index_path = self.accepted_root / "image_index.jsonl"
        self._image_index_error_path = self.accepted_root / "image_index_errors.log"
        for path_ in [self._image_index_path, self._image_index_error_path]:
            if not path_.exists():
                with contextlib.suppress(Exception):
                    path_.write_text("", encoding="utf-8")

        from src.tools.ocr_multiprocess import MultiProcessOCRPool

        ocr_cfg = cfg.get("ocr", {})
        use_multi_gpu = ocr_cfg.get("use_multi_gpu", False)
        use_multiprocess = ocr_cfg.get("use_multiprocess", True)
        max_ocr_instances = ocr_cfg.get("max_instances", 2)
        instances_per_gpu = ocr_cfg.get("instances_per_gpu", 1)

        self.ocr_multiprocess_pool = None
        self._rr_seq = 0
        self.ocr_instances = []
        self.ocr_queue_pool = None
        self._ocr_init_lock = threading.Lock()
        self._ocr_pending: Dict[str, Dict[str, Any]] = {}
        self._ocr_result_collector_task = None
        self._ocr_workers_done = False
        self._collector_stop_event = None

        self._result_buffer_queue = None
        self._collector_puller_thread = None
        self._collector_stop_flag = None

        self._all_subtopics: List[Tuple[str, str]] = []
        self._processed_subtopics: Set[Tuple[str, str]] = set()
        self._scheduled_subtopics: Set[Tuple[str, str]] = set()
        self._generated_queries: Set[str] = set()
        for tcfg in self.cfg.get("topics") or []:
            topic = tcfg.get("name")
            if not topic:
                continue
            for sub in tcfg.get("seed_queries") or []:
                if not sub:
                    continue
                self._all_subtopics.append((topic, sub))

        if torch is None or not torch.cuda.is_available():
            print("[Pipeline] CUDA not available, using single OCR on CPU")
            use_multiprocess = False
            use_multi_gpu = False
            try:
                ocr0 = OCR(cfg, device_id=-1)
                self.ocr_instances.append(("OCR-CPU", ocr0))
                self.ocr = ocr0
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OCR: {e }")

        elif use_multi_gpu and use_multiprocess:
            print("[Pipeline] Using multi-process multi-GPU OCR mode")
            num_gpus = torch.cuda.device_count()
            actual_gpus = min(num_gpus, max_ocr_instances)
            print(f"[Pipeline] Detected {num_gpus } GPUs, using {actual_gpus } GPU(s)")
            self.ocr_multiprocess_pool = MultiProcessOCRPool(cfg)
            self.ocr_multiprocess_pool.start(
                num_gpus=actual_gpus, instances_per_gpu=instances_per_gpu
            )
            self.ocr = None
            self.ocr_instances = []
            total_workers = actual_gpus * instances_per_gpu
            self.ocr_workers = total_workers
            print(
                f"[Pipeline] Multi-process OCR pool initialized: {total_workers } workers"
            )

        else:
            print("[Pipeline] Using single-process OCR mode")
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print("[Pipeline] Single GPU mode, using GPU:0")
            try:
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
                ocr_instance = OCR(cfg, device_id=0)
                self.ocr_instances.append(("OCR-GPU0", ocr_instance))
                self.ocr = ocr_instance
                print("[Pipeline] OCR initialized on GPU:0")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OCR on GPU:0: {e }")

            if not self.ocr_instances:
                raise RuntimeError("No OCR instance initialized successfully!")

            self.ocr_workers = min(
                len(self.ocr_instances),
                ocr_cfg.get("max_workers", len(self.ocr_instances)),
            )
            print(
                f"[Pipeline] OCR workers set to {self .ocr_workers } (instances: {len (self .ocr_instances )})"
            )

        print("[Pipeline] OCR initialization complete")

        self.qa = QualityAssessor(cfg)
        self.semantic = SemanticChecker(cfg)
        self.text_checker = TextConsistencyChecker(cfg)
        self.dedup = HashDeduplicator(cfg, str(self.gen_dir))
        self.writer = DatasetWriter(cfg)
        self.writer.init_files()
        self.rej_tracker = RejectTracker()
        rv_cfg = cfg.get("rejection_vector", {})
        self.rejection_vector_tracker: Optional[RejectionVectorRoundTracker] = None
        log_root = ensure_dir(
            pathlib.Path(
                paths.get("log_dir", str(self.accepted_root / "logs"))
            ).resolve()
        )
        critic_cfg = self.cfg.get("critic", {}) or {}
        self.critic: Optional[Critic] = None
        if critic_cfg.get("enabled", True):
            try:
                self.critic = Critic(
                    self.cfg,
                    log_root,
                    agents=self.agents,
                    qa=self.qa,
                    semantic=self.semantic,
                    text_checker=self.text_checker,
                )
            except Exception as e:
                print(f"[Critic] init error: {e }")
                self.critic = None
        if self.critic:
            self.critic.init_prompt_critic_log(log_root)
        if rv_cfg.get("enabled", True):
            rv_dir = ensure_dir(log_root / "Rejection Vector")
            self.rejection_vector_tracker = None

        self.accepted_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._accepted_counts_lock = threading.Lock()
        for tcfg in self.cfg.get("topics") or []:
            topic = tcfg.get("name")
            if not topic:
                continue
            for sub in tcfg.get("seed_queries") or []:
                if not sub:
                    continue
                self.accepted_counts[(topic, sanitize_subtopic_name(sub))] = 0

        self.queries_queue = None
        self.downloaded_queue = None
        self.ocr_queue = None
        self._max_ocr_pending = None
        self.quality_queue = None
        self.semantic_queue = None
        self._ocr_sem = None
        self.dedup_lock = None
        self._async_inited = False
        self._seen_urls = None
        self._seen_urls_lock = None

        self.loop_totals: Dict[str, int] = {
            k: 0
            for k in [
                "urls",
                "downloaded",
                "accepted",
                "rejected",
                "quality_rejected",
                "semantic_rejected",
                "text_consistency_rejected",
                "pHash_rejected",
                "dedup_error",
                "watermark_rejected",
                "ocr_submit_error",
                "generated",
                "generated_accepted",
                "generated_rejected",
            ]
        }
        self.start_ts = time.time()
        self.metrics_cfg = cfg.get("metrics") or {}
        self._metrics_task: Optional[asyncio.Task] = None
        self._last_metrics_snap = {
            "accepted": 0,
            "rejected": 0,
            "downloaded": 0,
            "t": self.start_ts,
        }

        cc = cfg.get("concurrency", {})
        self._queue_cfg = cc.get("max_queue_sizes", {})
        self.enable_async = cc.get("enable_async", True)
        self.query_workers = cc.get("query_workers", 3)
        legacy_eval = cc.get("eval_workers")
        self.quality_workers = cc.get("quality_workers") or (
            legacy_eval // 3 if legacy_eval else 4
        )
        self.semantic_workers = cc.get("semantic_workers") or (
            legacy_eval - 2 * (legacy_eval // 3) if legacy_eval else 4
        )
        self.batch_size = cc.get("batch_write_size", 25)
        self._batch_buffer: List[Dict[str, Any]] = []
        self._sentinel = object()

        self.dry_run = cc.get("dry_run", False)
        self._limit_topics = cc.get("test_max_topics")
        self._limit_subtopics = cc.get("test_max_subtopics")
        self._limit_queries = cc.get("test_max_queries")
        self._limit_images_per_query = cc.get("test_max_images_per_query")

        rl = cc.get("rate_limits", {})
        self._query_bucket = (
            AsyncTokenBucket(rl.get("query_qps", 2.0))
            if rl.get("query_qps", 0) > 0
            else None
        )

        self._dedup_scope = self.cfg.get("hash_deduplication", {}).get("scope", "topic")

    def _remaining_subtopics_count(self) -> int:
        try:
            all_pairs = list(self._all_subtopics)
            rem = [
                p
                for p in all_pairs
                if p not in self._processed_subtopics
                and p not in self._scheduled_subtopics
            ]
            return len(rem)
        except Exception:
            return 0

    def _all_stage_queues_empty(self) -> bool:
        """Return True when every stage queue is empty (or not yet created)."""
        qs = [
            self.queries_queue,
            self.downloaded_queue,
            self.ocr_queue,
            self.quality_queue,
            self.semantic_queue,
        ]
        ok = True
        for q in qs:
            if q is not None:
                try:
                    if not q.empty():
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
        return ok

    def _init_async_structures(self):
        """Initialize async queues, events, and background tasks."""
        if self._async_inited:
            return

        mq = self._queue_cfg

        self.queries_queue = asyncio.PriorityQueue(maxsize=mq.get("queries", 300))
        self.downloaded_queue = asyncio.Queue(maxsize=mq.get("downloaded", 800))
        self.ocr_queue = asyncio.Queue(maxsize=mq.get("ocr", 800))
        self.quality_queue = asyncio.Queue(maxsize=mq.get("quality", 800))
        self.semantic_queue = asyncio.Queue(maxsize=mq.get("semantic", 800))

        self._max_ocr_pending = int(self.cfg.get("ocr", {}).get("max_pending", 2500))

        self.stop_event = asyncio.Event()
        self.feedback_event = asyncio.Event()
        self.round_completed_event = asyncio.Event()

        self._collector_stop_event = asyncio.Event()

        rv_cfg = self.cfg.get("rejection_vector", {})
        if rv_cfg.get("enabled", True):
            paths = self.cfg.get("paths", {})
            log_root = ensure_dir(
                pathlib.Path(
                    paths.get("log_dir", str(self.accepted_root / "logs"))
                ).resolve()
            )
            rv_dir = ensure_dir(log_root / "Rejection Vector")
            self.rejection_vector_tracker = RejectionVectorRoundTracker(
                rv_cfg, rv_dir, critic=self.critic, event=self.round_completed_event
            )

        if self.ocr_multiprocess_pool:
            self._result_buffer_queue = asyncio.Queue(maxsize=500)
            self._collector_stop_flag = threading.Event()

        if self.ocr_multiprocess_pool:
            self._ocr_sem = asyncio.Semaphore(self.ocr_workers)
        else:
            self._ocr_sem = asyncio.Semaphore(len(self.ocr_instances))
            if self.ocr_queue_pool is None:
                self.ocr_queue_pool = queue.Queue(maxsize=len(self.ocr_instances))
                for _, inst in self.ocr_instances:
                    self.ocr_queue_pool.put(inst)

        self.dedup_lock = asyncio.Lock()
        self._seen_urls: Dict[str, set] = defaultdict(set)
        self._seen_urls_lock = asyncio.Lock()

        dq_cfg = self.cfg.get("dynamic_query", {})
        self.low_watermark = int(dq_cfg.get("low_watermark", 0) or 0)
        self._dq_max_wait_sec = float(dq_cfg.get("max_wait_sec", 0) or 0)
        self._dq_max_empty_cycles = int(dq_cfg.get("max_empty_cycles", 0) or 0)
        self._dq_fallback_strategy = str(dq_cfg.get("fallback_strategy", "static"))
        self._dq_last_activity_ts = time.time()
        self._dq_empty_cycles = 0
        self._dq_fallback_triggered = False
        self._enqueue_lock = asyncio.Lock()

        self._feedback_task = asyncio.create_task(self._feedback_loop())

        self._async_inited = True
        print("[Pipeline] Async primitives initialized.")

    async def run(self):
        """Run the main pipeline: crawl phase followed by optional generation phase."""
        try:
            if not self.enable_async:
                print("[Pipeline] enable_async=False -> legacy pipeline")
                await self._run_legacy()
                self._print_metrics_snapshot(final=True)
                print(
                    f"[RejectSummary] {json .dumps (self .rej_tracker .summary (),ensure_ascii =False )}"
                )
                if self.rejection_vector_tracker:
                    self.rejection_vector_tracker.finalize()
                try:
                    self.dedup.flush()
                    self.dedup.save_stats()
                    print("[Pipeline] Dedup index flushed.")
                except Exception as e:
                    print(f"[Pipeline] Dedup flush error: {e }")
                return

            self._init_async_structures()

            if self.ocr_multiprocess_pool is not None:
                self._ocr_result_collector_task = asyncio.create_task(
                    self._ocr_result_collector()
                )

            print("[Pipeline] Starting async multi-stage pipeline ...")

            if self.metrics_cfg.get("log_interval_sec", 15) > 0:
                self._metrics_task = asyncio.create_task(self._metrics_loop())

            await self._run_async()

            print("[Pipeline] Async stage finished.")

            self._clear_all_cuda_memory()

            gen_cfg = self.cfg.get("generation", {}) or {}
            pipe_cfg = self.cfg.get("pipeline", {}) or {}
            generation_enabled = bool(gen_cfg.get("enabled", True))
            skip_generation = bool(pipe_cfg.get("skip_generation", False)) or (
                not generation_enabled
            )
            if skip_generation:
                print(
                    "[Pipeline] Generation disabled by config; exiting after crawling."
                )
            else:
                print("[Pipeline] Enter generation phase ...")
                await self._plan_prompts_and_generate()

        finally:
            await self.shutdown()

            if self._metrics_task:
                self._metrics_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._metrics_task

            self._print_metrics_snapshot(final=True)

            if self.rejection_vector_tracker:
                self.rejection_vector_tracker.finalize()

            try:
                self.dedup.flush()
                self.dedup.save_stats()
                print("[Pipeline] Dedup index flushed.")
            except Exception as e:
                print(f"[Pipeline] Dedup flush error: {e }")

            print("[Pipeline] Finished all phases.")

    async def shutdown(self):
        """Cancel all background tasks, stop the collector, and tear down async resources."""
        print("[Pipeline] Shutting down async tasks...")

        if self.stop_event:
            self.stop_event.set()

        if hasattr(self, "_feedback_task") and self._feedback_task:
            self._feedback_task.cancel()
            try:
                await self._feedback_task
            except asyncio.CancelledError:
                pass

        if self._collector_stop_flag:
            self._collector_stop_flag.set()

        if self._collector_puller_thread:
            self._collector_puller_thread.join(timeout=5)

        print("[Pipeline] Background tasks stopped.")

    async def _feedback_loop(self):
        """Listen for round-completion events and dynamically generate queries for unprocessed subtopics."""
        await feedback_loop(self)

    async def _metrics_loop(self):
        """Periodically print a snapshot of cumulative and incremental performance metrics."""
        await metrics_loop(self)

    def _print_metrics_snapshot(self, final: bool = False):
        """Print current cumulative and incremental performance metrics."""
        print_metrics_snapshot(self, final=final)

    def _record_sample_outcome(
        self, meta: Optional[Dict[str, Any]], reasons: List[str]
    ):
        """Feed decision data into the rejection-vector tracker."""
        meta = meta or {}
        raw_path = meta.get("path") or meta.get("image_path")
        if isinstance(raw_path, pathlib.Path):
            path = str(raw_path)
        else:
            path = str(raw_path) if raw_path else "unknown"

        self.rej_tracker.add(path, reasons)
        if self.rejection_vector_tracker:
            payload = {
                "path": path,
                "ocr": meta.get("ocr"),
                "quality": meta.get("quality"),
                "semantic": meta.get("semantic"),
                "text_consistency": meta.get("text_consistency"),
                "query": meta.get("query"),
            }
            self.rejection_vector_tracker.record(payload, reasons)

    async def _run_async(self):
        """Launch all async stage workers and drain every queue until completion."""
        print("[Pipeline-Flow] _run_async started. Creating workers...")
        query_tasks = [
            asyncio.create_task(self._query_worker(i))
            for i in range(self.query_workers)
        ]
        dispatch_task = asyncio.create_task(self._download_dispatcher())
        ocr_tasks = [
            asyncio.create_task(self._ocr_worker(i)) for i in range(self.ocr_workers)
        ]
        quality_tasks = [
            asyncio.create_task(self._quality_worker(i))
            for i in range(self.quality_workers)
        ]
        semantic_tasks = [
            asyncio.create_task(self._semantic_worker(i))
            for i in range(self.semantic_workers)
        ]
        print(
            f"[Pipeline-Flow] All workers created. Query={self .query_workers }, OCR={self .ocr_workers }, Quality={self .quality_workers }, Semantic={self .semantic_workers }."
        )

        print("[Pipeline-Flow] Enqueuing queries...")
        await self._enqueue_all_queries()
        print("[Pipeline-Flow] Query enqueue complete")

        dq_cfg = self.cfg.get("dynamic_query", {}) or {}
        if int(dq_cfg.get("low_watermark", 0)) > 0:
            self._queue_monitor_task = asyncio.create_task(
                self._queue_monitor_task_func()
            )

        print("[Pipeline-Flow] Waiting for query queue to drain (Dynamic Mode)...")
        while True:
            await self.queries_queue.join()
            if self.stop_event.is_set() and self.queries_queue.empty():
                if (
                    self.rejection_vector_tracker
                    and self.rejection_vector_tracker.processed > 0
                ):
                    print(
                        f"[Pipeline-Flow] Finalizing incomplete round before shutting down (processed={self .rejection_vector_tracker .processed })."
                    )
                    self.rejection_vector_tracker.finalize()
                print(
                    "[Pipeline-Flow] Stop event set and query queue empty. Exiting query phase."
                )
                break
            if self.queries_queue.empty():
                if (
                    self.rejection_vector_tracker
                    and self.rejection_vector_tracker.processed > 0
                ):
                    print(
                        f"[Pipeline-Flow] Queue empty mid-round (processed={self .rejection_vector_tracker .processed }); forcing finalize to trigger feedback generation."
                    )
                    self.rejection_vector_tracker.finalize()
                print(
                    "[Pipeline-Flow] Waiting for feedback loop / LLM to generate more queries..."
                )
                for _ in range(60):
                    if not self.queries_queue.empty() or (
                        self.stop_event.is_set() and self.queries_queue.empty()
                    ):
                        break
                    await asyncio.sleep(1.0)
                if self.stop_event.is_set() and self.queries_queue.empty():
                    print(
                        "[Pipeline-Flow] Stop set during wait and no new queries; exiting query loop."
                    )
                    break
                if not self.queries_queue.empty():
                    print(
                        "[Pipeline-Flow] New queries populated; continuing processing."
                    )
                    continue
                remaining = self._remaining_subtopics_count()
                if remaining == 0 and self._all_stage_queues_empty():
                    print(
                        "[Pipeline-Flow] All subtopics completed and all queues empty. Forcing stop."
                    )
                    self.stop_event.set()
                    break
                print(
                    "[Pipeline-Flow] No new queries after extended wait; re-checking stop status."
                )
                continue

        print("[Pipeline-Flow] Query queue drained.")
        for _ in range(self.query_workers):
            await self.queries_queue.put((99, time.time(), self._sentinel))
        await asyncio.gather(*query_tasks)
        if self._queue_monitor_task:
            self._queue_monitor_task.cancel()
            with contextlib.suppress(Exception):
                await self._queue_monitor_task

        if self._feedback_task:
            self._feedback_task.cancel()
            try:
                await self._feedback_task
            except asyncio.CancelledError:
                pass

        print("[Pipeline-Flow] Waiting for download queue to drain...")
        await self.downloaded_queue.join()
        print("[Pipeline-Flow] Download queue drained.")
        await self.downloaded_queue.put(self._sentinel)
        await dispatch_task

        print("[Pipeline-Flow] Waiting for OCR queue to drain...")
        await self.ocr_queue.join()
        print("[Pipeline-Flow] OCR queue drained.")

        for _ in range(self.ocr_workers):
            await self.ocr_queue.put(self._sentinel)

        await asyncio.gather(*ocr_tasks)

        self._ocr_workers_done = True
        print("[Pipeline-Flow] All OCR workers exited, signaling collector...")

        while self._ocr_pending and not self._ocr_result_collector_task.done():
            await asyncio.sleep(0.5)
            print(
                f"[Pipeline-Flow] Waiting for OCR collector to drain {len (self ._ocr_pending )} pending tasks..."
            )

        self._collector_stop_event.set()

        print("[Pipeline-Flow] Waiting for quality queue to drain...")
        await self.quality_queue.join()
        print("[Pipeline-Flow] Quality queue drained.")
        for _ in range(self.quality_workers):
            await self.quality_queue.put(self._sentinel)
        await asyncio.gather(*quality_tasks)

        print("[Pipeline-Flow] Waiting for semantic queue to drain...")
        await self.semantic_queue.join()
        print("[Pipeline-Flow] Semantic queue drained.")
        for _ in range(self.semantic_workers):
            await self.semantic_queue.put(self._sentinel)
        await asyncio.gather(*semantic_tasks)
        await self._flush_batch(force=True)
        print(f"[Totals] {json .dumps (self .loop_totals ,ensure_ascii =False )}")
        print(
            f"[RejectSummary] {json .dumps (self .rej_tracker .summary (),ensure_ascii =False )}"
        )

    def _print_queue_head(self):
        """Debug: Print top 20 queries in the queue."""
        if not self.queries_queue:
            return
        try:
            import heapq

            q_copy = list(self.queries_queue._queue)
            top_items = heapq.nsmallest(20, q_copy)
            # print(
            #     f"\n[QueueDebug] Top {len (top_items )}/{self .queries_queue .qsize ()} pending queries:"
            # )
            # for i, item in enumerate(top_items):
            #     prio, ts, (t, s, q) = item
            #     print(f"  {i +1 }. [{prio }] {t }/{s }: {q }")
            # print("-" * 40 + "\n")
            pass
        except Exception as e:
            # print(f"[QueueDebug] Error inspecting queue: {e }")
            pass

    async def _enqueue_all_queries(self):
        """Initial warmup enqueue: only a subset of subtopics before dynamic feedback takes over."""
        await enqueue_all_queries(self)

    async def _queue_monitor_task_func(self):
        """Continuously monitor the queries queue and refill when below the low-water mark."""
        await queue_monitor(self)

    async def _apply_dq_fallback(self):
        """Execute the configured fallback strategy when the query queue is starved."""
        await apply_dq_fallback(self)

    async def _fallback_enqueue_static_queries(self):
        """Enqueue seed queries from config for remaining unprocessed subtopics."""
        await fallback_enqueue_static(self)

    async def _fallback_enqueue_warmup_queries(self):
        """Use warmup-phase query generation logic to enqueue queries for remaining subtopics."""
        await fallback_enqueue_warmup(self)

    def _generate_queries(self, topic: str, subtopic: str) -> Tuple[List[str], bool]:
        """Generate queries and return (queries, is_strategy_used)."""
        return generate_queries(self, topic, subtopic)

    def _call_agent_list(self, name: str, prompt: str) -> List[str]:
        """Invoke a named agent with a prompt and return its list output."""
        return call_agent_list(self, name, prompt)

    async def _query_worker(self, idx: int):
        """Fetch images for queued queries and push downloaded file paths to the download queue."""
        await query_worker(self, idx)

    async def _download_dispatcher(self):
        """Forward downloaded items from the download queue to the OCR queue."""
        await download_dispatcher(self)

    async def _ocr_result_collector(self):
        """Aggregate OCR results from the multi-process pool and route them to the quality queue."""
        await ocr_result_collector(self)

    async def _ocr_worker(self, idx: int):
        """Submit images for OCR processing and forward results to the quality queue."""
        await ocr_worker(self, idx)

    async def _quality_worker(self, idx: int):
        """Assess image quality and watermark-check, then forward to semantic queue."""
        await quality_worker(self, idx)

    async def _semantic_worker(self, idx: int):
        """Perform pHash dedup, semantic and text-consistency checks, generate captions, and buffer accepted items."""
        await semantic_worker(self, idx)

    async def _flush_batch(self, force: bool = False):
        """Write the current buffer of accepted records to dataset and disk."""
        await flush_batch(self, force=force)

    def _clear_all_cuda_memory(self):
        """Best-effort cleanup of all CUDA device caches before generation."""
        if torch is None or not torch.cuda.is_available():
            return
        try:
            device_count = torch.cuda.device_count()
            for did in range(device_count):
                with torch.cuda.device(did):
                    torch.cuda.empty_cache()
                    with contextlib.suppress(Exception):
                        torch.cuda.ipc_collect()
            print(f"[Pipeline] Cleared CUDA caches on {device_count } device(s)")
        except Exception as e:
            print(f"[Pipeline][Warn] Failed to clear CUDA memory: {e }")

    def _save_accepted(self, item: Dict[str, Any]) -> Optional[str]:
        """Save an accepted (crawled) image and its metadata to the accepted directory."""
        return save_accepted(self, item)

    def _sanitize_caption(self, cap: Optional[str]) -> str:
        """Clean a caption string into compact single-line form."""
        return sanitize_caption(cap)

    def _init_filename_counter(self):
        """Scan existing files to initialize the sequential filename counter."""
        images_dir = pathlib.Path(self.accepted_images_dir)
        self._filename_counter = sum(1 for f in images_dir.iterdir() if f.is_file())
        print(f"[Pipeline] Filename counter initialized: {self ._filename_counter }")

    def _init_gen_filename_counter(self):
        """Scan generated images directory to initialize the gen filename counter."""
        root = pathlib.Path(self.gen_accepted_root)
        if root.exists():
            self._gen_filename_counter = sum(1 for _ in root.glob("gen_*.jpg"))
        else:
            self._gen_filename_counter = 0
        print(f"[Pipeline] Gen filename counter initialized: {self ._gen_filename_counter }")

    def _next_filename_sync(self, ext: str) -> str:
        """Allocate the next sequential filename (thread-safe)."""
        with self._filename_lock_sync:
            self._filename_counter += 1
            return f"img_{self ._filename_counter :07d}{ext }"

    def _next_gen_filename_sync(self, ext: str) -> str:
        """Allocate the next sequential filename for generated images (thread-safe)."""
        with self._gen_filename_lock_sync:
            self._gen_filename_counter += 1
            return f"gen_{self ._gen_filename_counter :07d}{ext }"

    def _alloc_filename(self, ext: str) -> Optional[str]:
        """Allocate a filename according to the configured strategy."""
        return alloc_filename(self, ext)

    def _content_hash_filename(self, fp: pathlib.Path) -> str:
        """Generate a unique filename based on file content hash."""
        return content_hash_filename(fp)

    def _unique_final(self, name: str) -> str:
        """Ensure filename uniqueness by appending a random suffix if a conflict exists."""
        return unique_final(self, name)

    def _write_image_index_entry(
        self, filename: str, topic: str, subtopic: str, query: str
    ) -> bool:
        """Append an image record to the index file."""
        return write_image_index_entry(self, filename, topic, subtopic, query)

    async def _plan_prompts_and_generate(self):
        """Run count analysis, plan prompts, generate images, and post-process results."""
        await plan_prompts_and_generate(self)

    async def _post_process_generated(self, images: List[Dict[str, Any]]):
        """Screen generated images through quality, semantic, and text-consistency checks."""
        await post_process_generated(self, images)

    def _save_generated(self, item: Dict[str, Any], img_path: pathlib.Path):
        """Save a generated image and its metadata to the gen_accepted directory."""
        save_generated(self, item, img_path)


__all__ = ["OrchestratorPipeline"]
