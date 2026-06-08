from __future__ import annotations
import asyncio, json, time, pathlib, hashlib, random
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import shutil, os, math
import contextlib
import threading
import queue
import uuid
import multiprocessing as mp

try:
    import torch
except ImportError:
    torch = None
try:
    import psutil

    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

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
from src.annotation.qwen_vl import generate_image_caption
from src.tools.qwen_image import generate_qwen_images
from src.critic import Critic
from src.dataset.writer import DatasetWriter
from src.tools.bing_noapi import async_fetch_and_download
from src.utils.common import ensure_dir, sha1_of_file, sanitize_subtopic_name
from src.utils.trackers import (
    RejectTracker,
    RejectionVectorRoundTracker,
    AsyncTokenBucket,
)


class OrchestratorPipeline:
    """Enhanced asynchronous multi-stage pipeline with legacy fallback."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        agents: Dict[str, Any],
        shard_id: str,
        keep_rejects: bool = True,
        config_path: str = "config.yaml",
    ):
        """Initialize pipeline directories, OCR, tools, and queue placeholders."""
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
                raise RuntimeError(f"Failed to initialize OCR: {e}")

        elif use_multi_gpu and use_multiprocess:
            print("[Pipeline] Using multi-process multi-GPU OCR mode")
            num_gpus = torch.cuda.device_count()
            actual_gpus = min(num_gpus, max_ocr_instances)
            print(f"[Pipeline] Detected {num_gpus} GPUs, using {actual_gpus} GPU(s)")
            self.ocr_multiprocess_pool = MultiProcessOCRPool(cfg)
            self.ocr_multiprocess_pool.start(
                num_gpus=actual_gpus, instances_per_gpu=instances_per_gpu
            )
            self.ocr = None
            self.ocr_instances = []
            total_workers = actual_gpus * instances_per_gpu
            self.ocr_workers = total_workers
            print(
                f"[Pipeline] Multi-process OCR pool initialized: {total_workers} workers"
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
                raise RuntimeError(f"Failed to initialize OCR on GPU:0: {e}")

            if not self.ocr_instances:
                raise RuntimeError("No OCR instance initialized successfully!")

            self.ocr_workers = min(
                len(self.ocr_instances),
                ocr_cfg.get("max_workers", len(self.ocr_instances)),
            )
            print(
                f"[Pipeline] OCR workers set to {self.ocr_workers} (instances: {len(self.ocr_instances)})"
            )

        print(f"[Pipeline] OCR initialization complete")

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
                print(f"[Critic] init error: {e}")
                self.critic = None
        if rv_cfg.get("enabled", True):
            rv_dir = ensure_dir(log_root / "Rejection Vector")
            self.rejection_vector_tracker = None
        self._critic_threshold_overrides: Dict[str, float] = {}

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
            import threading

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
        from collections import defaultdict

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
        """Run the main pipeline: async crawl phase followed by optional generation phase."""
        try:

            if not self.enable_async:
                print("[Pipeline] enable_async=False -> legacy pipeline")
                await self._run_legacy()
                self._print_metrics_snapshot(final=True)
                print(
                    f"[RejectSummary] {json.dumps(self.rej_tracker.summary(), ensure_ascii=False)}"
                )
                if self.rejection_vector_tracker:
                    self.rejection_vector_tracker.finalize()
                try:
                    self.dedup.flush()
                    self.dedup.save_stats()
                    print("[Pipeline] Dedup index flushed.")
                except Exception as e:
                    print(f"[Pipeline] Dedup flush error: {e}")
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
                print(f"[Pipeline] Dedup flush error: {e}")

            print("[Pipeline] Finished all phases.")

    async def shutdown(self):
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
        critic_strategy = (self.cfg.get("critic", {}) or {}).get("strategy", {}) or {}
        warmup_rounds = int(critic_strategy.get("warmup_rounds", 2) or 2)
        fb_cfg = self.cfg.get("feedback_query", {}) or {}
        per_round_limit = int(fb_cfg.get("max_llm_calls_per_loop", 5) or 5)
        dq_cfg = self.cfg.get("dynamic_query", {}) or {}

        print(
            f"[FeedbackLoop] started (warmup_rounds={warmup_rounds}, per_round_limit={per_round_limit})"
        )
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(self.round_completed_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                remaining = self._remaining_subtopics_count()
                if remaining == 0 and self._all_stage_queues_empty():
                    print(
                        "[FeedbackLoop] All subtopics completed & queues empty; signaling stop_event."
                    )
                    self.stop_event.set()
                    break
                continue
            self.round_completed_event.clear()

            try:
                if self.rejection_vector_tracker and getattr(
                    self.rejection_vector_tracker, "_last_payload", None
                ):
                    pl = self.rejection_vector_tracker._last_payload or {}
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
                print(f"[FeedbackLoop] Print round summary error: {e}")

            if self.critic and self.critic.should_use_new_strategy():
                try:
                    latest = self.critic.latest_strategy or {}
                    strat = latest.get("strategy") or {}
                    strat_queries: List[str] = strat.get("queries") or []
                    added = 0
                    if strat_queries:
                        async with self._enqueue_lock:
                            targets = [
                                p
                                for p in self._all_subtopics
                                if p not in self._processed_subtopics
                                and p not in self._scheduled_subtopics
                            ]
                            if not targets:
                                targets = list(self._all_subtopics)
                            for topic, sub in targets:
                                for q in strat_queries:
                                    if not isinstance(q, str) or not q.strip():
                                        continue
                                    if q in self._generated_queries:
                                        continue
                                    try:
                                        self.queries_queue.put_nowait(
                                            (1, -time.time(), (topic, sub, q))
                                        )
                                        self._generated_queries.add(q)
                                        added += 1
                                    except asyncio.QueueFull:
                                        break
                            if added > 0:
                                print(
                                    f"[FeedbackLoop] Enqueued {added} queries from latest strategy."
                                )
                                self._print_queue_head()
                except Exception as e:
                    print(f"[FeedbackLoop] Strategy enqueue error: {e}")

            if not self.rejection_vector_tracker or not getattr(
                self.rejection_vector_tracker, "last_round_details", None
            ):
                continue
            details = self.rejection_vector_tracker.last_round_details
            rid = details.get("round_index", 0)
            if rid <= warmup_rounds:
                print(
                    f"[FeedbackLoop] Round {rid} <= warmup ({warmup_rounds}) warmup mode: using base query generation."
                )
            async with self._enqueue_lock:
                remaining = [
                    p
                    for p in self._all_subtopics
                    if p not in self._processed_subtopics
                    and p not in self._scheduled_subtopics
                ]
                if (
                    remaining
                    and self.critic
                    and hasattr(self.critic, "set_next_subtopics")
                ):
                    with contextlib.suppress(Exception):
                        preview_limit = max(per_round_limit, 5)
                        preview_subs = [s for _, s in remaining[:preview_limit]]
                        self.critic.set_next_subtopics(preview_subs)
                        print(
                            f"[FeedbackLoop] set_next_subtopics preview={preview_subs}"
                        )
                if not remaining:
                    if self.queries_queue.empty():
                        print("[FeedbackLoop] No remaining subtopics; signaling stop.")
                        self.stop_event.set()
                    break

                batch = remaining[:per_round_limit]
                added_subtopics = 0
                for topic, sub in batch:
                    if self.stop_event.is_set():
                        break
                    queries, is_strategy = await asyncio.to_thread(
                        self._generate_queries, topic, sub
                    )
                    if self._limit_queries:
                        queries = queries[: self._limit_queries]
                    added_any = False
                    priority = 1 if is_strategy else 10
                    for q in queries:
                        if q in self._generated_queries:
                            continue
                        while True:
                            try:
                                self.queries_queue.put_nowait(
                                    (priority, -time.time(), (topic, sub, q))
                                )
                                self._generated_queries.add(q)
                                added_any = True
                                break
                            except asyncio.QueueFull:
                                await asyncio.sleep(0.05)
                    self._processed_subtopics.add((topic, sub))
                    if added_any:
                        added_subtopics += 1
                        self._print_queue_head()
            print(
                f"[FeedbackLoop] Round {rid} added {added_subtopics} subtopics; processed={len(self._processed_subtopics)}/{len(self._all_subtopics)}"
            )
        print("[FeedbackLoop] exiting.")

    async def _metrics_loop(self):
        """Periodically print a snapshot of cumulative and incremental performance metrics."""
        interval = float(self.metrics_cfg.get("log_interval_sec", 15))
        while True:
            await asyncio.sleep(interval)
            try:
                self._print_metrics_snapshot()
            except Exception as e:
                print(f"[MetricsLoopError] {e}")

    def _print_metrics_snapshot(self, final: bool = False):
        """Print current cumulative and incremental performance metrics."""
        now = time.time()
        elapsed = now - self.start_ts
        tot = self.loop_totals
        checked = tot["accepted"] + tot["rejected"]
        acc_rate = (tot["accepted"] / checked) if checked else 0.0
        delta_downloaded = tot["downloaded"] - self._last_metrics_snap["downloaded"]
        delta_accepted = tot["accepted"] - self._last_metrics_snap["accepted"]
        delta_rejected = tot["rejected"] - self._last_metrics_snap["rejected"]
        delta_t = max(now - self._last_metrics_snap["t"], 1e-6)
        speed_download = delta_downloaded / delta_t
        speed_accept = delta_accepted / delta_t
        queue_sizes = {}
        if self.metrics_cfg.get("enable_queue_sizes", True):

            queue_sizes = {
                "Q_queries": self.queries_queue.qsize() if self.queries_queue else 0,
                "Q_downloaded": (
                    self.downloaded_queue.qsize() if self.downloaded_queue else 0
                ),
                "Q_ocr": self.ocr_queue.qsize() if self.ocr_queue else 0,
                "Q_quality": self.quality_queue.qsize() if self.quality_queue else 0,
                "Q_semantic": self.semantic_queue.qsize() if self.semantic_queue else 0,
            }
        mem_info = {}
        if _HAS_PSUTIL and self.metrics_cfg.get("enable_memory", False):
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
        print(f"[Metrics-{tag}] {json.dumps(line, ensure_ascii=False)}")
        self._last_metrics_snap.update(
            {
                "accepted": tot["accepted"],
                "rejected": tot["rejected"],
                "downloaded": tot["downloaded"],
                "t": now,
            }
        )

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
        """启动并协调各异步工作协程直到全部队列耗尽."""
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
            f"[Pipeline-Flow] All workers created. Query={self.query_workers}, OCR={self.ocr_workers}, Quality={self.quality_workers}, Semantic={self.semantic_workers}."
        )

        print("[Pipeline-Flow] 查询入队...")
        await self._enqueue_all_queries()
        print("[Pipeline-Flow] 查询入队完成")

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
                        f"[Pipeline-Flow] Finalizing incomplete round before shutting down (processed={self.rejection_vector_tracker.processed})."
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
                        f"[Pipeline-Flow] Queue empty mid-round (processed={self.rejection_vector_tracker.processed}); forcing finalize to trigger feedback generation."
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
                f"[Pipeline-Flow] Waiting for OCR collector to drain {len(self._ocr_pending)} pending tasks..."
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
        print(f"[Totals] {json.dumps(self.loop_totals, ensure_ascii=False)}")
        print(
            f"[RejectSummary] {json.dumps(self.rej_tracker.summary(), ensure_ascii=False)}"
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
            #     f"\n[QueueDebug] Top {len(top_items)}/{self.queries_queue.qsize()} pending queries:"
            # )
            # for i, item in enumerate(top_items):
            #     prio, ts, (t, s, q) = item
            #     print(f"  {i+1}. [{prio}] {t}/{s}: {q}")
            # print("-" * 40 + "\n")
            pass
        except Exception as e:
            # print(f"[QueueDebug] Error inspecting queue: {e}")
            pass

    async def _enqueue_all_queries(self):
        """Initial warmup enqueue: only a subset of subtopics before dynamic feedback takes over."""
        critic_strategy = (self.cfg.get("critic", {}) or {}).get("strategy", {}) or {}
        warmup_rounds = int(critic_strategy.get("warmup_rounds", 5) or 5)
        fb_cfg = self.cfg.get("feedback_query", {}) or {}
        initial_limit = int(fb_cfg.get("initial_warmup_subtopics", 3) or 3)
        if self._limit_subtopics:
            initial_limit = min(initial_limit, self._limit_subtopics)
        selected = self._all_subtopics[:initial_limit]
        if self.critic and hasattr(self.critic, "set_next_subtopics"):
            try:
                preview_subs = [
                    s for _, s in self._all_subtopics[: max(initial_limit, 1) * 2]
                ]
                self.critic.set_next_subtopics(preview_subs)
                print(f"[Warmup] set_next_subtopics preview={preview_subs[:10]}")
            except Exception as e:
                print(f"[Warmup] set_next_subtopics error: {e}")
        for topic, sub in selected:
            key = (topic, sub)
            if key in self._processed_subtopics:
                continue
            queries, is_strategy = await asyncio.to_thread(
                self._generate_queries, topic, sub
            )
            if self._limit_queries:
                queries = queries[: self._limit_queries]
            added_any = False
            priority = 1 if is_strategy else 10
            for q in queries:
                if q in self._generated_queries:
                    continue
                while True:
                    try:
                        self.queries_queue.put_nowait(
                            (priority, -time.time(), (topic, sub, q))
                        )
                        self._generated_queries.add(q)
                        added_any = True
                        break
                    except asyncio.QueueFull:
                        await asyncio.sleep(0.05)
            if added_any:
                self._processed_subtopics.add(key)
                self._print_queue_head()
        print(
            f"[Warmup] Enqueued {len(self._processed_subtopics)} subtopics (limit={initial_limit}, warmup_rounds={warmup_rounds})."
        )

    async def _queue_monitor_task_func(self):
        """持续监控 queries_queue，当低于低水位阈值时补充后续子主题查询."""
        dq_cfg = self.cfg.get("dynamic_query", {}) or {}
        low_watermark = int(dq_cfg.get("low_watermark", 30))
        batch_size = int(dq_cfg.get("monitor_batch_size", 3))
        per_subtopic_limit = int(dq_cfg.get("monitor_queries_per_subtopic", 15) or 15)

        if low_watermark <= 0:
            return

        # print(
        #     f"[QueueMonitor] started low_watermark={low_watermark}, batch_size={batch_size}"
        # )

        while not self.stop_event.is_set():
            try:
                current_size = self.queries_queue.qsize()
                if current_size < low_watermark:
                    async with self._enqueue_lock:
                        remaining = [
                            p
                            for p in self._all_subtopics
                            if p not in self._processed_subtopics
                            and p not in self._scheduled_subtopics
                        ]
                        if not remaining:
                            if (
                                self.queries_queue.empty()
                                and self.rejection_vector_tracker
                                and self.rejection_vector_tracker.processed > 0
                            ):
                                # print(
                                #     f"[QueueMonitor] No remaining subtopics; forcing finalize round incomplete processed={self.rejection_vector_tracker.processed}"
                                # )
                                with contextlib.suppress(Exception):
                                    self.rejection_vector_tracker.finalize()
                        else:
                            batch = remaining[:batch_size]
                            # print(
                            #     f"[QueueMonitor] Low watermark ({current_size} < {low_watermark}). Fetching batch of {len(batch)} subtopics..."
                            # )
                            if self.critic and hasattr(
                                self.critic, "set_next_subtopics"
                            ):
                                with contextlib.suppress(Exception):
                                    preview_len = max(batch_size, 5)
                                    preview_subs = [
                                        s for _, s in remaining[:preview_len]
                                    ]
                                    self.critic.set_next_subtopics(preview_subs)
                            for topic, sub in batch:
                                key = (topic, sub)
                                if (
                                    key in self._processed_subtopics
                                    or key in self._scheduled_subtopics
                                ):
                                    continue
                                try:
                                    queries, is_strategy = await asyncio.to_thread(
                                        self._generate_queries, topic, sub
                                    )
                                    if self._limit_queries:
                                        queries = queries[: self._limit_queries]
                                    added_count = 0
                                    priority = 1 if is_strategy else 10
                                    for q in queries:
                                        if q in self._generated_queries:
                                            continue
                                        try:
                                            self.queries_queue.put_nowait(
                                                (
                                                    priority,
                                                    -time.time(),
                                                    (topic, sub, q),
                                                )
                                            )
                                            self._generated_queries.add(q)
                                            added_count += 1
                                        except asyncio.QueueFull:
                                            break
                                    if added_count > 0:
                                        # print(
                                        #     f"[QueueMonitor] Added subtopic {topic}/{sub} queries={added_count}"
                                        # )
                                        self._print_queue_head()
                                    else:
                                        # print(
                                        #     f"[QueueMonitor] Subtopic {topic}/{sub} yielded 0 NEW queries (dups/empty). Marked processed."
                                        # )
                                        pass
                                except Exception as e:
                                    # print(
                                    #     f"[QueueMonitor] Error processing subtopic {topic}/{sub}: {e}"
                                    # )
                                    pass
                                finally:
                                    self._processed_subtopics.add(key)
                                    self._scheduled_subtopics.add(key)

                now_size = self.queries_queue.qsize()
                if now_size <= current_size:
                    self._dq_empty_cycles += 1
                else:
                    self._dq_empty_cycles = 0
                if now_size > current_size:
                    self._dq_last_activity_ts = time.time()

                elapsed = time.time() - self._dq_last_activity_ts
                max_wait_hit = (
                    self._dq_max_wait_sec > 0 and elapsed >= self._dq_max_wait_sec
                )
                max_cycles_hit = (
                    self._dq_max_empty_cycles > 0
                    and self._dq_empty_cycles >= self._dq_max_empty_cycles
                )
                if not self._dq_fallback_triggered and (max_wait_hit or max_cycles_hit):
                    # print(
                    #     f"[QueueMonitor] Fallback triggered (elapsed={round(elapsed,1)}s, empty_cycles={self._dq_empty_cycles})."
                    # )
                    self._dq_fallback_triggered = True
                    await self._apply_dq_fallback()
                    self._dq_empty_cycles = 0
                    self._dq_last_activity_ts = time.time()

                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                # print("[QueueMonitor] cancelled")
                break
            except Exception as e:
                # print(f"[QueueMonitor] error {e}")
                await asyncio.sleep(2.0)

    async def _apply_dq_fallback(self):
        """根据配置执行队列监控回退策略."""
        strategy = (self._dq_fallback_strategy or "static").lower()
        if strategy == "static":
            # print("[QueueMonitor] Applying static fallback: enqueue seed queries.")
            await self._fallback_enqueue_static_queries()
        elif strategy == "warmup":
            # print(
            #     "[QueueMonitor] Applying warmup fallback: generate queries via warmup logic."
            # )
            await self._fallback_enqueue_warmup_queries()
        else:
            # print(f"[QueueMonitor] Unknown fallback strategy '{strategy}', skipping.")
            pass

    async def _fallback_enqueue_static_queries(self):
        """将剩余未处理子主题的 seed_queries 作为静态查询入队."""
        dq_cfg = self.cfg.get("dynamic_query", {}) or {}
        per_subtopic_limit = int(dq_cfg.get("monitor_queries_per_subtopic", 15) or 15)
        batch_limit = int(dq_cfg.get("monitor_batch_size", 3) or 3)
        remaining = [
            pair
            for pair in self._all_subtopics
            if pair not in self._processed_subtopics
        ]
        if not remaining:
            # print("[QueueMonitor] No remaining subtopics for static fallback.")
            return
        selected = remaining[:batch_limit]
        topics_cfg = self.cfg.get("topics") or []
        topic_seed_map = {
            tc.get("name"): (tc.get("seed_queries") or []) for tc in topics_cfg
        }
        added = 0
        for topic, sub in selected:
            seeds = topic_seed_map.get(topic, [])
            if not seeds:
                continue
            for q in seeds[:per_subtopic_limit]:
                key = f"{topic}::{sub}::{q}".strip()
                if key in self._generated_queries:
                    continue
                priority = 0
                item = (priority, -time.time(), (topic, sub, q))
                try:
                    await self.queries_queue.put(item)
                    self._generated_queries.add(key)
                    added += 1
                except Exception:
                    break
        # print(
        #     f"[QueueMonitor] Static fallback enqueued {added} queries across {len(selected)} subtopics."
        # )

    async def _fallback_enqueue_warmup_queries(self):
        """使用暖身阶段生成逻辑为剩余未处理子主题生成并入队查询."""
        dq_cfg = self.cfg.get("dynamic_query", {}) or {}
        per_subtopic_limit = int(dq_cfg.get("monitor_queries_per_subtopic", 15) or 15)
        batch_limit = int(dq_cfg.get("monitor_batch_size", 3) or 3)
        remaining = [
            pair
            for pair in self._all_subtopics
            if pair not in self._processed_subtopics
            and pair not in self._scheduled_subtopics
        ]
        if not remaining:
            # print("[QueueMonitor] No remaining subtopics for warmup fallback.")
            return
        selected = remaining[:batch_limit]
        added_total = 0
        for topic, sub in selected:
            try:
                queries, is_strategy = await asyncio.to_thread(
                    self._generate_queries, topic, sub
                )
                priority = 10
                added = 0
                for q in queries[:per_subtopic_limit]:
                    if q in self._generated_queries:
                        continue
                    try:
                        await self.queries_queue.put(
                            (priority, -time.time(), (topic, sub, q))
                        )
                        self._generated_queries.add(q)
                        added += 1
                    except asyncio.QueueFull:
                        break
                if added > 0:
                    self._processed_subtopics.add((topic, sub))
                    self._scheduled_subtopics.add((topic, sub))
                    added_total += added
                    self._print_queue_head()
            except Exception as e:
                # print(f"[QueueMonitor] Warmup fallback error {topic}/{sub}: {e}")
                pass
        # print(
        #     f"[QueueMonitor] Warmup fallback enqueued {added_total} queries across {len(selected)} subtopics."
        # )

    def _generate_queries(self, topic: str, subtopic: str) -> Tuple[List[str], bool]:
        """Generate queries and return (queries, is_strategy_used)."""
        if (
            self.critic
            and getattr(self.critic, "should_use_query_strategy", None)
            and self.critic.should_use_query_strategy()
        ):
            try:
                critic_queries = self.critic.generate_queries_for(topic, subtopic)
                if critic_queries:
                    print(
                        f"[Critic-Strategy] Generated {len(critic_queries)} queries for {topic}/{subtopic}"
                    )
                    return critic_queries, True
            except Exception as e:
                print(f"[Critic-Strategy] generate_queries_for error: {e}")

        base_prompt = f"Subtheme: {subtopic}\nGenerate <=3 Bing search queries highly relevant to '{subtopic}'."
        var_prompt = f"Generate <=5 diverse search queries for '{subtopic}'."
        base = self._call_agent_list("Retriever", base_prompt) or [subtopic]
        variants = self._call_agent_list("QueryGenerator", var_prompt)
        queries = [q for q in dict.fromkeys((base or []) + (variants or [])) if q] or [
            subtopic
        ]
        return queries, False

    def _call_agent_list(self, name: str, prompt: str) -> List[str]:
        agent = self.agents.get(name)
        if not agent:
            return []
        try:
            values = agent.run_list(prompt) or []
            return [v.strip() for v in values if isinstance(v, str) and v.strip()]
        except Exception as e:
            print(f"[AgentError] {name} prompt='{prompt[:60]}' err={e}")
            return []

    async def _query_worker(self, idx: int):
        """
        执行查询抓取与下载并将结果推入下载队列。

        v3.3 变更：
        [Fix-D]  同一 subtopic 内多个 query 共享 already_seen 集合，
                        避免重复抓取 Bing 对不同 query 返回的相同 URL。
                        集合超过 MAX_SEEN_CAP(3000) 后自动停止过滤，防止误伤新 URL。
        [Fix-F]  每次 query 处理完毕后随机冷却 2-5s，
                        减缓 Bing IP 信誉消耗，降低返回无关内容的概率。
        """
        print(f"[Pipeline-Flow] QueryWorker-{idx} started.")

        while True:
            priority_tuple = await self.queries_queue.get()
            if len(priority_tuple) == 3:
                _, _, item = priority_tuple
            else:
                _, item = priority_tuple

            if item is self._sentinel:
                self.queries_queue.task_done()
                break

            topic, sub, query = item
            print(f"[Pipeline-Flow] QueryWorker-{idx} got query: '{query}'")

            out_dir = ensure_dir(
                self.images_root / self.shard_id / topic / sub.replace("/", "_")
            )

            try:
                if self._query_bucket:
                    await self._query_bucket.acquire()

                start = time.time()

                if self.dry_run:
                    urls = [
                        f"dry://{query}/{i}"
                        for i in range(self._limit_images_per_query or 3)
                    ]
                    saved = []
                    for u in urls:
                        fn = hashlib.sha1(u.encode()).hexdigest() + ".jpg"
                        fp = out_dir / fn
                        if not fp.exists():
                            with open(fp, "wb") as f:
                                f.write(b"\xff\xd8\xff\xd9")
                        saved.append(str(fp))
                else:
                    seen_key = f"{topic}::{sub}"
                    async with self._seen_urls_lock:
                        topic_seen = self._seen_urls[seen_key]

                    urls, saved = await async_fetch_and_download(
                        query,
                        str(out_dir),
                        max_images=min(
                            int(
                                self.cfg.get("bing", {}).get("per_subtopic_images", 500)
                            ),
                            self._limit_images_per_query or 10**9,
                        ),
                        download_concurrency_initial=48,
                        download_concurrency_max=64,
                        per_host_limit=6,
                        head_precheck=False,
                        debug=False,
                        already_seen=topic_seen,
                    )

                elapsed = time.time() - start

            except Exception as e:
                print(f"[QueryWorker {idx}] error query='{query}': {e}")
                urls, saved, elapsed = [], [], 0.0

            self.loop_totals["urls"] += len(urls)
            self.loop_totals["downloaded"] += len(saved)

            for fp in saved:
                await self.downloaded_queue.put(
                    {
                        "topic": topic,
                        "subtopic": sub,
                        "query": query,
                        "path": pathlib.Path(fp),
                    }
                )

            print(
                f"[QueryWorker {idx}] '{query}' "
                f"urls={len(urls)} downloaded={len(saved)} t={elapsed:.2f}s"
            )

            if not self.dry_run:
                await asyncio.sleep(random.uniform(2.0, 5.0))

            self.queries_queue.task_done()

    async def _download_dispatcher(self):
        """将下载完成的条目转发到 OCR 队列."""
        print("[Pipeline-Flow] DownloadDispatcher started.")
        while True:
            item = await self.downloaded_queue.get()
            if item is self._sentinel:
                self.downloaded_queue.task_done()
                break
            await self.ocr_queue.put(item)
            self.downloaded_queue.task_done()

    async def _ocr_result_collector(self):
        """汇聚多进程 OCR 结果并路由至质量队列."""
        loop = asyncio.get_running_loop()
        collected = 0
        orphans = 0
        errors = 0

        import threading

        def _background_puller():
            pulled_count = 0
            thread_errors = 0
            while not self._collector_stop_flag.is_set():
                try:
                    r = self.ocr_multiprocess_pool.result_q.get(timeout=3.0)
                    loop.call_soon_threadsafe(self._result_buffer_queue.put_nowait, r)
                    pulled_count += 1
                    if pulled_count % 1000 == 0:
                        print(
                            f"[OCRPullerThread] Pulled {pulled_count} results, "
                            f"buffer={self._result_buffer_queue.qsize()}"
                        )
                except Exception as e:
                    if "Empty" not in str(type(e).__name__):
                        thread_errors += 1
                        if thread_errors % 100 == 0:
                            print(f"[OCRPullerThreadError] {e} (total={thread_errors})")
                    continue
            print(
                f"[OCRPullerThread] Stopped. pulled={pulled_count} errors={thread_errors}"
            )

        puller_thread = threading.Thread(
            target=_background_puller, daemon=True, name="OCRResultPuller"
        )
        puller_thread.start()
        self._collector_puller_thread = puller_thread

        print("[OCRResultCollector] Started with background puller thread")

        batch_size = 50
        idle_timeout = 0.1

        while True:
            try:
                batch = []
                for _ in range(batch_size):
                    try:
                        r = self._result_buffer_queue.get_nowait()
                        batch.append(r)
                    except asyncio.QueueEmpty:
                        break

                if not batch:
                    if self._collector_stop_event.is_set():
                        print("[OCRResultCollector] Stop event received")
                        break

                    if self._ocr_workers_done and not self._ocr_pending:
                        print(
                            "[OCRResultCollector] All workers done and no pending tasks"
                        )
                        break

                    if self._ocr_workers_done and self._ocr_pending:
                        max_wait = 30
                        wait_start = time.time()
                        print(
                            f"[OCRResultCollector] Waiting for {len(self._ocr_pending)} "
                            f"pending tasks to arrive..."
                        )
                        while (
                            self._ocr_pending and (time.time() - wait_start) < max_wait
                        ):
                            temp_batch = []
                            for _ in range(100):
                                try:
                                    r = self._result_buffer_queue.get_nowait()
                                    temp_batch.append(r)
                                except asyncio.QueueEmpty:
                                    break
                            if temp_batch:
                                for r in temp_batch:
                                    jid = r.get("job_id")
                                    if not jid:
                                        errors += 1
                                        continue
                                    meta = self._ocr_pending.pop(jid, None)
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
                                    await self.quality_queue.put(meta)
                                    collected += 1
                                print(
                                    f"[OCRResultCollector] Drained {len(temp_batch)}, "
                                    f"remaining={len(self._ocr_pending)}"
                                )
                                continue
                            if self._ocr_pending:
                                remaining_time = max_wait - (time.time() - wait_start)
                                if remaining_time > 0:
                                    print(
                                        f"[OCRResultCollector] Still waiting for "
                                        f"{len(self._ocr_pending)} tasks "
                                        f"(timeout in {remaining_time:.1f}s)"
                                    )
                                    await asyncio.sleep(0.5)
                                else:
                                    break
                        if self._ocr_pending:
                            print(
                                f"[OCRResultCollector] WARNING: {len(self._ocr_pending)} "
                                f"tasks still unmatched after {max_wait}s"
                            )
                            print("[OCRResultCollector] Final drain from result_q...")
                            final_drained = 0
                            for _ in range(500):
                                try:
                                    r = self.ocr_multiprocess_pool.result_q.get_nowait()
                                    jid = r.get("job_id")
                                    if jid in self._ocr_pending:
                                        meta = self._ocr_pending.pop(jid)
                                        meta.update(
                                            {
                                                "ocr": r.get("ocr", {}),
                                                "ocr_time": r.get("ocr_time"),
                                                "ocr_gpu_id": r.get("ocr_gpu_id"),
                                                "success": r.get("success", False),
                                            }
                                        )
                                        await self.quality_queue.put(meta)
                                        final_drained += 1
                                except:
                                    break
                            print(
                                f"[OCRResultCollector] Final drained {final_drained}, "
                                f"still_orphaned={len(self._ocr_pending)}"
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
                    meta = self._ocr_pending.pop(jid, None)
                    if meta is None:
                        orphans += 1
                        if orphans % 100 == 0:
                            print(
                                f"[OCRResultCollector] orphan job_id={jid[:8]}, "
                                f"total_orphans={orphans}"
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
                    await self.quality_queue.put(meta)

                collected += valid_count
                if collected % 100 == 0:
                    print(
                        f"[OCRResultCollector] Collected {collected}, "
                        f"batch_size={len(batch)}, valid={valid_count}, "
                        f"pending={len(self._ocr_pending)}, "
                        f"buffer={self._result_buffer_queue.qsize()}"
                    )

            except Exception as e:
                print(f"[OCRResultCollectorError] {e}")
                import traceback

                traceback.print_exc()
                await asyncio.sleep(0.1)

        print("[OCRResultCollector] Stopping background puller thread...")
        self._collector_stop_flag.set()
        if puller_thread.is_alive():
            puller_thread.join(timeout=5.0)
            if puller_thread.is_alive():
                print(
                    "[OCRResultCollector] WARNING: Puller thread did not exit cleanly"
                )
            else:
                print("[OCRResultCollector] Puller thread exited")

        print(
            f"[OCRResultCollector] Stopped. "
            f"collected={collected} orphans={orphans} errors={errors}"
        )

    async def _ocr_worker(self, idx: int):
        """提交或执行 OCR 任务并将结果送入质量队列（pHash去重已移至质量检查之后）."""
        loop = asyncio.get_running_loop()
        submitted = 0
        submit_errors = 0

        while True:
            meta = await self.ocr_queue.get()
            if meta is self._sentinel:
                self.ocr_queue.task_done()
                print(
                    f"[OCR-Worker-{idx}] Exit. submitted={submitted} submit_errors={submit_errors}"
                )
                break

            fp: pathlib.Path = meta["path"]

            if self.ocr_multiprocess_pool:
                throttle_wait = 0
                while len(self._ocr_pending) >= self._max_ocr_pending:
                    if self._ocr_workers_done:
                        print(
                            f"[OCR-Worker-{idx}] Workers marked done, "
                            f"breaking throttle (pending={len(self._ocr_pending)})"
                        )
                        break
                    throttle_wait += 0.2
                    if idx == 0 and throttle_wait > 5:
                        print(
                            f"[OCRThrottle] pending={len(self._ocr_pending)} "
                            f">= {self._max_ocr_pending}, waiting..."
                        )
                        throttle_wait = 0
                    await asyncio.sleep(0.2)
                job_id = uuid.uuid4().hex
                meta["job_id"] = job_id

                if not meta.get("path"):
                    print(f"[OCR-Worker-{idx}] ERROR: meta missing 'path'")
                    self._record_sample_outcome(meta, ["ocr_meta_error"])
                    self.loop_totals["rejected"] += 1
                    self.loop_totals["ocr_submit_error"] += 1
                    submit_errors += 1
                    self.ocr_queue.task_done()
                    continue

                try:
                    self._ocr_pending[job_id] = meta
                    self.ocr_multiprocess_pool.submit_task_round_robin(
                        meta, self._rr_seq
                    )
                    self._rr_seq += 1
                    submitted += 1
                    if submitted % 200 == 0:
                        print(
                            f"[OCR-Submit-{idx}] submitted={submitted} pending={len(self._ocr_pending)} Q_ocr={self.ocr_queue.qsize()}"
                        )
                except Exception as e:
                    submit_errors += 1
                    self._ocr_pending.pop(job_id, None)
                    print(f"[OCR-SubmitError-{idx}] {fp.name}: {type(e).__name__} {e}")
                    self._record_sample_outcome(meta, ["ocr_failed"])
                    self.loop_totals["rejected"] += 1
                    self.loop_totals["ocr_submit_error"] += 1
                finally:
                    self.ocr_queue.task_done()
                continue

            ocr_instance = None
            rec = None
            try:
                async with self._ocr_sem:
                    ocr_instance = await loop.run_in_executor(
                        None, self.ocr_queue_pool.get, True, 10.0
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
                                        f"[OCRMemWarn] GPU:{gpu_id} free={free_mb:.0f}MB"
                                    )
                                    torch.cuda.empty_cache()
                        except Exception as mem_e:
                            print(f"[OCRMemCheckError] {mem_e}")
                    for attempt in range(3):
                        try:
                            rec = await asyncio.wait_for(
                                loop.run_in_executor(None, ocr_instance.run, str(fp)),
                                timeout=30.0,
                            )
                            break
                        except asyncio.TimeoutError:
                            print(f"[OCRTimeout] {fp.name} attempt={attempt+1}")
                            if attempt == 2:
                                raise
                            await asyncio.sleep(1.0)
                        except Exception as oe:
                            print(f"[OCRError] {fp.name} attempt={attempt+1} {oe}")
                            if attempt == 2:
                                raise
                            await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[OCRFatal] {fp.name} {e}")
                self._record_sample_outcome(meta, ["ocr_error"])
                self.loop_totals["rejected"] += 1
                self.ocr_queue.task_done()
                if ocr_instance:
                    with contextlib.suppress(Exception):
                        self.ocr_queue_pool.put(ocr_instance, block=False)
                continue
            finally:
                if ocr_instance:
                    with contextlib.suppress(Exception):
                        self.ocr_queue_pool.put(ocr_instance, block=False)

            meta["ocr"] = rec
            await self.quality_queue.put(meta)
            self.ocr_queue.task_done()

    async def _quality_worker(self, idx: int):
        """对图片进行质量评分与筛选，通过后执行 pHash 去重，再送入语义队列."""
        loop = asyncio.get_running_loop()
        while True:
            meta = await self.quality_queue.get()
            if meta is self._sentinel:
                self.quality_queue.task_done()
                break
            fp: pathlib.Path = meta["path"]

            try:
                qscore = await loop.run_in_executor(
                    None, self.qa.score, str(fp), meta.get("ocr", {})
                )
                meta["quality"] = qscore
                if hasattr(self.qa, "check"):
                    ok, reasons = self.qa.check(qscore, meta.get("ocr", {}))
                else:
                    ok = self.qa.accept(qscore, meta.get("ocr", {}))
                    reasons = [] if ok else ["legacy_reject"]
                if not ok:
                    self._record_sample_outcome(meta, reasons)
                    self.loop_totals["rejected"] += 1
                    self.loop_totals["quality_rejected"] += 1

                    if "has_watermark" in reasons:
                        self.loop_totals["watermark_rejected"] += 1
                        wm_keys = meta.get("quality", {}).get("watermark_keywords", [])
                        wm_methods = meta.get("quality", {}).get(
                            "watermark_methods", []
                        )
                        # print(
                        #     f"[WatermarkReject] {fp.name} "
                        #     f"methods={wm_methods} keywords={wm_keys[:3]}"
                        # )
                    if not self.keep_rejects:
                        fp.unlink(missing_ok=True)
                    self.quality_queue.task_done()
                    continue
            except Exception as e:
                print(f"[QualityError] {fp.name} {e}")
                self._record_sample_outcome(meta, ["quality_error"])
                self.loop_totals["rejected"] += 1
                self.quality_queue.task_done()
                continue

            key = meta["topic"] if self._dedup_scope == "topic" else ""
            try:
                async with self.dedup_lock:
                    is_dup, _ = self.dedup.check_and_add(str(fp), key)
                if is_dup:
                    self._record_sample_outcome(meta, ["dedup_duplicate"])
                    self.loop_totals["rejected"] += 1
                    self.loop_totals["pHash_rejected"] += 1
                    if not self.keep_rejects:
                        fp.unlink(missing_ok=True)
                    self.quality_queue.task_done()
                    continue
            except Exception as e:
                self.loop_totals["dedup_error"] += 1
                self.loop_totals["rejected"] += 1
                self._record_sample_outcome(meta, ["dedup_error"])
                if not self.keep_rejects:
                    fp.unlink(missing_ok=True)
                try:
                    self.dedup.discard_path(str(fp), key)
                except Exception:
                    pass
                self.quality_queue.task_done()
                print(f"[DedupError-quality] {fp.name} {e}")
                continue
            await self.semantic_queue.put(meta)
            self.quality_queue.task_done()

    async def _semantic_worker(self, idx: int):
        """执行语义与文本一致性检查并批量缓冲接受项."""
        while True:
            meta = await self.semantic_queue.get()
            if meta is self._sentinel:
                self.semantic_queue.task_done()
                break
            fp: pathlib.Path = meta["path"]
            dedup_key = meta["topic"] if self._dedup_scope == "topic" else ""
            sem_res = {}
            try:
                sem_ok, sem_res = self.semantic.check_relevance(
                    str(fp), topic=meta["topic"], subtopic=meta["subtopic"]
                )
                meta["semantic"] = sem_res
                if not sem_ok:
                    reasons = sem_res.get("rejection_reasons", ["semantic_error"])
                    self._record_sample_outcome(meta, reasons)
                    self.loop_totals["rejected"] += 1
                    self.loop_totals["semantic_rejected"] += 1
                    if not self.keep_rejects:
                        fp.unlink(missing_ok=True)
                    try:
                        self.dedup.discard_path(str(fp), dedup_key)
                    except Exception:
                        pass
                    self.semantic_queue.task_done()
                    continue
            except Exception as e:
                print(f"[SemanticError] {fp.name} {e}")
                self._record_sample_outcome(meta, ["semantic_error"])
                self.loop_totals["rejected"] += 1
                try:
                    self.dedup.discard_path(str(fp), dedup_key)
                except Exception:
                    pass
                self.semantic_queue.task_done()
                continue
            if not self.text_checker.is_available():
                meta["text_consistency"] = None
            else:
                try:
                    tc = self.text_checker.check(
                        meta.get("ocr", {}), meta["topic"], meta["subtopic"]
                    )
                    meta["text_consistency"] = tc
                    if not tc.get("passed", True):
                        reasons = tc.get("rejection_reasons") or ["text_error"]
                        self._record_sample_outcome(meta, reasons)
                        self.loop_totals["rejected"] += 1
                        self.loop_totals["text_consistency_rejected"] += 1
                        if not self.keep_rejects:
                            fp.unlink(missing_ok=True)
                        try:
                            self.dedup.discard_path(str(fp), dedup_key)
                        except Exception:
                            pass
                        self.semantic_queue.task_done()
                        continue
                except Exception as e:
                    print(f"[TextConsistencyError] {fp.name} {e}")
                    self._record_sample_outcome(meta, ["text_error"])
                    self.loop_totals["rejected"] += 1
                    try:
                        self.dedup.discard_path(str(fp), dedup_key)
                    except Exception:
                        pass
                    self.semantic_queue.task_done()
                    continue
            caption = ""
            try:
                loop = asyncio.get_event_loop()
                caption = await loop.run_in_executor(
                    None,
                    generate_image_caption,
                    str(fp),
                    self.cfg,
                    meta["topic"],
                    meta["subtopic"],
                    meta.get("ocr"),
                    meta.get("quality"),
                )
                if not caption:
                    caption = f'{meta["topic"]} - {meta["subtopic"]}'
            except Exception as e:
                print(f"[CaptionError] {fp.name} {e}")
                caption = f'{meta["topic"]} - {meta["subtopic"]}'
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
                "shard": self.shard_id,
            }
            self._record_sample_outcome(meta, [])
            self.loop_totals["accepted"] += 1
            self._batch_buffer.append(item)

            await self._flush_batch(force=True)
            self.semantic_queue.task_done()

    async def _flush_batch(self, force: bool = False):
        """写出当前缓冲的已接受记录到数据集与磁盘."""
        if not self._batch_buffer:
            return
        if (not force) and len(self._batch_buffer) < self.batch_size:
            return
        batch = list(self._batch_buffer)
        self._batch_buffer.clear()

        def _do_write(items: List[Dict[str, Any]]):
            for it in items:
                img_path = it.get("image_path", "unknown")
                try:
                    self.writer.write_record(**it)
                    # print(f"[DoWrite] write_record OK: {pathlib.Path(img_path).name}")
                except Exception as e:
                    # print(
                    #     f"[DoWrite][ERROR] write_record failed for {img_path}: {type(e).__name__}: {e}"
                    # )
                    continue

                try:
                    if it.get("is_generated"):
                        self._save_generated(it, pathlib.Path(img_path))
                    else:
                        result = self._save_accepted(it)
                        if result is None:
                            # print(
                            #     f"[DoWrite][WARN] _save_accepted returned None for {img_path}"
                            # )
                            pass
                        else:
                            # print(f"[DoWrite] _save_accepted OK → {result}")
                            pass
                except Exception as e:
                    import traceback

                    # print(
                    #     f"[DoWrite][ERROR] _save_accepted/generated raised for {img_path}: "
                    #     f"{type(e).__name__}: {e}"
                    # )
                    traceback.print_exc()

        await asyncio.to_thread(_do_write, batch)
        if self.metrics_cfg.get("print_every_batch", True):
            print(
                f"[BatchFlush] size={len(batch)} totalAccepted={self.loop_totals['accepted']} "
                f"totalRejected={self.loop_totals['rejected']}"
            )

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
            print(f"[Pipeline] Cleared CUDA caches on {device_count} device(s)")
        except Exception as e:
            print(f"[Pipeline][Warn] Failed to clear CUDA memory: {e}")

    def _save_accepted(self, item: Dict[str, Any]) -> Optional[str]:
        img_src = pathlib.Path(item["image_path"])

        # print(f"[SaveAccepted] enter: src={img_src.name} exists={img_src.exists()}")

        if not img_src.exists():
            # print(f"[SaveAccepted][WARN] source not found: {img_src}")
            return None

        ext = img_src.suffix.lower() or ".jpg"
        name = self._alloc_filename(ext)
        if name is None:
            name = self._content_hash_filename(img_src)
        name = self._unique_final(name)
        if not name:
            # print(f"[SaveAccepted][ERROR] could not alloc filename for {img_src.name}")
            return None

        img_dst = self.accepted_images_dir / name
        cap_dst = self.accepted_captions_dir / f"{name.rsplit('.', 1)[0]}.txt"
        json_dst = self.accepted_json_dir / f"{name.rsplit('.', 1)[0]}.json"
        # print(f"[SaveAccepted] allocated name={name} dst={img_dst}")

        try:
            shutil.copy2(img_src, img_dst)
            # print(f"[SaveAccepted] copy OK: {img_src.name} → {img_dst}")
        except Exception as e:
            # print(f"[SaveAccepted][ERROR] copy failed: {img_src} → {img_dst}: {e}")
            return None

        try:
            dedup_topic = "" if self._dedup_scope == "global" else item.get("topic", "")
            source_abs = str(img_src.resolve())
            # print(
            #     f"[SaveAccepted] calling commit_final_path: "
            #     f"final={img_dst} source={source_abs} topic={dedup_topic!r} "
            #     f"scope={self._dedup_scope}"
            # )

            ok, tag = self.dedup.commit_final_path(
                final_path=str(img_dst),
                topic=dedup_topic,
                source_path=source_abs,
            )

            try:
                index_size = self.dedup._total_indexed()
            except Exception:
                index_size = -1

            # print(
            #     f"[SaveAccepted][DedupCommit] file={name} ok={ok} tag={tag} "
            #     f"index_total={index_size} index_file={self.dedup.index_file}"
            # )

            if not ok:
                # print(
                #     f"[SaveAccepted][ERROR] commit_final_path returned ok=False, "
                #     f"tag={tag}. Rolling back {img_dst}"
                # )
                img_dst.unlink(missing_ok=True)
                return None

        except Exception as e:
            import traceback

            # print(
            #     f"[SaveAccepted][ERROR] commit_final_path raised: {type(e).__name__}: {e}"
            # )
            traceback.print_exc()
            img_dst.unlink(missing_ok=True)
            return None

        cap = self._sanitize_caption(item.get("caption"))
        try:
            cap_dst.write_text(cap, encoding="utf-8")
        except Exception as e:
            # print(f"[SaveAccepted][WARN] caption write failed: {e}")
            pass

        try:
            meta = {
                "filename": name,
                "topic": item.get("topic"),
                "subtopic": item.get("subtopic"),
                "query": item.get("query"),
                "caption": cap,
                "ocr": item.get("ocr"),
                "quality": item.get("quality"),
                "semantic": item.get("semantic"),
                "text_consistency": item.get("text_consistency"),
                "content_sha1": item.get("content_sha1"),
                "source": item.get("source"),
                "is_generated": item.get("is_generated", False),
                "timestamp": item.get("ts"),
                "shard": item.get("shard"),
            }
            with open(json_dst, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            # print(f"[SaveAccepted] metadata JSON written: {json_dst.name}")
        except Exception as e:
            # print(f"[SaveAccepted][WARN] meta JSON write failed: {e}")
            pass

        t = item.get("topic", "NA")
        s = sanitize_subtopic_name(item.get("subtopic", "NA"))
        with self._accepted_counts_lock:
            self.accepted_counts[(t, s)] += 1
        self._write_image_index_entry(name, t, s, item.get("query", ""))

        # print(f"[SaveAccepted] done: {name} topic={t} subtopic={s}")
        return str(img_dst)

    def _sanitize_caption(self, cap: Optional[str]) -> str:
        """清洗 caption 文本为单行紧凑格式."""
        if cap is None:
            return ""
        cap = str(cap).strip()
        return " ".join(cap.split())

    def _init_filename_counter(self):
        """扫描已存在文件以初始化顺序命名计数器."""
        images_dir = pathlib.Path(self.accepted_images_dir)
        self._filename_counter = sum(1 for f in images_dir.iterdir() if f.is_file())
        print(f"[Pipeline] Filename counter initialized: {self._filename_counter}")

    def _next_filename_sync(self, ext: str) -> str:
        """分配下一个顺序文件名（线程安全）。"""
        with self._filename_lock_sync:
            self._filename_counter += 1
            return f"img_{self._filename_counter:07d}{ext}"

    def _alloc_filename(self, ext: str) -> Optional[str]:
        """根据策略分配文件名；hash 策略返回 None 以备用内容哈希."""
        st = self._filename_strategy
        if st == "uuid":
            return f"img_{uuid.uuid4().hex}{ext}"
        if st == "hash":
            return None
        return self._next_filename_sync(ext)

    def _content_hash_filename(self, fp: pathlib.Path) -> str:
        """基于文件内容哈希生成唯一文件名."""
        try:
            h = sha1_of_file(fp)
        except Exception:
            h = uuid.uuid4().hex
        return f"img_{h}{fp.suffix.lower() or '.jpg'}"

    def _unique_final(self, name: str) -> str:
        """若命名冲突则追加随机后缀保证唯一性."""
        if not name:
            return f"img_{uuid.uuid4().hex[:8]}.jpg"
        if not (self.accepted_images_dir / name).exists():
            return name
        base, ext = name.rsplit(".", 1) if "." in name else (name, "jpg")
        return f"{base}_{uuid.uuid4().hex[:6]}.{ext}"

    def _write_image_index_entry(
        self, filename: str, topic: str, subtopic: str, query: str
    ) -> bool:
        """向索引文件追加一条图像记录."""
        entry = {
            "filename": filename,
            "topic": topic,
            "subtopic": subtopic,
            "query": query,
            "timestamp": time.time(),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            with open(self._image_index_path, "a", encoding="utf-8") as f:
                f.write(line)
            return True
        except Exception as e:
            with contextlib.suppress(Exception):
                with open(self._image_index_error_path, "a", encoding="utf-8") as ferr:
                    ferr.write(f"WriteFail: {e} | line={line[:200]}\n")
            print(f"[ImageIndexError] {e}")
            return False

    async def _plan_prompts_and_generate(self):
        """执行统计需求、生成提示、生成图片并后处理."""
        log_dir = pathlib.Path(self.cfg.get("paths", {}).get("log_dir", "."))
        prompt_root = log_dir / "prompt_plans"
        prompt_root.mkdir(parents=True, exist_ok=True)

        need_map: Dict[Tuple[str, str], int] = {}
        try:
            from src.tools.count_analyse import run_count_analyse

            ca_res = run_count_analyse(
                config_path=self.config_path,
                agents=self.agents,
                use_llm=("CountAnalyse" in self.agents),
                enforce_caps=False,
                fallback_strategy="uniform",
                show_stats=False,
                show_lines=False,
            )
            need_map = ca_res.get("final_need_map", {})
            print(f"[CountAnalyse] need_items={len(need_map)} gap={ca_res.get('gap')}")
        except Exception as e:
            print(f"[CountAnalyseError] {e}")

        if "PromptPlanner" in self.agents and need_map:
            for (tp, sub), need in need_map.items():
                if need <= 0:
                    continue
                pp_prompt = (
                    "Based on the following information, generate high-quality English image generation prompts (only return a JSON array, list of strings):\n"
                    f"Theme: {tp}\nSubtheme: {sub}\n"
                    f"Number of prompts needed: {int(need)}\n"
                    "Generation Requirements:\n"
                    "1) The prompt must focus on the theme, listing key textual elements that must appear in the image.\n"
                    "2) Text must be clear, high contrast, watermark-free.\n"
                    "3) Layout natural & neat; avoid clutter.\n"
                    "4) For complex scenes request simpler layout if needed.\n"
                    "5) Output only the JSON array of prompt strings."
                )
                try:
                    prompts = self.agents["PromptPlanner"].run_list(pp_prompt) or []
                    print(f"[PromptPlanner] {tp}/{sub} -> {len(prompts)} prompts")
                    dst_dir = prompt_root / tp
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    safe_sub = sub.replace("/", "_")
                    out_fp = dst_dir / f"{safe_sub}.json"
                    with open(out_fp, "w", encoding="utf-8") as f:
                        json.dump(
                            [
                                {"topic": tp, "subtopic": sub, "prompt": p}
                                for p in prompts
                                if isinstance(p, str) and p.strip()
                            ],
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f"[PromptPlanner] saved {out_fp}")
                except Exception as e:
                    print(f"[PromptPlannerError] {tp}/{sub} {e}")

        prompt_files = list(prompt_root.glob("**/*.json"))
        all_generated: List[Dict[str, Any]] = []
        for pf in prompt_files:
            try:
                recs = json.load(open(pf, "r", encoding="utf-8"))
            except Exception:
                continue
            for rec in recs:
                ptxt = rec.get("prompt")
                if not ptxt:
                    continue
                topic = rec.get("topic")
                sub = rec.get("subtopic")
                try:
                    res = generate_qwen_images(
                        topic=topic,
                        subtopic=sub,
                        prompts=[ptxt],
                        config_path=self.config_path,
                    )
                    gen_results = res.get("results") if res else []
                    if gen_results:
                        for img_path, prompt in gen_results:
                            all_generated.append(
                                {
                                    "image_path": img_path,
                                    "topic": topic,
                                    "subtopic": sub,
                                    "prompt": prompt,
                                }
                            )
                    self.loop_totals["generated"] += len(gen_results or [])
                    print(f"[Generate] {topic}/{sub} +{len(gen_results or [])} images")
                except Exception as e:
                    print(f"[GenerateError] {topic}/{sub} {e}")
        print(f"[GenPhase] total_generated={len(all_generated)}")
        await self._post_process_generated(all_generated)

    async def _post_process_generated(self, images: List[Dict[str, Any]]):
        """对生成图片做质量/语义/文本一致性筛选并保存通过者."""
        loop = asyncio.get_running_loop()
        accepted = 0
        for info in images:
            img_path = pathlib.Path(info.get("image_path"))
            if not img_path.exists():
                continue

            try:
                if hasattr(self.qa, "check_image_quality"):
                    quality_fail_reasons = self.qa.check_image_quality(str(img_path))
                    if quality_fail_reasons:
                        self.rej_tracker.add(
                            str(img_path),
                            (
                                quality_fail_reasons
                                if isinstance(quality_fail_reasons, list)
                                else ["quality_reject"]
                            ),
                        )
                        self.loop_totals["generated_rejected"] += 1
                        continue
                else:
                    qscore = await loop.run_in_executor(
                        None, self.qa.score, str(img_path), {}
                    )
                    if hasattr(self.qa, "check"):
                        ok, reasons = self.qa.check(qscore, {})
                    else:
                        ok = self.qa.accept(qscore, {})
                        reasons = [] if ok else ["legacy_reject"]
                    if not ok:
                        self.rej_tracker.add(str(img_path), reasons)
                        self.loop_totals["generated_rejected"] += 1
                        continue
            except Exception as e:
                print(f"[GenQualityError] {img_path.name} {e}")
                self.rej_tracker.add(str(img_path), ["quality_error"])
                self.loop_totals["generated_rejected"] += 1
                continue

            try:
                sem_ok, sem_res = self.semantic.check_relevance(
                    str(img_path),
                    topic=info.get("topic"),
                    subtopic=info.get("subtopic"),
                )
                if not sem_ok:
                    self.rej_tracker.add(
                        str(img_path),
                        sem_res.get("rejection_reasons", ["semantic_error"]),
                    )
                    self.loop_totals["generated_rejected"] += 1
                    continue
            except Exception as e:
                print(f"[GenSemanticError] {img_path.name} {e}")
                self.rej_tracker.add(str(img_path), ["semantic_error"])
                self.loop_totals["generated_rejected"] += 1
                continue

            if self.text_checker.is_available():
                try:
                    rec = await loop.run_in_executor(None, self.ocr.run, str(img_path))
                    tr = self.text_checker.check(
                        rec, info.get("topic"), info.get("subtopic")
                    )
                    if not tr.get("passed", True):
                        self.rej_tracker.add(
                            str(img_path), tr.get("rejection_reasons", ["text_error"])
                        )
                        self.loop_totals["generated_rejected"] += 1
                        continue
                except Exception as e:
                    print(f"[GenTextConsistencyError] {img_path.name} {e}")
                    self.rej_tracker.add(str(img_path), ["text_error"])
                    self.loop_totals["generated_rejected"] += 1
                    continue

            sha1 = sha1_of_file(img_path)
            meta = {
                "image_path": str(img_path),
                "topic": info.get("topic"),
                "subtopic": info.get("subtopic"),
                "prompt": info.get("prompt"),
                "is_generated": True,
                "content_sha1": sha1,
                "ts": time.time(),
                "source": "qwen_gen",
                "shard": self.shard_id,
            }
            self.writer.write_record(**meta)
            self._save_generated(meta, img_path)
            self.rej_tracker.add(str(img_path), [])
            accepted += 1
        self.loop_totals["generated_accepted"] += accepted
        print(
            f"[GenPostProcess] accepted={accepted} rejected={self.loop_totals['generated_rejected']}"
        )

    def _save_generated(self, item: Dict[str, Any], img_path: pathlib.Path):
        """保存通过的生成图片及元数据到生成接受目录."""
        topic = (item.get("topic") or "NA").replace("/", "_")
        sub = (item.get("subtopic") or "NA").replace("/", "_")
        dst_dir = ensure_dir(self.gen_accepted_root / topic / sub)
        sha1 = item.get("content_sha1") or "na"
        ext = img_path.suffix.lower() or ".jpg"
        img_dst = dst_dir / f"{sha1}{ext}"
        json_dst = dst_dir / f"{sha1}.json"
        if not img_dst.exists():
            try:
                import shutil

                shutil.copy2(str(img_path), str(img_dst))
            except Exception as e:
                print(f"[GenCopyWarn] {e}")
        with open(json_dst, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)


__all__ = ["OrchestratorPipeline"]
