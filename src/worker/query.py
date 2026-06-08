"""Fetches images from search engines and pushes results to the download queue."""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.tools.bing_noapi import async_fetch_and_download
from src.utils.common import ensure_dir

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def query_worker(pipeline: OrchestratorPipeline, idx: int) -> None:
    """Fetch images for queued queries and push downloaded file paths to the download queue."""
    print(f"[Pipeline-Flow] QueryWorker-{idx } started.")

    while True:
        priority_tuple = await pipeline.queries_queue.get()
        if len(priority_tuple) == 3:
            _, _, item = priority_tuple
        else:
            _, item = priority_tuple

        if item is pipeline._sentinel:
            pipeline.queries_queue.task_done()
            break

        topic, sub, query = item
        print(f"[Pipeline-Flow] QueryWorker-{idx } got query: '{query }'")

        out_dir = ensure_dir(
            pipeline.images_root / pipeline.shard_id / topic / sub.replace("/", "_")
        )

        try:
            if pipeline._query_bucket:
                await pipeline._query_bucket.acquire()

            start = time.time()

            if pipeline.dry_run:
                urls = [
                    f"dry://{query }/{i }"
                    for i in range(pipeline._limit_images_per_query or 3)
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
                seen_key = f"{topic }::{sub }"
                async with pipeline._seen_urls_lock:
                    topic_seen = pipeline._seen_urls[seen_key]

                urls, saved = await async_fetch_and_download(
                    query,
                    str(out_dir),
                    max_images=min(
                        int(
                            pipeline.cfg.get("bing", {}).get("per_subtopic_images", 500)
                        ),
                        pipeline._limit_images_per_query or 10**9,
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
            print(f"[QueryWorker {idx }] error query='{query }': {e }")
            urls, saved, elapsed = [], [], 0.0

        pipeline.loop_totals["urls"] += len(urls)
        pipeline.loop_totals["downloaded"] += len(saved)

        for fp in saved:
            await pipeline.downloaded_queue.put(
                {
                    "topic": topic,
                    "subtopic": sub,
                    "query": query,
                    "path": Path(fp),
                }
            )

        print(
            f"[QueryWorker {idx }] '{query }' "
            f"urls={len (urls )} downloaded={len (saved )} t={elapsed :.2f}s"
        )

        if not pipeline.dry_run:
            await asyncio.sleep(random.uniform(2.0, 5.0))

        pipeline.queries_queue.task_done()
