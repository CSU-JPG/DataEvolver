"""Bridges the download queue to the OCR queue."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def download_dispatcher(pipeline: OrchestratorPipeline) -> None:
    """Forward downloaded items from the download queue to the OCR queue."""
    print("[Pipeline-Flow] DownloadDispatcher started.")
    while True:
        item = await pipeline.downloaded_queue.get()
        if item is pipeline._sentinel:
            pipeline.downloaded_queue.task_done()
            break
        await pipeline.ocr_queue.put(item)
        pipeline.downloaded_queue.task_done()
