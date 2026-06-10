"""Flushes accepted records to disk, saves items, and manages filenames."""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import shutil
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.utils.common import sha1_of_file, sanitize_subtopic_name

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def flush_batch(pipeline: OrchestratorPipeline, force: bool = False) -> None:
    """Write the current buffer of accepted records to dataset and disk."""
    if not pipeline._batch_buffer:
        return
    if (not force) and len(pipeline._batch_buffer) < pipeline.batch_size:
        return
    batch = list(pipeline._batch_buffer)
    pipeline._batch_buffer.clear()

    def _do_write(items: List[Dict[str, Any]]) -> None:
        for it in items:
            img_path = it.get("image_path", "unknown")
            try:
                pipeline.writer.write_record(**it)
                # print(f"[DoWrite] write_record OK: {pathlib .Path (img_path ).name }")
            except Exception as e:
                # print(
                #     f"[DoWrite][ERROR] write_record failed for {img_path }: "
                #     f"{type (e ).__name__ }: {e }"
                # )
                continue

            try:
                if it.get("is_generated"):
                    save_generated(pipeline, it, pathlib.Path(img_path))
                else:
                    result = save_accepted(pipeline, it)
                    if result is None:
                        # print(
                        #     f"[DoWrite][WARN] _save_accepted returned None for {img_path }"
                        # )
                        pass
                    else:
                        # print(f"[DoWrite] _save_accepted OK -> {result }")
                        pass
            except Exception as e:
                import traceback

                # print(
                #     f"[DoWrite][ERROR] _save_accepted/generated raised for {img_path }: "
                #     f"{type (e ).__name__ }: {e }"
                # )
                traceback.print_exc()

    await asyncio.to_thread(_do_write, batch)
    if pipeline.metrics_cfg.get("print_every_batch", True):
        print(
            f"[BatchFlush] size={len (batch )} "
            f"totalAccepted={pipeline .loop_totals ['accepted']} "
            f"totalRejected={pipeline .loop_totals ['rejected']}"
        )


def save_accepted(
    pipeline: OrchestratorPipeline, item: Dict[str, Any]
) -> Optional[str]:
    """Save an accepted image and its metadata to the accepted directory."""
    img_src = pathlib.Path(item["image_path"])

    # print(f"[SaveAccepted] enter: src={img_src .name } exists={img_src .exists ()}")

    if not img_src.exists():
        # print(f"[SaveAccepted][WARN] source not found: {img_src }")
        return None

    ext = img_src.suffix.lower() or ".jpg"
    name = alloc_filename(pipeline, ext)
    if name is None:
        name = content_hash_filename(img_src)
    name = unique_final(pipeline, name)
    if not name:
        # print(f"[SaveAccepted][ERROR] could not alloc filename for {img_src .name }")
        return None

    img_dst = pipeline.accepted_images_dir / name
    cap_dst = pipeline.accepted_captions_dir / f"{name .rsplit ('.',1 )[0 ]}.txt"
    json_dst = pipeline.accepted_json_dir / f"{name .rsplit ('.',1 )[0 ]}.json"
    # print(f"[SaveAccepted] allocated name={name } dst={img_dst }")

    try:
        shutil.copy2(img_src, img_dst)
        # print(f"[SaveAccepted] copy OK: {img_src .name } -> {img_dst }")
    except Exception as e:
        # print(f"[SaveAccepted][ERROR] copy failed: {img_src } -> {img_dst }: {e }")
        return None

    try:
        dedup_topic = "" if pipeline._dedup_scope == "global" else item.get("topic", "")
        source_abs = str(img_src.resolve())
        # print(
        #     f"[SaveAccepted] calling commit_final_path: "
        #     f"final={img_dst} source={source_abs} topic={dedup_topic!r} "
        #     f"scope={pipeline._dedup_scope}"
        # )

        ok, tag = pipeline.dedup.commit_final_path(
            final_path=str(img_dst),
            topic=dedup_topic,
            source_path=source_abs,
        )

        try:
            index_size = pipeline.dedup._total_indexed()
        except Exception:
            index_size = -1

        # print(
        #     f"[SaveAccepted][DedupCommit] file={name } ok={ok } tag={tag } "
        #     f"index_total={index_size } index_file={pipeline .dedup .index_file }"
        # )

        if not ok:
            # print(
            #     f"[SaveAccepted][ERROR] commit_final_path returned ok=False, "
            #     f"tag={tag }. Rolling back {img_dst }"
            # )
            img_dst.unlink(missing_ok=True)
            return None

    except Exception as e:
        import traceback

        # print(
        #     f"[SaveAccepted][ERROR] commit_final_path raised: {type (e ).__name__ }: {e }"
        # )
        traceback.print_exc()
        img_dst.unlink(missing_ok=True)
        return None

    cap = sanitize_caption(item.get("caption"))
    try:
        cap_dst.write_text(cap, encoding="utf-8")
    except Exception as e:
        # print(f"[SaveAccepted][WARN] caption write failed: {e }")
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
        # print(f"[SaveAccepted] metadata JSON written: {json_dst .name }")
    except Exception as e:
        # print(f"[SaveAccepted][WARN] meta JSON write failed: {e }")
        pass

    t = item.get("topic", "NA")
    s = sanitize_subtopic_name(item.get("subtopic", "NA"))
    with pipeline._accepted_counts_lock:
        pipeline.accepted_counts[(t, s)] += 1
    write_image_index_entry(pipeline, name, t, s, item.get("query", ""))

    # print(f"[SaveAccepted] done: {name } topic={t } subtopic={s }")
    return str(img_dst)


def save_generated(
    pipeline: OrchestratorPipeline, item: Dict[str, Any], img_path: pathlib.Path
) -> None:
    """Save a generated image and its metadata to the gen_accepted directory."""
    topic = (item.get("topic") or "NA").replace("/", "_")
    sub = (item.get("subtopic") or "NA").replace("/", "_")
    dst_dir = _ensure_dir(pipeline.gen_accepted_root / topic / sub)
    sha1 = item.get("content_sha1") or "na"
    ext = img_path.suffix.lower() or ".jpg"
    img_dst = dst_dir / f"{sha1 }{ext }"
    json_dst = dst_dir / f"{sha1 }.json"
    if not img_dst.exists():
        try:
            shutil.copy2(str(img_path), str(img_dst))
        except Exception as e:
            print(f"[GenCopyWarn] {e }")

    if pipeline.dedup.enabled:
        dedup_topic = (
            "" if pipeline._dedup_scope == "global"
            else item.get("topic", "")
        )
        try:
            pipeline.dedup.commit_final_path(
                final_path=str(img_dst),
                topic=dedup_topic,
                source_path=str(img_path),
            )
        except Exception as e:
            print(f"[GenDedupCommitError] {img_dst.name} {e}")

    with open(json_dst, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)


def sanitize_caption(cap: Optional[str]) -> str:
    """Clean a caption string into compact single-line form."""
    if cap is None:
        return ""
    cap = str(cap).strip()
    return " ".join(cap.split())


def alloc_filename(pipeline: OrchestratorPipeline, ext: str) -> Optional[str]:
    """Allocate a filename according to the configured strategy."""
    st = pipeline._filename_strategy
    if st == "uuid":
        return f"img_{uuid .uuid4 ().hex }{ext }"
    if st == "hash":
        return None
    return _next_filename_sync(pipeline, ext)


def content_hash_filename(fp: pathlib.Path) -> str:
    """Generate a unique filename based on file content hash."""
    try:
        h = sha1_of_file(fp)
    except Exception:
        h = uuid.uuid4().hex
    return f"img_{h }{fp .suffix .lower ()or '.jpg'}"


def unique_final(pipeline: OrchestratorPipeline, name: str) -> str:
    """Ensure filename uniqueness by appending a random suffix if a conflict exists."""
    if not name:
        return f"img_{uuid .uuid4 ().hex [:8 ]}.jpg"
    if not (pipeline.accepted_images_dir / name).exists():
        return name
    base, ext = name.rsplit(".", 1) if "." in name else (name, "jpg")
    return f"{base }_{uuid .uuid4 ().hex [:6 ]}.{ext }"


def write_image_index_entry(
    pipeline: OrchestratorPipeline,
    filename: str,
    topic: str,
    subtopic: str,
    query: str,
) -> bool:
    """Append an image record to the index file."""
    entry = {
        "filename": filename,
        "topic": topic,
        "subtopic": subtopic,
        "query": query,
        "timestamp": time.time(),
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    try:
        with open(pipeline._image_index_path, "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except Exception as e:
        with contextlib.suppress(Exception):
            with open(pipeline._image_index_error_path, "a", encoding="utf-8") as ferr:
                ferr.write(f"WriteFail: {e } | line={line [:200 ]}\n")
        print(f"[ImageIndexError] {e }")
        return False


def _ensure_dir(p: pathlib.Path) -> pathlib.Path:
    """Create directory if it doesn't exist and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _next_filename_sync(pipeline: OrchestratorPipeline, ext: str) -> str:
    """Allocate the next sequential filename (thread-safe)."""
    with pipeline._filename_lock_sync:
        pipeline._filename_counter += 1
        return f"img_{pipeline ._filename_counter :07d}{ext }"
