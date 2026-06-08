"""Perceptual-hash-based image deduplication with persistent JSON index."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

import imagehash
from PIL import Image

__all__ = [
    "file_sha1",
    "file_md5",
    "image_phash",
    "hamming",
    "phash_dedup",
    "HashDeduplicator",
]

PathLike = Union[str, Path]


def file_sha1(path: PathLike, bufsize: int = 8192) -> str:
    """Compute the SHA-1 hex digest of a file."""
    p = Path(path)
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: PathLike, bufsize: int = 8192) -> str:
    """Compute the MD5 hex digest of a file."""
    p = Path(path)
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def image_phash(path: PathLike) -> Optional[imagehash.ImageHash]:
    """Compute the perceptual hash (pHash) of an image."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return imagehash.phash(im)
    except Exception:
        return None


def hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    """Hamming distance between two perceptual hashes."""
    return int(abs(a - b))


def phash_dedup(
    paths: Iterable[PathLike],
    max_distance: int = 3,
) -> List[Path]:
    """In-memory deduplication: keep the first of each near-duplicate group."""
    kept: List[Path] = []
    kept_hashes: List[imagehash.ImageHash] = []

    for p in paths:
        path = Path(p)
        if not path.exists() or path.stat().st_size == 0:
            continue
        ph = image_phash(path)
        if ph is None:
            continue
        is_dup = any(hamming(ph, kh) <= max_distance for kh in kept_hashes)
        if not is_dup:
            kept.append(path)
            kept_hashes.append(ph)

    return kept


class _FileLock:
    """Exclusive cross-process file lock using fcntl.flock."""
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._fd: Optional[int] = None
        self._thread_lock = Lock()

    def __enter__(self) -> _FileLock:
        self._thread_lock.acquire()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        if _HAS_FCNTL:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._fd is not None:
            if _HAS_FCNTL:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        self._thread_lock.release()


def _load_raw_index(index_file: Path) -> Dict[str, Any]:
    """Load raw JSON index from disk."""
    if not index_file.exists():
        return {}
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(
            f"[HashDedup] Failed to load index {index_file }: {e }, " f"starting fresh."
        )
        return {}


def _save_raw_index(index_file: Path, data: Dict[str, Any]) -> None:
    """Atomically write index JSON to disk."""
    tmp = index_file.with_suffix(index_file.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(index_file))
    except Exception as e:
        print(f"[HashDedup] Failed to save index: {e }")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


class HashDeduplicator:
    """Persistent perceptual-hash deduplication manager with global and topic-scoped modes."""
    def __init__(self, config: Dict[str, Any], log_dir: str) -> None:
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        dedup_config = config.get("hash_deduplication", {}) or {}
        self.enabled = bool(dedup_config.get("enabled", True))
        self.max_distance = int(dedup_config.get("max_distance", 3))
        self.scope = str(dedup_config.get("scope", "topic")).strip().lower()
        self.verbose = bool(dedup_config.get("verbose", False))
        self.auto_flush = bool(dedup_config.get("auto_flush", False))
        self.strict_index_validation = bool(
            dedup_config.get("strict_index_validation", True)
        )

        index_path_cfg = dedup_config.get("index_path", "")
        if index_path_cfg:
            self.index_file = Path(index_path_cfg)
        else:
            self.index_file = self.log_dir / "hash_index.json"
        self.index_file.parent.mkdir(parents=True, exist_ok=True)

        self._lock_file = self.index_file.with_suffix(self.index_file.suffix + ".lock")
        self._lock = Lock()

        self.path_index: Dict[str, Any] = {}
        self.hash_buckets: Dict[str, Any] = {}

        self.duplicate_stats: Dict[str, Any] = {
            "total_processed": 0,
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "by_topic": {},
        }

        if self.enabled:
            self._load_and_rebuild()
            print(
                f"[HashDedup] Enabled | scope={self .scope } "
                f"| max_distance={self .max_distance } "
                f"| index={self .index_file } "
                f"| loaded={self ._total_indexed ()} paths "
                f"| auto_flush={self .auto_flush }"
            )
        else:
            print("[HashDedup] Image deduplication disabled")

    @staticmethod
    def _is_reserved_key(key: Any) -> bool:
        return isinstance(key, str) and key.startswith("__")

    def _count_entries(self, data: Dict[str, Any]) -> int:
        """Count valid hash entries in a raw index dict."""
        if self.scope == "global":
            return sum(1 for _, v in data.items() if isinstance(v, str) and v)
        return sum(
            1
            for _, topic_index in data.items()
            if isinstance(topic_index, dict)
            for _, v in topic_index.items()
            if isinstance(v, str) and v
        )

    def _validate_disk_format(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Return ``(valid, detected_scope)`` for a raw index."""
        sample_items = [(k, v) for k, v in data.items() if not self._is_reserved_key(k)]
        if not sample_items:
            return True, self.scope

        if self.scope == "global":
            bad = [k for k, v in sample_items if not isinstance(v, str)]
            if bad:
                return False, "topic"
            return True, "global"

        bad = [k for k, v in sample_items if not isinstance(v, dict)]
        if bad:
            return False, "global"
        return True, "topic"

    def _load_and_rebuild(self) -> None:
        """Load index from disk and rebuild in-memory hash buckets."""
        with _FileLock(self._lock_file):
            raw = _load_raw_index(self.index_file)

        if not raw:
            self.path_index = {}
            self.hash_buckets = {}
            return

        data = {k: v for k, v in raw.items() if not self._is_reserved_key(k)}
        if not data:
            self.path_index = {}
            self.hash_buckets = {}
            return

        ok, disk_scope = self._validate_disk_format(data)
        if not ok:
            print(
                f"[HashDedup] WARNING: disk format looks like "
                f"'{disk_scope }' scope but config scope='{self .scope }'. "
                f"Discarding disk index to avoid data corruption."
            )
            self.path_index = {}
            self.hash_buckets = {}
            return

        if self.scope == "global":
            self.path_index = {k: v for k, v in data.items() if isinstance(v, str)}
            self.hash_buckets = {}
            for abs_path, hash_str in self.path_index.items():
                if hash_str:
                    self.hash_buckets.setdefault(hash_str, []).append(abs_path)
        else:
            self.path_index = {k: v for k, v in data.items() if isinstance(v, dict)}
            self.hash_buckets = {}
            for topic, topic_index in self.path_index.items():
                buckets: Dict[str, List[str]] = {}
                for abs_path, hash_str in topic_index.items():
                    if isinstance(hash_str, str) and hash_str:
                        buckets.setdefault(hash_str, []).append(abs_path)
                self.hash_buckets[topic] = buckets

        if self.strict_index_validation:
            expected = self._count_entries(data)
            actual = self._total_indexed()
            if expected != actual:
                print(
                    f"[HashDedup] WARNING: counted entries mismatch "
                    f"after rebuild: expected={expected }, "
                    f"actual={actual }. "
                    f"Index will continue with in-memory state, "
                    f"but please inspect the file."
                )

    def _total_indexed(self) -> int:
        """Return total number of indexed paths."""
        if self.scope == "global":
            return len(self.path_index)
        return sum(len(v) for v in self.path_index.values() if isinstance(v, dict))

    def _compute_hash(self, image_path: str) -> Optional[str]:
        """Compute perceptual hash string for an image path."""
        try:
            ph = image_phash(image_path)
            return str(ph) if ph is not None else None
        except Exception as e:
            if self.verbose:
                print(f"[HashDedup] Hash failed for {image_path }: {e }")
            return None

    def _is_similar(self, h1_str: str, h2_str: str) -> bool:
        """Return ``True`` if two hash strings are within max_distance."""
        try:
            return (
                hamming(
                    imagehash.hex_to_hash(h1_str),
                    imagehash.hex_to_hash(h2_str),
                )
                <= self.max_distance
            )
        except Exception:
            return False

    def _get_tables(self, topic: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return ``(path_index, hash_buckets)`` for the current scope."""
        if self.scope == "global":
            return self.path_index, self.hash_buckets
        if topic not in self.path_index:
            self.path_index[topic] = {}
            self.hash_buckets[topic] = {}
        return self.path_index[topic], self.hash_buckets[topic]

    def _find_similar_hash_locked(
        self, current_hash: str, buckets: Dict[str, List[str]]
    ) -> Optional[str]:
        """Scan bucket keys for a hash similar to *current_hash*."""
        for stored_hash in buckets.keys():
            if self._is_similar(current_hash, stored_hash):
                return stored_hash
        return None

    def _attach_to_bucket_locked(
        self, topic: str, norm_path: str, hash_str: str
    ) -> None:
        """Add a path→hash mapping (caller must hold lock)."""
        if self.scope == "global":
            bucket = self.hash_buckets.setdefault(hash_str, [])
            if norm_path not in bucket:
                bucket.append(norm_path)
            self.path_index[norm_path] = hash_str
            return

        topic_buckets = self.hash_buckets.setdefault(topic, {})
        bucket = topic_buckets.setdefault(hash_str, [])
        if norm_path not in bucket:
            bucket.append(norm_path)
        topic_index = self.path_index.setdefault(topic, {})
        topic_index[norm_path] = hash_str

    def _remove_path_locked(self, topic: str, norm_path: str) -> bool:
        """Remove a path from its bucket (caller must hold lock)."""
        path_idx, buckets = self._get_tables(topic)
        if norm_path not in path_idx:
            return False

        stored_hash = path_idx.pop(norm_path)

        if stored_hash and stored_hash in buckets:
            kept_paths = [p for p in buckets[stored_hash] if p != norm_path]
            if kept_paths:
                buckets[stored_hash] = kept_paths
            else:
                buckets.pop(stored_hash, None)
        return True

    def _flush_locked(self) -> None:
        """Write the in-memory path_index to disk atomically."""
        with _FileLock(self._lock_file):
            _save_raw_index(self.index_file, self.path_index)

    def check_and_add(self, image_path: str, topic: str = "") -> Tuple[bool, str]:
        """Check for duplicates; if unique, add to the in-memory index."""
        if not self.enabled:
            return False, "disabled"

        norm_path = str(Path(image_path).resolve())

        with self._lock:
            path_idx, buckets = self._get_tables(topic)
            if norm_path in path_idx:
                return False, "already_seen"
            self.duplicate_stats["total_processed"] += 1

        current_hash = self._compute_hash(norm_path)

        with self._lock:
            path_idx, buckets = self._get_tables(topic)

            if current_hash is None:
                path_idx[norm_path] = ""
                if self.auto_flush:
                    self._flush_locked()
                return False, "hash_failed"

            similar_hash = self._find_similar_hash_locked(current_hash, buckets)
            if similar_hash is not None:
                self.duplicate_stats["duplicates_found"] += 1
                self.duplicate_stats["by_topic"].setdefault(topic, {"duplicates": 0})[
                    "duplicates"
                ] += 1
                return True, buckets[similar_hash][0]

            self._attach_to_bucket_locked(topic, norm_path, current_hash)
            if self.auto_flush:
                self._flush_locked()
            return False, "unique"

    def discard_path(self, image_path: str, topic: str = "") -> bool:
        """Remove a staged entry from the index (e.g. rejected sample)."""
        if not self.enabled:
            return False

        norm_path = str(Path(image_path).resolve())
        with self._lock:
            return self._remove_path_locked(topic, norm_path)

    def commit_final_path(
        self,
        final_path: str,
        topic: str = "",
        source_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Register the final on-disk path in the index."""
        if not self.enabled:
            return False, "disabled"

        norm_final = str(Path(final_path).resolve())
        norm_source = str(Path(source_path).resolve()) if source_path else ""

        with self._lock:
            path_idx, buckets = self._get_tables(topic)

            if norm_final in path_idx:
                if (
                    norm_source
                    and norm_source in path_idx
                    and norm_source != norm_final
                ):
                    self._remove_path_locked(topic, norm_source)
                stored_hash = path_idx.get(norm_final, "")
                if stored_hash:
                    if self.scope == "global":
                        bucket = buckets.setdefault(stored_hash, [])
                        if norm_final not in bucket:
                            bucket.append(norm_final)
                    else:
                        tb = buckets.setdefault(topic, {})
                        bucket = tb.setdefault(stored_hash, [])
                        if norm_final not in bucket:
                            bucket.append(norm_final)
                self._flush_locked()
                return True, "already_committed"
            stored_hash = ""
            migrated = False

            if norm_source and norm_source in path_idx:
                stored_hash = path_idx.pop(norm_source) or ""
                if stored_hash:
                    if self.scope == "global":
                        bucket = buckets.setdefault(stored_hash, [])
                        bucket = [p for p in bucket if p != norm_source]
                        if norm_final not in bucket:
                            bucket.append(norm_final)
                        buckets[stored_hash] = bucket
                        path_idx[norm_final] = stored_hash
                    else:
                        tb = buckets.setdefault(topic, {})
                        bucket = tb.setdefault(stored_hash, [])
                        bucket = [p for p in bucket if p != norm_source]
                        if norm_final not in bucket:
                            bucket.append(norm_final)
                        tb[stored_hash] = bucket
                        path_idx[norm_final] = stored_hash
                    migrated = True
                else:
                    path_idx.pop(norm_source, None)

        if not migrated:
            computed_hash = self._compute_hash(norm_final) or ""
            with self._lock:
                path_idx, buckets = self._get_tables(topic)

                if norm_final in path_idx:
                    if (
                        norm_source
                        and norm_source in path_idx
                        and norm_source != norm_final
                    ):
                        self._remove_path_locked(topic, norm_source)
                    self._flush_locked()
                    return True, "already_committed"

                if computed_hash:
                    similar_hash = self._find_similar_hash_locked(
                        computed_hash, buckets
                    )
                    if similar_hash is not None:
                        self._attach_to_bucket_locked(topic, norm_final, similar_hash)
                        self._flush_locked()
                        return True, "committed_to_existing_bucket"

                    self._attach_to_bucket_locked(topic, norm_final, computed_hash)
                    self._flush_locked()
                    return True, "committed"

                if self.scope == "global":
                    path_idx[norm_final] = ""
                else:
                    topic_index = self.path_index.setdefault(topic, {})
                    topic_index[norm_final] = ""
                self._flush_locked()
                return True, "hash_failed"

        with self._lock:
            self._flush_locked()
        return True, "migrated"

    def flush(self) -> None:
        """Persist the in-memory index to disk."""
        with self._lock:
            self._flush_locked()
        if self.verbose:
            print(
                f"[HashDedup] Index flushed to {self .index_file } "
                f"(total={self ._total_indexed ()} entries)"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Return a copy of the duplicate statistics."""
        with self._lock:
            return self.duplicate_stats.copy()

    def save_stats(self, filename: str = "hash_deduplication_stats.json") -> None:
        """Write statistics to a timestamped JSON file in the log dir."""
        stats_file = self.log_dir / filename
        stats_data = {
            "timestamp": time.time(),
            "config": {
                "enabled": self.enabled,
                "scope": self.scope,
                "max_distance": self.max_distance,
            },
            "stats": self.duplicate_stats,
        }
        try:
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[HashDedup] Failed to save stats: {e }")

    def cleanup_duplicates(self, remove_duplicates: bool = False) -> int:
        """Remove duplicate entries from the index (and optionally disk)."""
        if not self.enabled:
            return 0

        removed_count = 0
        all_buckets = (
            [self.hash_buckets]
            if self.scope == "global"
            else list(self.hash_buckets.values())
        )

        with self._lock:
            for buckets in all_buckets:
                for hash_val, file_paths in list(buckets.items()):
                    if len(file_paths) <= 1:
                        continue
                    for dup_path in file_paths[1:]:
                        removed_count += 1
                        if remove_duplicates:
                            try:
                                Path(dup_path).unlink(missing_ok=True)
                                if self.verbose:
                                    print(f"[HashDedup] Removed: " f"{dup_path }")
                            except Exception as e:
                                if self.verbose:
                                    print(
                                        f"[HashDedup] Remove failed "
                                        f"{dup_path }: {e }"
                                    )
                    buckets[hash_val] = [file_paths[0]]

            self.duplicate_stats["duplicates_removed"] += removed_count
            self._flush_locked()

        return removed_count
