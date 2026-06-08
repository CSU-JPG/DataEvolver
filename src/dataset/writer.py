"""Persists accepted image records to JSONL files."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List

try:
    import ujson as jsonlib
except Exception:
    import json as jsonlib

from .schema import DataItem, OCRRecord, QualityRecord


class DatasetWriter:
    """Write accepted samples to JSONL with optional train/dev split."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.ann_dir = pathlib.Path(
            self.cfg.get("paths", {}).get("ann_dir", "./data/annotations")
        ).resolve()
        self.ann_dir.mkdir(parents=True, exist_ok=True)

        split_cfg = self.cfg.get("split", {})
        self.train_ratio: float = float(split_cfg.get("train_ratio", 1.0))
        self._buf: List[Dict[str, Any]] = []

        self.train_fp = self.ann_dir / "train.jsonl"
        self.dev_fp = self.ann_dir / "dev.jsonl"

    def init_files(self) -> None:
        """Create parent directories for output files."""
        self.train_fp.parent.mkdir(parents=True, exist_ok=True)
        if self.train_ratio < 1.0:
            self.dev_fp.parent.mkdir(parents=True, exist_ok=True)

    def write_record(self, **kwargs) -> None:
        """Validate via Pydantic schema and buffer a record."""
        if isinstance(kwargs.get("ocr"), dict):
            kwargs["ocr"] = OCRRecord(**kwargs["ocr"])
        if isinstance(kwargs.get("quality"), dict):
            kwargs["quality"] = QualityRecord(**kwargs["quality"])
        item = DataItem(**kwargs)
        self._buf.append(item.dict(by_alias=False, exclude_none=True))

    def flush(self) -> None:
        """Write buffered records to JSONL, splitting train/dev by ratio."""
        if not self._buf:
            return

        n = len(self._buf)
        k = n if self.train_ratio >= 1.0 else int(n * self.train_ratio)
        train_part = self._buf[:k]
        dev_part = self._buf[k:] if self.train_ratio < 1.0 else []

        with open(self.train_fp, "a", encoding="utf-8") as f_tr:
            for it in train_part:
                f_tr.write(jsonlib.dumps(it, ensure_ascii=False) + "\n")

        if dev_part:
            with open(self.dev_fp, "a", encoding="utf-8") as f_dev:
                for it in dev_part:
                    f_dev.write(jsonlib.dumps(it, ensure_ascii=False) + "\n")

        self._buf.clear()
