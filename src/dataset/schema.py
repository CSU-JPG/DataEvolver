"""Pydantic data models for the DataEvolver dataset."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OCRRecord(BaseModel):
    """OCR result for a single image."""

    boxes: List[List[List[int]]] = Field(default_factory=list)
    texts: List[str] = Field(default_factory=list)
    confs: List[float] = Field(default_factory=list)
    avg_conf: Optional[float] = None
    word_count: Optional[int] = None


class QualityRecord(BaseModel):
    """Quality assessment result for a single image."""

    coverage: float
    legibility: float
    words: int
    sharpness: float
    language_ok: bool = True


class DataItem(BaseModel):
    """Complete data record for one image in the dataset."""

    image_path: str
    topic: str
    subtopic: str
    source: str = "bing-noapi"
    is_generated: bool = False

    query: Optional[str] = None
    prompt: Optional[str] = None
    source_url: Optional[str] = None
    license: Optional[str] = None

    ocr: OCRRecord
    quality: QualityRecord

    content_sha1: Optional[str] = None
    shard: Optional[str] = None
    ts: Optional[float] = None

    extras: Optional[Any] = None

    qwen_vl: Optional[List[Dict[str, Any]]] = None
    t2i: Optional[Dict[str, Any]] = None
