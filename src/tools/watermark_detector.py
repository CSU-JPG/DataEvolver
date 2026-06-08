"""Lightweight watermark detection via OCR keyword matching."""

from __future__ import annotations

import re
from typing import List

__all__ = ["check_alamy"]

_ALAMY_RE = re.compile(r"alamy", re.IGNORECASE)


def check_alamy(ocr_texts: List[str]) -> bool:
    """Check whether any OCR text line contains the 'alamy' watermark."""
    for text in ocr_texts:
        if _ALAMY_RE.search(text):
            return True
    return False
