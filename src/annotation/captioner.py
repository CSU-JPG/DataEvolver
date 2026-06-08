"""Layout caption builder for text-dominant image typography."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _token_stats(texts: List[str]) -> Dict[str, Any]:
    """Count Latin, CJK, and digit characters across a list of texts."""
    joined = " ".join([t for t in texts if t])
    latin = len(re.findall(r"[A-Za-z]", joined))
    han = len(re.findall(r"[一-鿿]", joined))
    digits = len(re.findall(r"[0-9]", joined))
    return {"latin": latin, "han": han, "digits": digits}


def _layout_brief(q: Dict[str, Any]) -> str:
    """Produce a one-line summary of layout metrics."""
    lc = q.get("line_count", 0)
    dens = int(q.get("text_density", 0))
    ch = int(q.get("char_h_mean", 0))
    return f"{lc } lines, avg glyph height ~{ch }px, density ~{dens } words/MP."


def build_layout_caption(
    ocr_rec: Dict[str, Any],
    q: Dict[str, Any],
    topic: str,
    subtopic: str,
) -> str:
    """Build a concise layout/typography caption without reproducing long text."""
    texts = ocr_rec.get("texts", []) or []
    tok = _token_stats(texts)
    parts = []

    parts.append(
        f"Text-dominant {topic .replace ('_',' ')} / {subtopic .replace ('_',' ')}."
    )

    if tok["han"] > tok["latin"]:
        parts.append("Primary language: Chinese; secondary English.")
    elif tok["latin"] > tok["han"]:
        parts.append("Primary language: English; secondary Chinese.")
    else:
        parts.append("Bilingual layout.")

    parts.append(_layout_brief(q))

    if q.get("contrast", 0) >= 0.16:
        parts.append("High contrast between text and background.")
    if q.get("centrality", 0) >= 0.6:
        parts.append("Text blocks centered within the canvas.")
    if q.get("line_angle_std", 999) <= 4.5:
        parts.append("Lines are horizontally aligned.")
    if q.get("char_h_cv", 1.0) <= 0.45:
        parts.append("Consistent glyph sizes across lines.")
    if "sign" in topic.lower() or "sign" in subtopic.lower():
        parts.append("Legible signage style with bold typography.")
    if "menu" in topic.lower() or "菜单" in subtopic:
        parts.append("Multi-column menu layout with clear sections.")
    if "blackboard" in subtopic.lower() or ("poster" in topic.lower()):
        parts.append("Dense text blocks suitable for chalkboard/poster rendering.")

    return " ".join(parts)
