"""Image quality assessment via geometric, typographic, and photometric metrics."""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from langdetect import detect as lang_detect
from PIL import Image
from rapidfuzz.distance import Levenshtein

__all__ = ["QualityAssessor"]


def _quad_h(quad: List[List[int]]) -> float:
    """Height of a quadrilateral bounding box."""
    ys = [p[1] for p in quad]
    return float(max(ys) - min(ys))


def _bbox_from_quad(quad: List[List[int]]) -> Tuple[int, int, int, int]:
    """Axis-aligned bounding rectangle from a quadrilateral."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _angle_from_quad(quad: List[List[int]]) -> float:
    """Estimated orientation angle of a text line (degrees)."""
    q = sorted(quad, key=lambda p: (p[1], p[0]))[:2]
    if len(q) < 2:
        return 0.0
    dx = (q[1][0] - q[0][0]) + 1e-6
    dy = q[1][1] - q[0][1]
    return float(np.degrees(np.arctan2(dy, dx)))


class QualityAssessor:
    """Computes quality metrics for an image + OCR result pair."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    @staticmethod
    def _coverage(img_wh: Tuple[int, int], boxes: List[List[List[int]]]) -> float:
        """Fraction of image pixels covered by OCR bounding boxes."""
        W, H = img_wh
        mask = np.zeros((H, W), dtype=np.uint8)
        for box in boxes:
            poly = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
        return float(mask.sum()) / float(W * H)

    @staticmethod
    def _legibility(confs: List[float]) -> float:
        """Mean OCR confidence across all detections."""
        if not confs:
            return 0.0
        return float(np.mean([c for c in confs if c is not None]))

    def _language_ok(self, texts: List[str]) -> Tuple[bool, str]:
        """Check whether detected language matches the allow-list."""
        joined = " ".join(texts)[:2000]
        try:
            lang = lang_detect(joined) if joined.strip() else "unknown"
        except Exception:
            lang = "unknown"
        allow = set([l.lower() for l in self.cfg["quality"].get("allow_languages", [])])
        return (not allow) or (lang.lower() in allow), lang

    @staticmethod
    def _bg_fg_contrast(img: np.ndarray) -> float:
        """Per-pixel value-channel standard deviation (proxy for contrast)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v = hsv[..., 2].astype(np.float32) / 255.0
        return float(np.std(v))

    @staticmethod
    def _sharpness(gray: np.ndarray) -> float:
        """Variance of the Laplacian — higher = sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _clutter_index(gray: np.ndarray, text_mask: np.ndarray) -> float:
        """Fraction of Canny edges that lie *outside* the text regions."""
        edges = cv2.Canny(gray, 50, 150)
        non_text = cv2.bitwise_and(edges, edges, mask=(255 - text_mask))
        return float(non_text.sum()) / float(
            (non_text.shape[0] * non_text.shape[1]) * 255.0 + 1e-6
        )

    @staticmethod
    def _char_stats(
        boxes: List[List[List[int]]],
    ) -> Tuple[float, float, int]:
        """Mean character height, coefficient-of-variation, and count."""
        hs = []
        for q in boxes:
            try:
                hs.append(_quad_h(q))
            except Exception:
                pass
        if not hs:
            return 0.0, 1.0, 0
        arr = np.array(hs, dtype=np.float32)
        mean = float(np.mean(arr))
        cv_ = float(np.std(arr) / (mean + 1e-6))
        return mean, cv_, len(hs)

    @staticmethod
    def _line_stats(
        boxes: List[List[List[int]]],
    ) -> Tuple[int, float]:
        """Estimate number of text lines and angular dispersion."""
        if not boxes:
            return 0, 999.0

        hs = []
        centers_y = []
        angles = []
        for q in boxes:
            _, y1, _, y2 = _bbox_from_quad(q)
            centers_y.append((y1 + y2) / 2.0)
            hs.append(_quad_h(q))
            angles.append(_angle_from_quad(q))

        centers_y_arr = np.array(centers_y, dtype=np.float32)
        h_mean = float(np.mean(hs)) if hs else 1.0
        if h_mean <= 0:
            h_mean = 1.0

        idx = np.argsort(centers_y_arr)
        lines = []
        cur = [centers_y_arr[idx[0]]]
        for i in idx[1:]:
            if abs(centers_y_arr[i] - cur[-1]) <= 0.8 * h_mean:
                cur.append(centers_y_arr[i])
            else:
                lines.append(cur)
                cur = [centers_y_arr[i]]
        if cur:
            lines.append(cur)

        line_count = len(lines)
        angle_std = float(np.std(angles)) if angles else 999.0
        return int(line_count), angle_std

    @staticmethod
    def _centrality(W: int, H: int, boxes: List[List[List[int]]]) -> float:
        """Fraction of text-polygon pixels inside the central 70 % region."""
        cx1, cy1 = int(W * 0.15), int(H * 0.15)
        cx2, cy2 = int(W * 0.85), int(H * 0.85)
        center_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(center_mask, (cx1, cy1), (cx2, cy2), 255, -1)

        poly_mask = np.zeros((H, W), dtype=np.uint8)
        for box in boxes:
            poly = np.array(box, dtype=np.int32)
            cv2.fillPoly(poly_mask, [poly], 255)

        inter = cv2.bitwise_and(center_mask, poly_mask)
        return float(inter.sum()) / float(poly_mask.sum() + 1e-6)

    @staticmethod
    def _border_margin(W: int, H: int, boxes: List[List[List[int]]]) -> float:
        """Fraction of text pixels falling within a 5 %-wide border strip."""
        margin = int(min(W, H) * 0.05)
        border_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(border_mask, (0, 0), (W - 1, margin), 255, -1)
        cv2.rectangle(border_mask, (0, H - margin), (W - 1, H - 1), 255, -1)
        cv2.rectangle(border_mask, (0, 0), (margin, H - 1), 255, -1)
        cv2.rectangle(border_mask, (W - margin, 0), (W - 1, H - 1), 255, -1)

        poly_mask = np.zeros((H, W), dtype=np.uint8)
        for box in boxes:
            poly = np.array(box, dtype=np.int32)
            cv2.fillPoly(poly_mask, [poly], 255)

        inter = cv2.bitwise_and(border_mask, poly_mask)
        return float(inter.sum()) / float(poly_mask.sum() + 1e-6)

    def score(self, image_path: str, ocr_rec: Dict[str, Any]) -> Dict[str, Any]:
        """Compute all quality metrics for a single image."""
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        min_side_val = min(W, H)
        max_side_val = max(W, H)
        aspect_ratio = float(W) / float(H + 1e-6)
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        boxes: List[List[List[int]]] = ocr_rec.get("boxes", []) or []
        confs: List[float] = ocr_rec.get("confs", []) or []
        texts: List[str] = ocr_rec.get("texts", []) or []

        coverage = self._coverage((W, H), boxes) if boxes else 0.0
        legibility = self._legibility(confs)
        words = len(texts)
        sharp = self._sharpness(gray)
        contrast = self._bg_fg_contrast(arr)

        char_h_mean, char_h_cv, _ = self._char_stats(boxes)
        line_count, line_angle_std = self._line_stats(boxes)

        text_mask = np.zeros((H, W), dtype=np.uint8)
        for box in boxes:
            poly = np.array(box, dtype=np.int32)
            cv2.fillPoly(text_mask, [poly], 255)
        clutter = self._clutter_index(gray, text_mask)

        centrality = self._centrality(W, H, boxes) if boxes else 0.0
        border_ratio = self._border_margin(W, H, boxes) if boxes else 0.0

        lang_ok, lang = self._language_ok(texts)

        text_density = float(words) / float((W * H) / 1e6 + 1e-6)
        char_count = sum(len(t) for t in texts)
        char_density = float(char_count) / float((W * H) / 1e6 + 1e-6)

        wm_cfg = self.cfg.get("watermark", {}) or {}
        wm_enabled = bool(wm_cfg.get("enabled", True))
        has_watermark = False
        methods_triggered: List[str] = []
        watermark_keywords: List[str] = []
        vote_count = 0

        if wm_enabled:
            alamy_pattern = re.compile(r"alamy", re.IGNORECASE)
            for text in texts:
                if alamy_pattern.search(text):
                    has_watermark = True
                    watermark_keywords.append("alamy")
                    break
            if has_watermark:
                methods_triggered.append("ocr_keyword")
                vote_count = 1

        return {
            "coverage": coverage,
            "legibility": legibility,
            "words": words,
            "sharpness": sharp,
            "contrast": contrast,
            "char_h_mean": char_h_mean,
            "char_h_cv": char_h_cv,
            "line_count": line_count,
            "line_angle_std": line_angle_std,
            "clutter_index": clutter,
            "centrality": centrality,
            "border_ratio": border_ratio,
            "text_density": text_density,
            "char_density": char_density,
            "lang": lang,
            "lang_ok": lang_ok,
            "min_side_val": min_side_val,
            "max_side_val": max_side_val,
            "aspect_ratio": aspect_ratio,
            "has_watermark": has_watermark,
            "watermark_methods": methods_triggered,
            "watermark_keywords": watermark_keywords,
            "watermark_vote_count": vote_count,
        }

    def check(
        self,
        q: Dict[str, Any],
        ocr_rec: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Apply hard thresholds; return (accepted, reasons). Deprecated."""
        warnings.warn(
            "QualityAssessor.check() is deprecated; use LLMQualityDecider.",
            DeprecationWarning,
        )

        reasons: List[str] = []
        Q = self.cfg["quality"]
        policy = (Q.get("lang_policy", "require") or "require").lower()

        def fail(cond: bool, code: str) -> None:
            if cond:
                reasons.append(code)

        fail(q["coverage"] < Q.get("min_ocr_coverage", 0.04), "coverage_lt")
        fail(q["legibility"] < Q.get("min_legibility", 0.55), "legibility_lt")
        fail(q["words"] < self.cfg["ocr"].get("min_words", 4), "words_lt")
        fail(q["sharpness"] < Q.get("min_sharpness", 35.0), "sharpness_lt")

        fail(q["char_h_mean"] < Q.get("min_char_h_px", 10), "char_h_mean_lt")
        fail(q["char_h_cv"] > Q.get("max_char_h_cv", 0.80), "char_h_cv_gt")
        fail(
            q["line_angle_std"] > Q.get("max_line_angle_std", 12.0),
            "line_angle_std_gt",
        )

        fail(q["centrality"] < Q.get("min_centrality", 0.35), "centrality_lt")
        fail(
            q["border_ratio"] > Q.get("max_border_ratio", 0.35),
            "border_ratio_gt",
        )

        min_char_density = Q.get("min_char_density", 80)
        if "char_density" in q and q["char_density"] is not None:
            fail(q["char_density"] < min_char_density, "char_density_lt")
        else:
            fail(
                q["text_density"] < Q.get("min_text_density", 20),
                "text_density_lt",
            )

        fail(q["clutter_index"] > Q.get("max_clutter", 0.18), "clutter_gt")
        fail(
            q.get("contrast", 0.0) < Q.get("min_contrast", 0.08),
            "contrast_lt",
        )

        fail(
            q.get("min_side_val", 0) < Q.get("min_side", 512),
            "min_side_lt",
        )
        fail(
            q.get("max_side_val", 0) > Q.get("max_side", 4096),
            "max_side_gt",
        )
        fail(q.get("aspect_ratio", 1.0) < Q.get("min_ar", 0.35), "ar_lt")
        fail(q.get("aspect_ratio", 1.0) > Q.get("max_ar", 3.0), "ar_gt")

        wm_cfg = self.cfg.get("watermark", {}) or {}
        if wm_cfg.get("enabled", True):
            fail(q.get("has_watermark", False), "has_watermark")

        if policy == "require":
            if not q.get("lang_ok", True):
                reasons.append("lang_not_allowed")

        accepted = len(reasons) == 0
        return accepted, reasons

    def accept(self, q: Dict[str, Any], ocr_rec: Dict[str, Any]) -> bool:
        """Legacy boolean accept/reject.  Prefer :meth:`check`."""
        warnings.warn(
            "QualityAssessor.accept() is deprecated; use LLMQualityDecider.",
            DeprecationWarning,
        )
        ok, _ = self.check(q, ocr_rec)
        return ok

    @staticmethod
    def make_prompt(subtopic: str) -> str:
        """Generate a quality-oriented generation prompt for a subtopic."""
        return (
            f"A high-resolution, text-dominant image of '{subtopic }', "
            f"with clean grid layout, strong contrast, large readable "
            f"glyphs, and consistent line orientation. "
            f"No watermark or brand logos."
        )

    @staticmethod
    def match_prompt(prompt: str, ocr_texts: List[str]) -> float:
        """Levenshtein-based fuzzy match between prompt and OCR output."""
        txt = " ".join(ocr_texts)[:2000]
        ratio = 1.0 - Levenshtein.normalized_distance(prompt.lower(), txt.lower())
        return float(ratio)
