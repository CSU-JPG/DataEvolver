"""Unified OCR engine wrapper supporting PaddleOCR and Tesseract backends."""

from __future__ import annotations

import asyncio
import fcntl
import inspect
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from paddleocr import PaddleOCR as PP

    _HAS_PADDLE = True
except Exception:
    print("*" * 100)
    print("PaddleOCR not installed")
    print("*" * 100)
    _HAS_PADDLE = False

try:
    import pytesseract

    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

_INIT_LOCK_PATH = os.path.join(tempfile.gettempdir(), "paddleocr_init.lock")

__all__ = ["OCR"]


class OCR:
    """Unified OCR engine with automatic backend selection."""

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        device_id: Optional[int] = None,
    ) -> None:
        self.cfg = cfg or {}
        ocr_cfg = self.cfg.get("ocr") or {}
        self.engine = str(ocr_cfg.get("engine", "paddle")).lower()
        self.lang = str(ocr_cfg.get("lang", "ch")).lower()
        self.use_gpu = bool(ocr_cfg.get("use_gpu", True))
        self.device_id = (
            device_id if device_id is not None else int(ocr_cfg.get("device_id", 0))
        )
        self.paddle = None
        self._paddle_fail_reason: Optional[str] = None
        self._lock = threading.Lock()

        if self.engine == "paddle":
            self._init_paddle(ocr_cfg)
        else:
            self.tesseract = pytesseract

    def run(self, image_path: str) -> Dict[str, Any]:
        """Synchronous OCR call (thread-safe). Validates image readability first."""
        with self._lock:
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as img_err:
                print(
                    f"[OCR][Error] Corrupt or unreadable image: "
                    f"{image_path }, {img_err }"
                )
                return {
                    "error": "corrupt_image",
                    "boxes": [],
                    "texts": [],
                    "confs": [],
                    "avg_conf": 0.0,
                    "word_count": 0,
                }

            if self.engine == "paddle":
                if self.paddle is None:
                    return {
                        "error": "paddle_not_initialized",
                        "boxes": [],
                        "texts": [],
                        "confs": [],
                        "avg_conf": 0.0,
                        "word_count": 0,
                    }
                return self._run_paddle(image_path)

            return self._run_tesseract(image_path)

    async def arun(self, image_path: str) -> Dict[str, Any]:
        """Async wrapper that offloads run() to the default executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, image_path)

    def _init_paddle(self, ocr_cfg: Dict[str, Any]) -> None:
        """Initialise PaddleOCR with fcntl file-lock to serialise model loading across processes."""
        self._paddle_fail_reason = None
        if self.engine != "paddle":
            return
        if not _HAS_PADDLE:
            self._paddle_fail_reason = "import_paddleocr_failed"
            print("[OCR][Fallback] PaddleOCR import failed → tesseract")
            self.engine = "tesseract"
            self._check_fallback()
            return

        try:
            with open(_INIT_LOCK_PATH, "w") as lk:
                fcntl.flock(lk.fileno(), fcntl.LOCK_EX)

                lang = ocr_cfg.get("lang", "ch")
                use_angle_cls = bool(ocr_cfg.get("use_angle_cls", True))
                show_log = bool(ocr_cfg.get("show_log", False))
                use_gpu = bool(ocr_cfg.get("use_gpu", True))

                print(
                    f"[OCR] Init PaddleOCR (locked) lang={lang } "
                    f"gpu={use_gpu } device_id={self .device_id }"
                )

                if use_gpu:
                    try:
                        import paddle

                        paddle.device.set_device(f"gpu:{self .device_id }")
                    except Exception as se:
                        print(
                            f"[OCR][Warn] set_device gpu:{self .device_id } "
                            f"failed: {se }"
                        )

                sig = inspect.signature(PP.__init__)
                k: Dict[str, Any] = {
                    "use_angle_cls": use_angle_cls,
                    "lang": lang,
                }
                if "show_log" in sig.parameters:
                    k["show_log"] = show_log
                if "use_gpu" in sig.parameters:
                    k["use_gpu"] = use_gpu

                self.paddle = PP(**k)

                try:
                    import paddle

                    print(
                        f"[OCR][Debug] "
                        f"paddle.is_cuda={paddle .device .is_compiled_with_cuda ()} "
                        f"current_device={paddle .device .get_device ()} "
                        f"PP.use_gpu={getattr (self .paddle ,'use_gpu',None )}"
                    )
                except Exception as ie:
                    print(f"[OCR][Debug] paddle state check failed: {ie }")

                print(
                    f"[OCR] PaddleOCR OK on "
                    f"gpu:{self .device_id if use_gpu else 'cpu'}"
                )

        except Exception as e:
            self._paddle_fail_reason = str(e)
            print(f"[OCR] Paddle init failed: {e } → fallback tesseract")
            self.paddle = None
            self.engine = "tesseract"
            self._check_fallback()

    def _check_fallback(self) -> None:
        """Raise RuntimeError if no OCR backend is available."""
        if self.engine == "tesseract" and not _HAS_TESS:
            raise RuntimeError(
                "OCR initialization failed: neither PaddleOCR nor Tesseract "
                "is available. Install 'paddleocr' (and 'paddlepaddle[-gpu]') "
                "or 'pytesseract'."
            )
        if _HAS_TESS:
            try:
                import shutil

                tesseract_path = shutil.which("tesseract")
                if tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                else:
                    print(
                        "[OCR][Warning] Tesseract executable not found in "
                        "PATH. Pytesseract might fail."
                    )
            except Exception as e:
                print(f"[OCR][Warning] Failed to set tesseract_cmd path: {e }")

    @staticmethod
    def _to_quad(box: Any) -> Optional[List[List[int]]]:
        """Normalise a detection box into a [[x,y], ...] × 4 list."""
        try:
            pts: List[List[int]] = []
            if isinstance(box, dict) and "points" in box:
                box = box["points"]
            for p in box:
                pts.append([int(round(p[0])), int(round(p[1]))])
            if len(pts) == 4:
                return pts
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_text_conf(meta: Any) -> Tuple[str, float]:
        """Extract (text, confidence) from a Paddle result element."""
        txt, conf = "", 0.0
        try:
            if isinstance(meta, (list, tuple)):
                if len(meta) >= 2:
                    txt = str(meta[0] or "")
                    conf = float(meta[1])
                elif len(meta) == 1:
                    txt = str(meta[0] or "")
            elif isinstance(meta, dict):
                txt = str(meta.get("text", "") or "")
                conf = float(meta.get("score", meta.get("confidence", 0.0)) or 0.0)
            elif isinstance(meta, str):
                txt = meta
        except Exception:
            pass
        return txt, conf

    def _run_paddle(self, image_path: str) -> Dict[str, Any]:
        """Execute PaddleOCR and return a normalised result dict."""
        assert self.paddle is not None, "PaddleOCR not initialized"

        try:
            res = self.paddle.ocr(image_path, cls=True)
        except TypeError:
            res = self.paddle.ocr(image_path)

        boxes: List[List[List[int]]] = []
        texts: List[str] = []
        confs: List[float] = []

        for page in res or []:
            if not page:
                continue
            for line in page or []:
                if not line:
                    continue

                box = None
                meta = None
                try:
                    box = line[0]
                    meta = line[1] if len(line) > 1 else None
                except Exception:
                    if isinstance(line, dict):
                        box = line.get("box") or line.get("points")
                        meta = line.get("res") or line.get("txt") or line.get("data")

                quad = self._to_quad(box) if box is not None else None
                if quad is None:
                    continue

                text, conf = self._parse_text_conf(meta)
                text = (text or "").strip()
                if not text:
                    continue

                try:
                    if conf > 1.0:
                        conf = conf / 100.0
                except Exception:
                    conf = 0.0

                boxes.append(quad)
                texts.append(text)
                confs.append(float(conf))

        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        word_count = sum(1 for t in texts if t.strip())

        return {
            "boxes": boxes,
            "texts": texts,
            "confs": confs,
            "avg_conf": avg_conf,
            "word_count": int(word_count),
        }

    def _run_tesseract(self, image_path: str) -> Dict[str, Any]:
        """Execute Tesseract and return a normalised result dict."""
        if not _HAS_TESS:
            raise RuntimeError(
                "pytesseract not installed but requested as OCR fallback."
            )

        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        boxes: List[List[List[int]]] = []
        texts: List[str] = []
        confs: List[float] = []

        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                x, y = int(data["left"][i]), int(data["top"][i])
                w, h = int(data["width"][i]), int(data["height"][i])
                conf_raw = data.get("conf", ["-1"] * n)[i]
                conf = float(conf_raw) if conf_raw != "-1" else 0.0
            except Exception:
                continue

            boxes.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            texts.append(txt)

            confs.append(conf / 100.0)

        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        word_count = sum(len(t.strip()) > 0 for t in texts)

        return {
            "boxes": boxes,
            "texts": texts,
            "confs": confs,
            "avg_conf": avg_conf,
            "word_count": int(word_count),
        }
