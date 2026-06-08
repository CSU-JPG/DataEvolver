"""VLM client with Ollama connection pooling and automatic image resizing."""

from __future__ import annotations

import base64
import io
import random
import time
from typing import Dict, Optional

from PIL import Image


class VLMClient:
    """Multi-instance VLM client with random load balancing and retries."""

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        qc = self.cfg.get("qwen_vl") or {}
        self.enabled = bool(qc.get("enabled", True))
        self.backend = str(qc.get("backend", "ollama")).lower()
        self.model = qc.get("model", "qwen2.5-vl:latest")
        self.verbose = bool(qc.get("verbose", False))
        self.system = "You are a helpful vision-language assistant."

        self.clients = []
        self.available = False

        if self.enabled and self.backend == "ollama":
            self._init_ollama_pool()
        else:
            self.available = False

    def _init_ollama_pool(self) -> None:
        """Build a connection pool by probing Ollama instances on ports 11434–11441."""
        import ollama

        base_port = 11434
        num_instances = 8

        valid_clients = []
        for i in range(num_instances):
            port = base_port + i
            host_url = f"http://127.0.0.1:{port }"
            try:
                client = ollama.Client(host=host_url)
                client.list()
                valid_clients.append(client)
            except Exception:
                if self.verbose:
                    print(f"[VLMClient] Port {port } is not reachable. Skipping.")

        self.clients = valid_clients
        self.available = len(self.clients) > 0

        if self.verbose:
            print(
                f"[VLMClient] Pool Ready: {len (self .clients )}/{num_instances } "
                f"instances active."
            )

    def _get_client(self):
        """Return a random client from the pool for load balancing."""
        if not self.clients:
            return None
        return random.choice(self.clients)

    def generate(
        self, image_path: str, user_prompt: str, max_retries: int = 3
    ) -> Optional[str]:
        """Run VLM inference on an image with automatic retries."""
        if not (self.enabled and self.available):
            return None

        try:
            with Image.open(image_path) as im:
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.thumbnail((512, 512), Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=85)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            print(f"[VLMClient] Image processing failed: {e }")
            return None

        msgs = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_prompt, "images": [b64]},
        ]

        for attempt in range(max_retries):
            client = self._get_client()
            if not client:
                return None

            try:
                resp = client.chat(
                    model=self.model,
                    messages=msgs,
                    stream=False,
                    options={"num_ctx": 4096},
                )
                txt = (resp.get("message", {}) or {}).get("content", "") or ""
                return " ".join(txt.split())

            except Exception as e:
                if self.verbose:
                    print(
                        f"[VLMClient] Error on attempt {attempt +1 }/{max_retries } "
                        f"(Host: {client ._client .base_url }): {e }"
                    )
                time.sleep(0.5)
                continue

        print("[VLMClient] All retries failed.")
        return None
