"""Ollama chat API wrapper with robust JSON-list parsing."""

import ast
import json
import re
from typing import Any, List, Optional

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _strip_code_fence(s: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```)."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _find_balanced_json_array(s: str) -> Optional[str]:
    """Locate the first bracket-balanced JSON array substring."""
    start = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "[":
            if start is None:
                start = i
            depth += 1
        elif ch == "]":
            if start is not None:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    m = _JSON_ARRAY_RE.search(s)
    return m.group(0) if m else None


def _remove_trailing_commas(s: str) -> str:
    """Remove trailing commas before ``}`` or ``]``."""
    return re.sub(r",\s*([}\]])", r"\1", s)


def _object_to_string(x: Any) -> Optional[str]:
    """Best-effort conversion of a parsed JSON value to a string."""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, dict):
        for k in ["query", "subject", "title", "keyword", "text", "name"]:
            if k in x and isinstance(x[k], (str, int, float)):
                return str(x[k]).strip()
        parts = []
        for k, v in x.items():
            if isinstance(v, (str, int, float)):
                parts.append(str(v))
        if parts:
            return " ".join(p.strip() for p in parts if str(p).strip())
        return None
    if isinstance(x, (list, tuple)) and x:
        return _object_to_string(x[0])
    return None


def _lenient_array_items(arr_inner: str) -> List[Any]:
    """Lenient parser for JSON array contents with missing commas, single quotes, etc."""
    items: List[Any] = []
    i, n = 0, len(arr_inner)

    def _skip_ws(j: int) -> int:
        while j < n and arr_inner[j].isspace():
            j += 1
        return j

    while True:
        i = _skip_ws(i)
        if i >= n:
            break
        ch = arr_inner[i]

        if ch == '"':
            j = i + 1
            escape = False
            while j < n:
                cj = arr_inner[j]
                if escape:
                    escape = False
                elif cj == "\\":
                    escape = True
                elif cj == '"':
                    break
                j += 1
            try:
                s = json.loads(arr_inner[i : j + 1])
                items.append(s)
            except Exception:
                items.append(arr_inner[i + 1 : j])
            i = j + 1
            continue

        if ch == "'":
            j = i + 1
            while j < n and arr_inner[j] != "'":
                j += 1
            items.append(arr_inner[i + 1 : j])
            i = j + 1
            continue

        if ch == "{":
            brace = 1
            j = i + 1
            while j < n and brace > 0:
                if arr_inner[j] == "{":
                    brace += 1
                elif arr_inner[j] == "}":
                    brace -= 1
                j += 1
            obj_str = arr_inner[i:j]
            obj_str = _remove_trailing_commas(obj_str)
            try:
                obj = json.loads(obj_str)
            except Exception:
                try:
                    obj = ast.literal_eval(obj_str)
                except Exception:
                    obj = None
            if obj is not None:
                items.append(obj)
            i = j
            continue

        if ch == "[":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if arr_inner[j] == "[":
                    depth += 1
                elif arr_inner[j] == "]":
                    depth -= 1
                j += 1
            sub = arr_inner[i + 1 : j - 1].strip()
            if sub:
                items.append(sub[:128])
            i = j
            continue

        j = i
        while j < n and arr_inner[j] not in [",", "\n", "\r", "\t", " "]:
            j += 1
        token = arr_inner[i:j].strip().strip(",")
        if token:
            try:
                parsed = json.loads(token)
            except Exception:
                parsed = token
            items.append(parsed)
        i = j + 1

    return items


class OllamaLLM:
    """Minimal Ollama chat client with JSON-array response parsing."""
    def __init__(
        self,
        model: str = "mistral:latest",
        num_ctx: int = 32000,
        base_url: str = "http://127.0.0.1:11434",
    ):
        import ollama

        self.model = model
        self.num_ctx = num_ctx
        self.base_url = base_url
        self.client = ollama.Client(host=self.base_url)

    def ask(self, prompt: str, system: Optional[str] = None) -> str:
        """Send a prompt and return the model's raw text response."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        resp = self.client.chat(
            model=self.model,
            messages=msgs,
            stream=False,
            options={"num_ctx": self.num_ctx},
        )
        return resp["message"]["content"].strip()

    def ask_json_list(self, prompt: str, system: Optional[str] = None) -> List[str]:
        """Send a prompt and parse the response as a best-effort ``List[str]``.
        """
        try:
            txt = self.ask(
                prompt + "\n\nOutput only a JSON array. No explanations.", system
            )
        except Exception:
            return []

        txt = _strip_code_fence(txt)
        arr_str = _find_balanced_json_array(txt)
        if not arr_str:
            arr_str = "[" + txt + "]"

        try:
            raw = json.loads(_remove_trailing_commas(arr_str))
        except Exception:

            clean = re.sub(r"^\s*//.*?$", "", arr_str, flags=re.M)
            clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.S)
            clean = _remove_trailing_commas(clean)
            try:
                raw = json.loads(clean)
            except Exception:

                inner = clean.strip()
                if inner.startswith("[") and inner.endswith("]"):
                    inner = inner[1:-1]
                tokens = _lenient_array_items(inner)
                raw = tokens

        if not isinstance(raw, list):
            raw = [raw]

        out: List[str] = []
        for x in raw:
            s = _object_to_string(x)
            if s:
                out.append(s)

        dedup: List[str] = []
        seen: set = set()
        for s in out:
            s = s.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            dedup.append(s)
            if len(dedup) >= 32:
                break
        return dedup

    def ask_json_array(self, prompt: str, system: Optional[str] = None) -> List[str]:
        """Legacy alias for ``ask_json_list``."""
        return self.ask_json_list(prompt, system)
