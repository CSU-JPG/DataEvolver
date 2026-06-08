"""Image captioning and key-text extraction via Ollama-hosted Qwen-VL."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langdetect import detect as lang_detect

from .captioner import build_layout_caption
from .vl_client import VLMClient

_QWEN_SYSTEM = "You are a helpful vision-language assistant."
_QWEN_CAPTION = (
    "You are a visual captioning assistant that writes detailed, natural "
    "descriptions suitable for text-to-image generation."
)


def truncate_text(text: str, max_len: int) -> str:
    """Truncate text at sentence boundaries to fit within *max_len* chars."""
    sentences = re.split(r"([.!?])", text)
    result = ""
    for sentence in sentences:
        if len(result + sentence) <= max_len:
            result += sentence
        else:
            break
    return result.strip()


def _template_caption(topic: Optional[str], subtopic: Optional[str]) -> str:
    """Fallback caption when VLM is unavailable."""
    t = (topic or "").replace("_", " ")
    s = (subtopic or "").replace("_", " ")
    core = f"{t } {s }".strip() or "text-centric poster"
    return f"A concise description of a {core } focusing on layout and typography."


def generate_key_text(
    image_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None,
    subtopic: Optional[str] = None,
    ocr_rec: Optional[Dict[str, Any]] = None,
) -> str:
    """Extract key texts from an image using OCR results and VLM analysis."""
    try:
        vl = VLMClient(cfg.get("annotation", {}) or {})
        if not getattr(vl, "enabled", False):
            print("[VLMClient] unavailable")
            return _template_caption(topic, subtopic)

        user = f"""<image>
You are a text extraction specialist for enhancing image captions in text-to-image generation tasks. Your goal is to extract all textual elements from the OCR recognition results that can contribute to generating a comprehensive and relevant image caption, while referencing the OCR confidence (such as recognition accuracy and completeness). Prioritize segments that are relevant to the theme '{topic }' and subtheme '{subtopic }', such as texts aligning with documents, motivation, or business contexts. Include all information-rich texts, directly quoting phrases (using double quotes) where possible.

Task: Analyze the OCR results, extract all relevant texts based on the theme and subtopic, and output a complete list in the specified format. Focus on:
- All prominent and supporting texts (such as titles, slogans, key points, labels, body text), directly quoting full phrases or sentences.
- Relevance to the theme '{topic }' and subtheme '{subtopic }': Include texts that align with or enhance themes like documents, motivation, or business.
- High information density: Capture all key nouns, concepts, and details (e.g., 'revenue growth', 'team motto', 'sustainability workshop', full paragraphs if relevant).

Do NOT:
- Include irrelevant details (such as watermarks).
- Add non-OCR content or generate new information.

Output Format:
Start with "Extracted texts for caption:", followed by all relevant entries, each formatted as [attribute: OCR recognized content]. Attributes include title, label, slogan, body, etc., inferred based on context. Example: Extracted texts for caption: [title: Discrete Mathematics Chapter 1] [slogan: "Tomorrow Will Be Better"] [label: "Revenue: 15% Growth"] [body: "Our team achieved significant milestones in Q4..."] [...]
In the process, evaluate OCR confidence: If recognition is incomplete or erroneous, note it briefly in the entry (e.g., [title: "Discrete Mathe~atics"] (low OCR confidence) ) and prioritize the most reliable texts based on context, but aim to include all usable segments.
OCR result: {ocr_rec }
Now, analyze the provided image and OCR results to generate a full text extraction summary for caption enhancement.
"""

        ans = vl.generate_text_only(user, model_name="qwen3:8b")
        res = (ans or "").strip() or ""
        res = " ".join(res.split())
        print(f"[Qwen] KEY-TEXT res: {res }")
        return res
    except Exception:
        return ""


def extract_important_texts(
    ocr_rec: dict,
    topic: str,
    subtopic: str,
    max_items: int = 15,
    max_chars: int = 1200,
) -> List[str]:
    """Heuristic extraction of important OCR texts by bounding-box height."""
    texts = ocr_rec.get("texts", []) or []
    boxes = ocr_rec.get("boxes", []) or []

    sized = []
    for t, box in zip(texts, boxes):
        if not t.strip():
            continue
        h = 20
        try:
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                if isinstance(box[0], (list, tuple)):
                    h = abs(box[3][1] - box[0][1])
        except Exception:
            h = 20
        sized.append((t.strip(), h))

    sized.sort(key=lambda x: -x[1])
    seen = set()
    out = []
    for t, _ in sized:
        key = t.lower()[:30]
        if key in seen:
            continue
        seen.add(key)
        out.append(f'"{t }"')
        if len(out) >= max_items:
            break
    return out


def generate_image_caption(
    image_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None,
    subtopic: Optional[str] = None,
    ocr_rec: Optional[Dict[str, Any]] = None,
    q: Optional[Dict[str, Any]] = None,
    max_chars: int = 400,
) -> str:
    """Generate a concise, layout-aware image caption for text-to-image tasks."""
    try:
        vl = VLMClient(cfg.get("annotation", {}) or {})
        if not getattr(vl, "enabled", False):
            print("[VLMClient] unavailable")
            return _template_caption(topic, subtopic)

        layout_hint = ""
        if ocr_rec and q:
            try:
                layout_hint = build_layout_caption(
                    ocr_rec, q, topic or "document", subtopic or ""
                )
            except Exception:
                layout_hint = ""

        user = f"""<image>
You are a visual description assistant specialized in generating concise, natural captions for text-to-image tasks.
Task: Produce ONE brief paragraph (1–2 sentences) focusing on:
- primary subject(s) and materials/textures (e.g., "spalted maple top", "matte chrome bridge");
- color and tonal scheme;
- camera viewpoint and composition (e.g., "centered", "top-down");
- spatial relationships and key UI elements only if relevant (e.g., "flanked by a vertical panel");
- important textual elements (e.g., "vehicle parts catalog", "bold labels", "clear text annotations").

Style & constraints:
- Tone: neutral, descriptive, and concise.
- Structure: 1–2 sentences, focusing on key details.
- Length: max {max_chars } characters, maintain conciseness while capturing key elements.

Good example:
"A realistic render of a motivational poster with a sunrise scene in warm orange tones, centered composition featuring the integrated slogan 'Tomorrow Will Be Better' at the bottom, flanked by bullet points on community events."

Bad examples:
- "A cheap poster with unknown elements."  # avoid vagueness
"""
        if layout_hint:
            user += f"\nLayout context (from OCR analysis): {layout_hint }\n"
        user += "\nOutput should sound natural, detailed, and coherent. Now produce the caption."

        ans = vl.generate(image_path, user)
        cap = (ans or "").strip() or _template_caption(topic, subtopic)
        cap = " ".join(cap.split())
        if len(cap) > max_chars:
            cap = truncate_text(cap, max_chars)
        print(f"[Qwen-VL] caption: {cap }")
        return cap
    except Exception:
        return _template_caption(topic, subtopic)


def generate_image_caption_simple(
    image_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    max_chars: int = 400,
) -> str:
    """Generate a rich text-centric caption purely from image content."""
    _fallback = (
        "A text-rich image with structured layout and clear typographic elements."
    )
    try:
        vl = VLMClient(cfg.get("annotation", {}) or {})
        if not getattr(vl, "enabled", False):
            print("[VLMClient] unavailable")
            return _fallback

        user = f"""<image>
You are an expert visual captioning assistant specialized in generating high-quality, detailed captions for text-rich images, intended for training text-to-image generation models.

Task: Carefully analyze the image and produce ONE coherent paragraph (2–3 sentences) that comprehensively describes:

1. **Overall layout and composition**: Describe the spatial structure (e.g., "a vertically stacked layout", "a two-column grid", "a centered hero section with surrounding callouts").
2. **Text content and typography**: Identify and describe prominent textual elements — titles, headings, slogans, body text, labels, captions — including their visual style (e.g., "bold serif headline", "small sans-serif body text", "handwritten annotation").
3. **Visual design elements**: Colors, backgrounds, decorative elements, icons, illustrations, photos, or diagrams present alongside the text.
4. **Tone and purpose**: Infer the document/image type and purpose (e.g., "a motivational poster", "a product catalog page", "an educational slide", "a business report cover", "a food packaging label").

Style & constraints:
- Write in fluent, natural English.
- Be specific and descriptive — avoid vague phrases like "some text" or "various elements".
- Directly quote or paraphrase key visible text when it significantly characterizes the image (e.g., 'featuring the headline "Think Different"').
- Do NOT invent text that is not visible in the image.
- Length: max {max_chars } characters. Prioritize completeness and informativeness within this limit.

Good examples:
- "A minimalist A4 poster with a deep navy background featuring the bold white headline 'BELIEVE IN YOURSELF' centered at the top, followed by three motivational bullet points in light gray sans-serif font, and a subtle sunrise illustration at the bottom."
- "A two-column product catalog page on a white background, with a high-resolution photo of a running shoe on the left and detailed specifications in small black serif text on the right, accented by orange price labels and a 'BUY NOW' call-to-action button."
- "An educational slide with a pale yellow background showing a flowchart titled 'Photosynthesis Process', with labeled arrows connecting colorful icon nodes, and a short explanatory paragraph in 12pt Times New Roman at the bottom."

Bad examples (avoid):
- "An image with text and some design elements." (too vague)
- "A poster about something." (no detail)

Now produce the caption for the provided image.
"""

        ans = vl.generate(image_path, user)
        cap = (ans or "").strip() or _fallback
        cap = " ".join(cap.split())
        if len(cap) > max_chars:
            cap = truncate_text(cap, max_chars)
        print(f"[Qwen-VL][Simple] caption: {cap }")
        return cap
    except Exception as e:
        print(f"[Qwen-VL][Simple] exception: {e }")
        return _fallback


def _join_texts(texts: List[str]) -> str:
    """Concatenate and collapse whitespace across text strings."""
    xs = []
    for t in texts or []:
        t = str(t).strip()
        if not t:
            continue
        t = re.sub(r"\s+", " ", t)
        xs.append(t)
    return " ".join(xs)[:4000]


def _detect_lang(text: str) -> str:
    """Detect the language of *text* using langdetect."""
    try:
        return lang_detect(text) if text.strip() else "unknown"
    except Exception:
        return "unknown"


def _pick_lines(texts: List[str], keys: List[str], max_lines: int = 12) -> List[str]:
    """Select up to *max_lines* text lines whose content matches any of *keys*."""
    out = []
    if not keys:
        return out
    pat = re.compile("|".join([re.escape(k) for k in keys]), re.I)
    for t in texts or []:
        if pat.search(t or ""):
            out.append((t or "").strip())
        if len(out) >= max_lines:
            break
    return out
