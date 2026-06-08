"""LLM agent definitions wrapping OllamaLLM with specific system prompts."""

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List

from src.llm.ollama_client import OllamaLLM

_PORTS = list(range(11437, 11445))
_port_cycle = itertools.cycle(_PORTS)


def _get_llm(model: str) -> OllamaLLM:
    """Return an OllamaLLM client round-robin'd across local instances."""
    port = next(_port_cycle)
    return OllamaLLM(model=model, base_url=f"http://127.0.0.1:{port }")


@dataclass
class Agent:
    """A named LLM agent with a fixed system prompt."""

    name: str
    system: str
    llm: OllamaLLM

    def run(self, prompt: str) -> str:
        """Send a prompt and return the raw response string."""
        return self.llm.ask(prompt, self.system)

    def run_list(self, prompt: str) -> List[str]:
        """Send a prompt and parse the response as a JSON string list."""
        return self.llm.ask_json_list(prompt, self.system)

    def run_template(self, context: Dict[str, Any]) -> str:
        """Format the system prompt with *context* and send it as the user prompt."""
        prompt = self.system.format(**context)
        return self.llm.ask(prompt)


def build_agents(model: str = "mistral:latest", cir_model: str = "qwen3.5:4b") -> Dict[str, Agent]:
    """Construct and return all active pipeline agents."""
    llm = _get_llm(model)
    cir_llm = _get_llm(cir_model)

    return {
        "Retriever": Agent(
            "Retriever",
            (
                "You are a search strategy generator. Given a subtopic, construct "
                "**up to 3** Bing image queries (return as a JSON array). Strong constraints:\n"
                "1) Each query **must** contain keywords that retrieve 'images with readable text':\n"
                "   label, packaging label, nutrition facts, ingredients, barcode, instruction, "
                "manual, leaflet, receipt, certificate, signage, timetable, menu, UI screenshot;\n"
                "   Chinese synonyms can include: 标签、外包装文字、营养成分表、配料表、条形码、"
                "说明书、票据、证书、指示牌、时间表、菜单、界面截图;\n"
                "2) It's recommended to add sites or data sources to improve retrieval accuracy: "
                "site:wikimedia.org OR site:flickr.com OR site:openfoodfacts.org;\n"
                "3) Emphasize clarity and richness with terms like 'high resolution', '4k', "
                "'detailed', 'rich texture', 'HD', 'with readable text'; avoid 'simple', "
                "'minimalist', 'plain background';\n"
                "4) Return only the JSON array (string list), **without any explanations**."
            ),
            llm,
        ),
        "QueryGenerator": Agent(
            "QueryGenerator",
            (
                "You are a diversified search strategy generator. Given a subtopic, "
                "generate **up to 5** distinct search queries (return as a JSON array). "
                "Strong constraints:\n"
                "1) Each query **must** explicitly include a 'text-related' description: "
                "'with readable text', 'packaging label', 'nutrition facts', 'ingredients list',\n"
                "   'back of packaging text', or Chinese equivalents: '含清晰文字', '外包装 文字', "
                "'营养成分表', '配料表', '包装 背面 文字 清晰';\n"
                "2) Emphasize 'no watermark/clean layout/high contrast/high resolution/rich details'; "
                "avoid 'simple', 'minimalist', 'plain background'; mix Chinese and English if necessary;\n"
                "3) Return only the JSON array (string list), **without any explanations**."
            ),
            llm,
        ),
        "PromptPlanner": Agent(
            "PromptPlanner",
            (
                "You are an image generation prompt engineer. Your goal is to generate "
                "high-quality image generation prompts for a specified **theme-subtheme**, "
                "in the required number.\n"
                "Strictly follow these requirements and return only the JSON array "
                "(string list), **without any explanations**:\n"
                "1) The prompts should combine the theme and subtheme, highlighting key "
                "elements of the text scene (keywords, text content, visual scene, "
                "quality features);\n"
                "2) Each prompt must ensure that the generated image **includes visible, "
                "OCR-readable text** (e.g., billboard, packaging, menu, poster, book page, etc.);\n"
                "3) Each prompt should generate a concise statement, clear and direct, "
                "without additional explanation;\n"
                "4) Output format: ['prompt1', 'prompt2', 'prompt3', ...]."
            ),
            cir_llm,
        ),
        "CountAnalyse": Agent(
            "CountAnalyse",
            (
                "You are an image dataset quantity balancing analyst. Your task is to "
                "plan the number of images needed for each **theme-subtheme** based on "
                "the total target and current data distribution.\n"
                "Strictly follow the requirements and return only the JSON array "
                "(string list), **without any explanations**:\n"
                "1) Only output subtopics that need additional images; for subtopics "
                "that have already reached or exceeded a reasonable number, return 0 "
                "or omit them;\n"
                "2) Do not output negative numbers or decimals;\n"
                "3) Output should be a string array, with each element formatted as "
                "'subtopic name: need X more images';\n"
                "4) Output format: JSON array, without any explanations, e.g., "
                "['TopicA SubtopicA: 50', 'TopicA SubtopicB: 30', 'SubtopicC: 20']."
            ),
            llm,
        ),
        "SemanticAdvantageAgent": Agent(
            "SemanticAdvantageAgent",
            (
                "### INPUT DATA (Base your analysis STRICTLY on these metrics)\n"
                "1. Current Round Rejections: {current_rejection_counts}\n"
                "2. Previous Round Rejections: {prev_rejection_counts}\n"
                "3. Keyword Performance (Accepted vs Rejected): {keyword_analysis}\n\n"
                "### TASK\n"
                "You are a critical analyst. Produce a highly abstract, concise "
                "semantic advantage summary (A_text) of 4-6 sentences.\n\n"
                "1. **Keyword Optimization:**\n"
                "   - Identify keywords from {keyword_analysis} that appear mostly in "
                "REJECTED items -> Suggest WEAKENING/REMOVING them.\n"
                "   - Identify keywords that appear in ACCEPTED items -> Suggest "
                "STRENGTHENING them.\n\n"
                "### OUTPUT FORMAT (Strict Constraints):\n"
                "1. Output exactly ONE continuous natural-language paragraph (4-6 sentences).\n"
                "2. NO bullet points, NO headings, NO Markdown formatting, NO JSON.\n"
                "3. **Required Style:**\n"
                "   'Blurry sample ratio increased (Prev: 5 -> Curr: 20) while text "
                "density issues improved; Suggestions: remove watermark-related keywords "
                "due to high rejection association.'\n"
                "4. If previous data is missing/empty, base analysis solely on current "
                "high-rejection areas."
            ),
            cir_llm,
        ),
        "StrategyPlannerAgent": Agent(
            "StrategyPlannerAgent",
            (
                "You are a strategy optimizer for a data construction system. You "
                "update thresholds strictly and only according to the explicit "
                "recommendations in the Semantic Advantage A_text.\n\n"
                "Input: Semantic Advantage A_text, Experience Library E "
                "(recent_advantages: window text, top_queries: best queries), "
                "and Threshold Snapshot Th.\n\n"
                "You must strictly follow these rules:\n"
                "1) Output exactly ONE JSON object with one key:\n"
                '    {{"thresholds": {{ ... }}}}\n'
                "2) thresholds:\n"
                "   - MUST be an object.\n"
                "   - MUST contain ONLY the keys you actually modify.\n"
                "   - Allowed keys (numeric only):\n"
                "     [Quality] min_ocr_coverage (0~1) | min_legibility (0~1) | "
                "min_sharpness (>0) | min_char_h_px (>0) | max_char_h_cv (0~1) | "
                "max_line_angle_std (>0) | min_centrality (0~1) | max_border_ratio (0~1) | "
                "min_text_density (>0) | min_char_density (>0) | max_clutter (0~1) | "
                "min_contrast (0~1)\n"
                "     [Semantic] min_topic_similarity (0~1) | min_subtopic_similarity (0~1) | "
                "min_combined_similarity (0~1) | min_similarity (0~1)\n"
                "     [Text Consistency] text_min_topic_similarity (0~1) | "
                "text_min_subtopic_similarity (0~1) | text_min_combined_similarity (0~1)\n"
                "3) **You may ONLY modify a threshold if A_text explicitly recommends "
                "adjusting it.**\n"
                "   - If A_text does NOT mention a threshold or does NOT indicate a "
                "direction -> DO NOT modify it.\n"
                "   - Do NOT infer, guess, generalize, broaden, or invent new adjustments.\n"
                "4) Direction rules (mandatory):\n"
                '   - If A_text says "tighten", "increase", "raise", "higher", '
                '"more strict" -> increase the threshold value.\n'
                '   - If A_text says "loosen", "decrease", "lower", "less strict" -> '
                "decrease the threshold value.\n"
                "   - Amount of adjustment: small and stable steps (e.g., +/-0.02 for "
                "[0~1] ranges; +/-0.5 for px-based thresholds).\n"
                "5) NEVER:\n"
                "   - Modify thresholds not explicitly named or clearly implied by A_text.\n"
                '   - Produce keys other than "thresholds".\n'
                "   - Output explanations, comments, or text outside the final JSON.\n"
                "6) If A_text contains NO actionable threshold suggestions:\n"
                '   -> Output: {{"thresholds": {{}}}}\n'
                "7) Example of valid output (for reference only, do not copy values):\n"
                '   -> {{"thresholds":{{"quality":{{"min_sharpness":60,"min_contrast":0.1,'
                '"min_text_density":7.0,"max_line_angle_std":250.0,"max_clutter":0.15}},'
                '"semantic":{{"min_combined_similarity":0.8}}}}}}\n'
                "Your final answer must be a single JSON object only."
            ),
            cir_llm,
        ),
        "QueryPlannerAgent": Agent(
            "QueryPlannerAgent",
            (
                "### INPUT (you can ONLY use this)\n"
                "pending_subtopics: {pending_subtopics}\n"
                "top_queries_examples: {top_queries}\n"
                "recent_a_text (Strategy Feedback): {recent_a_text}\n\n"
                "TASK: Generate 30-45 Bing image queries for the pending subtopics.\n\n"
                "STRICT RULES (break any = failure):\n"
                "1. EVERY query MUST contain at least ONE text keyword from: readable text, "
                "clear text, with text, label, annotation, dimensions, legend\n"
                "2. EVERY query MUST contain at least ONE quality keyword from: high resolution, "
                "4k, HD, clean layout, no watermark, high contrast\n"
                "3. **Apply the strategy in 'recent_a_text':** If it suggests avoiding specific "
                "patterns or strengthening certain terms, adjust your queries accordingly.\n"
                "4. You are FORBIDDEN to copy any domain-specific noun (nutrition, packaging, "
                "food, etc.) from top_queries_examples.\n"
                "5. Use the sentence structure style of top_queries_examples, but adapt to "
                "the specific domain of pending_subtopics.\n"
                "6. Output ONLY this JSON, nothing else:\n"
                '{{"queries": ["query1", "query2", ...]}}'
            ),
            llm,
        ),
        "ExperienceLibrarianAgent": Agent(
            "ExperienceLibrarianAgent",
            (
                "You are the **Experience Library Manager** for DataEvolver. "
                "Your ONLY job is to maintain a high-quality, traceable experience "
                "library E that continuously improves data generation quality.\n\n"
                "### INPUT (you MUST base ALL decisions ONLY on this input)\n"
                "Current A_text (latest semantic advantage): {current_a_text}\n"
                "Recent advantages (last K rounds, chronological, K={K}): "
                "{recent_advantages}\n"
                "Top historical queries (with reason): {top_queries}\n\n"
                "### TASK\n"
                "Decide exactly ONE operation: Add | Delete | Keep\n"
                "- Add    -> only if current_a_text is genuinely novel AND significantly "
                "stronger than existing ones\n"
                "- Delete -> only if you detect clear duplication or quality degradation "
                "in top_queries\n"
                "- Keep   -> default safe choice when uncertain\n\n"
                "### STRICT OUTPUT RULES (violating any rule = failure)\n"
                "1. You are FORBIDDEN to invent, imagine, or recall any A_text/query/reason "
                "not explicitly present in the input above.\n"
                "2. `updated_recent_advantages` MUST be exactly the last K items "
                "(chronological, newest last), length <= {K}. Never reorder, never drop "
                "without explicit Delete.\n"
                "3. Every entry in `updated_top_queries` MUST contain BOTH `query` "
                "(exact or polished) and `reason` (original or refined).\n"
                "4. When Modify/Delete, you MUST keep the original query string traceable "
                "(e.g., minor rephrasing only).\n"
                "5. Output MUST be a SINGLE valid JSON object and NOTHING else "
                "(no markdown, no explanations, no extra fields).\n\n"
                "Return ONLY this JSON:\n"
                "{{\n"
                '  "operation": "Add|Delete|Keep",\n'
                '  "updated_recent_advantages": [list of strings, length <= {K}, '
                "chronological],\n"
                '  "updated_top_queries": [{{"query": "...", "reason": "..."}}, ...]\n'
                "}}"
            ),
            cir_llm,
        ),
    }
