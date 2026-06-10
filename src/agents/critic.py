"""Critic module agents: SemanticAdvantageAgent, StrategyPlannerAgent,
ExperienceLibrarianAgent, PromptCriticAgent."""

from __future__ import annotations

from typing import Dict

from src.agents.local_agents import Agent, _get_llm


def build_critic_agents(llm, cir_llm) -> Dict[str, Agent]:
    return {
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
        "PromptCriticAgent": Agent(
            "PromptCriticAgent",
            (
                "You are a **prompt optimization critic** for an image generation "
                "pipeline. Your task: analyze WHY a generated image failed quality "
                "checks, then rewrite the prompt to fix those specific issues.\n\n"
                "### INPUT\n"
                "Original Prompt: {original_prompt}\n"
                "Topic: {topic}\n"
                "Subtopic: {subtopic}\n"
                "Failure Stage: {failure_stage}\n"
                "Failure Reasons: {failure_reasons}\n"
                "Quality Metrics: {quality_details}\n"
                "Semantic Scores: {semantic_details}\n"
                "Text Consistency Scores: {text_details}\n\n"
                "### DIAGNOSIS + REPAIR RULES\n"
                "1) **quality failure** (low sharpness/contrast/resolution/legibility):\n"
                "   → Add: 'high resolution', 'sharp', 'high contrast', '4K', "
                "'clean layout', 'large readable text', 'crisp typography'.\n"
                "2) **semantic failure** (image content off-topic):\n"
                "   → Strengthen topic/subtopic keywords; explicitly describe the "
                "expected scene and visual elements.\n"
                "3) **text_consistency failure** (OCR text doesn't match topic):\n"
                "   → Explicitly name the specific text content that MUST appear in "
                "the image (e.g., 'the sign must display the word CAFE').\n"
                "4) **generation failure** (model inference error, OOM, or pipeline "
                "crash during generation):\n"
                "   → Simplify the prompt: use a shorter sentence, fewer detail "
                "requirements, avoid resolution/size specifiers that may trigger OOM. "
                "Keep core topic keywords intact.\n"
                "5) Always keep the prompt concise, focused, and in English.\n"
                "6) Do NOT introduce unrelated styles or content.\n\n"
                "### OUTPUT\n"
                "Return ONLY a single JSON object, nothing else:\n"
                '{{"optimized_prompt": "your improved prompt here"}}'
            ),
            cir_llm,
        ),
    }
