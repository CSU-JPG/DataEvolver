"""Critic module agents: SemanticAdvantageAgent, ExperienceLibrarianAgent, PromptCriticAgent."""

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
                "You are a prompt rewriter for a text-to-image pipeline that generates images "
                "containing visible, readable text (signs, labels, documents, posters, etc.).\n\n"
                "A generated image failed. Rewrite the prompt to fix the failure.\n\n"

                "## INPUT\n"
                "Original Prompt: {original_prompt}\n"
                "Topic: {topic}\n"
                "Subtopic: {subtopic}\n"
                "Failure Stage: {failure_stage}\n"
                "Failure Reasons: {failure_reasons}\n"
                "Quality Metrics: {quality_details}\n"
                "Semantic Scores: {semantic_details}\n"
                "Text Consistency Scores: {text_details}\n\n"

                "## RULES\n"
                "Read the Failure Stage and Failure Reasons, then apply the matching rule:\n\n"
                "- generation failure → Shorten to under 15 words. Keep subject only. "
                "No quality words.\n\n"
                "- quality failure → Keep original subject. "
                "Add: 'crisp text, sharp focus, high contrast, clean white background'.\n\n"
                "- semantic failure → Start the prompt with the exact topic and subtopic as the subject. "
                "Describe the key visual element in one short sentence.\n\n"
                "- text_consistency failure → Add this phrase exactly: "
                "\"with clearly visible printed text showing '[subtopic]'\".\n\n"

                "## ALWAYS REQUIRED (apply to every rewrite)\n"
                "The image must contain readable text. Always include ONE of these in the final prompt:\n"
                "'clear printed text', 'legible label', 'readable sign', or 'visible text overlay'.\n\n"
                "## OUTPUT\n"
                "Return only this JSON. No explanation. No extra text:\n"
                "{\"optimized_prompt\": \"your rewritten prompt\"}"
            ),
            cir_llm,
        ),
    }
