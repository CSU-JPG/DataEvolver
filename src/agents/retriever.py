"""Retriever module agents: Retriever, QueryGenerator, QueryPlannerAgent."""

from __future__ import annotations

from typing import Dict

from src.agents.local_agents import Agent, _get_llm


def build_retriever_agents(llm, cir_llm=None) -> Dict[str, Agent]:
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
    }
