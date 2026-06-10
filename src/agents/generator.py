"""Generator module agents: PromptPlanner, CountAnalyse."""

from __future__ import annotations

from typing import Dict

from src.agents.local_agents import Agent, _get_llm


def build_generator_agents(llm, cir_llm) -> Dict[str, Agent]:
    return {
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
    }
