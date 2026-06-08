"""Initial warmup enqueue and query generation via agents."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def enqueue_all_queries(pipeline: OrchestratorPipeline) -> None:
    """Enqueue a subset of subtopics before dynamic feedback takes over."""
    critic_strategy = (pipeline.cfg.get("critic", {}) or {}).get("strategy", {}) or {}
    warmup_rounds = int(critic_strategy.get("warmup_rounds", 5) or 5)
    fb_cfg = pipeline.cfg.get("feedback_query", {}) or {}
    initial_limit = int(fb_cfg.get("initial_warmup_subtopics", 3) or 3)
    if pipeline._limit_subtopics:
        initial_limit = min(initial_limit, pipeline._limit_subtopics)
    selected = pipeline._all_subtopics[:initial_limit]
    if pipeline.critic and hasattr(pipeline.critic, "set_next_subtopics"):
        try:
            preview_subs = [
                s for _, s in pipeline._all_subtopics[: max(initial_limit, 1) * 2]
            ]
            pipeline.critic.set_next_subtopics(preview_subs)
            print(f"[Warmup] set_next_subtopics preview={preview_subs [:10 ]}")
        except Exception as e:
            print(f"[Warmup] set_next_subtopics error: {e }")
    for topic, sub in selected:
        key = (topic, sub)
        if key in pipeline._processed_subtopics:
            continue
        queries, is_strategy = await asyncio.to_thread(
            generate_queries, pipeline, topic, sub
        )
        if pipeline._limit_queries:
            queries = queries[: pipeline._limit_queries]
        added_any = False
        priority = 1 if is_strategy else 10
        for q in queries:
            if q in pipeline._generated_queries:
                continue
            while True:
                try:
                    pipeline.queries_queue.put_nowait(
                        (priority, -time.time(), (topic, sub, q))
                    )
                    pipeline._generated_queries.add(q)
                    added_any = True
                    break
                except asyncio.QueueFull:
                    await asyncio.sleep(0.05)
        if added_any:
            pipeline._processed_subtopics.add(key)
            pipeline._print_queue_head()
    print(
        f"[Warmup] Enqueued {len (pipeline ._processed_subtopics )} subtopics "
        f"(limit={initial_limit }, warmup_rounds={warmup_rounds })."
    )


def generate_queries(
    pipeline: OrchestratorPipeline, topic: str, subtopic: str
) -> Tuple[List[str], bool]:
    """Generate search queries for a topic/subtopic pair, optionally using critic strategy."""
    if (
        pipeline.critic
        and getattr(pipeline.critic, "should_use_query_strategy", None)
        and pipeline.critic.should_use_query_strategy()
    ):
        try:
            critic_queries = pipeline.critic.generate_queries_for(topic, subtopic)
            if critic_queries:
                print(
                    f"[Critic-Strategy] Generated {len (critic_queries )} queries for {topic }/{subtopic }"
                )
                return critic_queries, True
        except Exception as e:
            print(f"[Critic-Strategy] generate_queries_for error: {e }")

    base_prompt = f"Subtheme: {subtopic }\nGenerate <=3 Bing search queries highly relevant to '{subtopic }'."
    var_prompt = f"Generate <=5 diverse search queries for '{subtopic }'."
    base = call_agent_list(pipeline, "Retriever", base_prompt) or [subtopic]
    variants = call_agent_list(pipeline, "QueryGenerator", var_prompt)
    queries = [q for q in dict.fromkeys((base or []) + (variants or [])) if q] or [
        subtopic
    ]
    return queries, False


def call_agent_list(
    pipeline: OrchestratorPipeline, name: str, prompt: str
) -> List[str]:
    """Invoke a named agent with a prompt and return its list output."""
    agent = pipeline.agents.get(name)
    if not agent:
        return []
    try:
        values = agent.run_list(prompt) or []
        return [v.strip() for v in values if isinstance(v, str) and v.strip()]
    except Exception as e:
        print(f"[AgentError] {name } prompt='{prompt [:60 ]}' err={e }")
        return []
