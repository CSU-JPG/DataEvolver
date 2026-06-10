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
    return OllamaLLM(model=model, base_url=f"http://127.0.0.1:{port}")


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


def build_agents(
    model: str = "mistral:latest", cir_model: str = "qwen3.5:4b"
) -> Dict[str, Agent]:
    """Construct and return all active pipeline agents."""
    llm = _get_llm(model)
    cir_llm = _get_llm(cir_model)

    from src.agents.retriever import build_retriever_agents
    from src.agents.critic import build_critic_agents
    from src.agents.generator import build_generator_agents

    agents: Dict[str, Agent] = {}
    agents.update(build_retriever_agents(llm, cir_llm))
    agents.update(build_critic_agents(llm, cir_llm))
    agents.update(build_generator_agents(llm, cir_llm))
    return agents
