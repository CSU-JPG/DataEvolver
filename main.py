import asyncio
import argparse
from src.pipeline import OrchestratorPipeline
from src.utils.config import load_config
from src.agents.local_agents import build_agents

def main():
    parser = argparse.ArgumentParser(description="Run the OrchestratorPipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--shard-id", required=True, help="Unique shard ID for this pipeline run.")
    parser.add_argument("--keep-rejects", action="store_true", help="Keep rejected images in the output directory.")
    args = parser.parse_args()

    config_path = args.config
    cfg = load_config(config_path)

    llm_cfg = cfg.get("llm", {})
    agents = build_agents(
        model=llm_cfg.get("model", "mistral:latest"),
        cir_model=llm_cfg.get("cir_model", "qwen3.5:4b"),
    )

    pipeline = OrchestratorPipeline(
        cfg=cfg,
        agents=agents,
        shard_id=args.shard_id,
        keep_rejects=args.keep_rejects,
        config_path=config_path,
    )

    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()