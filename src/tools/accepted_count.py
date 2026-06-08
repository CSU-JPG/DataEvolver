"""Scans accepted image directories and counts images per subtopic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

__all__ = ["count_images_in_accepted_dir", "save_stats_to_json"]


def count_images_in_accepted_dir(
    config_path: str,
) -> Tuple[List[Dict[str, Any]], str, int]:
    """Count images in the accepted directory per (topic, subtopic)."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    accepted_dir = config["paths"]["accepted_dir"]
    log_dir: str = config["paths"]["log_dir"]
    topics = config["topics"]
    target_total: int = config.get("dataset_num", 10000)

    stats: List[Dict[str, Any]] = []

    for topic in topics:
        topic_name: str = topic["name"]
        topic_path = Path(accepted_dir) / topic_name

        for subtopic_raw in topic["seed_queries"]:
            subtopic_name = subtopic_raw.replace("/", "_")
            subtopic_path = topic_path / subtopic_name

            if subtopic_path.exists():
                image_count = len([f for f in subtopic_path.iterdir() if f.is_file()])
                stats.append(
                    {
                        "topic": topic_name,
                        "subtopic": subtopic_name,
                        "image_count": image_count,
                    }
                )
            else:
                print(
                    f"[Warning] Subtopic directory does not exist: " f"{subtopic_path }"
                )

    return stats, log_dir, target_total


def save_stats_to_json(
    stats: List[Dict[str, Any]],
    log_dir: str,
) -> None:
    """Persist image count statistics to a JSON file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    output_file = log_path / "accepted_image_stats.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print(f"[Info] Statistics saved to {output_file }")
