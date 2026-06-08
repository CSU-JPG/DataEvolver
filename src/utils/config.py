"""YAML configuration loader with env-var injection and path normalisation."""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load and normalise a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for section in ("bing", "qwen"):
        if section in cfg and isinstance(cfg[section], dict):
            key_env = cfg[section].get("key_env")
            if key_env:
                cfg[section]["api_key"] = os.getenv(key_env)

    paths = cfg.get("paths", {})
    for name in ("root", "images_crawled", "images_generated", "ann_dir", "log_dir"):
        if name in paths:
            paths[name] = os.path.abspath(paths[name])
    cfg["paths"] = paths

    return cfg
