"""Dataset size analysis and intelligent distribution with LLM-driven allocation."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.local_agents import build_agents
from src.tools.accepted_count import count_images_in_accepted_dir
from src.utils.config import load_config

__all__ = [
    "parse_count_analyse_output",
    "scale_count_needs",
    "fallback_distribution",
    "build_prompt",
    "run_count_analyse",
]


def parse_count_analyse_output(
    lines: List[str],
    topic_names: List[str],
) -> Dict[Tuple[str, str], int]:
    """Parse LLM output lines into a {(topic, subtopic): need} mapping."""
    mapping: Dict[Tuple[str, str], int] = {}
    if not lines:
        return mapping

    topic_sorted = sorted([t for t in topic_names if t], key=len, reverse=True)
    num_pat = re.compile(r"(-?\d+)")

    for raw in lines:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue

        pos = s.rfind(":")
        if pos == -1:
            continue

        head = s[:pos].strip()
        tail = s[pos + 1 :].strip()

        m = num_pat.search(tail)
        if not m:
            continue
        val = int(m.group(1))
        need = max(0, val)

        topic = None
        sub = None
        for tp in topic_sorted:
            if head.startswith(tp):
                rest = head[len(tp) :].strip()
                if rest:
                    topic, sub = tp, rest
                    break

        if topic is None:
            parts = head.split(maxsplit=1)
            if len(parts) == 2:
                topic, sub = parts[0].strip(), parts[1].strip()

        if not topic or not sub:
            continue

        mapping[(topic, sub)] = need

    return mapping


def scale_count_needs(
    raw_need: Dict[Tuple[str, str], int],
    current_counts: Dict[Tuple[str, str], int],
    gap: int,
    desired_caps: Optional[Dict[Tuple[str, str], int]] = None,
) -> Dict[Tuple[str, str], int]:
    """Scale / expand raw allocations to fill gap while respecting caps."""
    if gap <= 0 or not raw_need:
        return {k: 0 for k in raw_need}

    caps_remaining: Dict[Tuple[str, str], Optional[int]] = {}
    for k in raw_need:
        if desired_caps and k in desired_caps:
            cap_total = desired_caps[k]
            cur = current_counts.get(k, 0)
            caps_remaining[k] = max(0, cap_total - cur)
        else:
            caps_remaining[k] = None

    clipped: Dict[Tuple[str, str], int] = {}
    for k, v in raw_need.items():
        rem = caps_remaining[k]
        clipped[k] = min(v, rem) if rem is not None else v

    raw_total = sum(clipped.values())
    if raw_total == 0:
        return {k: 0 for k in clipped}

    if raw_total >= gap:
        ratio = gap / raw_total
        prelim: Dict[Tuple[str, str], int] = {}
        fracs: List[Tuple[float, Tuple[str, str]]] = []
        used = 0
        for k, v in clipped.items():
            scaled = v * ratio
            base = math.floor(scaled)
            rem = caps_remaining[k]
            if rem is not None:
                base = min(base, rem)
            prelim[k] = base
            used += base
            fracs.append((scaled - base, k))

        leftover = gap - used
        fracs.sort(reverse=True)
        for frac, k in fracs:
            if leftover <= 0:
                break
            rem = caps_remaining[k]
            if rem is not None and prelim[k] >= rem:
                continue
            prelim[k] += 1
            leftover -= 1
        return prelim

    deficit = gap - raw_total
    final_map = dict(clipped)

    addable_capacity: Dict[Tuple[str, str], Optional[int]] = {}
    total_weight = 0
    for k, base in final_map.items():
        rem = caps_remaining[k]
        if rem is None:
            addable_capacity[k] = None
            total_weight += base if base > 0 else 1
        else:
            extra = max(0, rem - base)
            if extra > 0:
                addable_capacity[k] = extra
                total_weight += base if base > 0 else 1
            else:
                addable_capacity[k] = 0

    if total_weight == 0:
        return final_map

    prelim_extra: Dict[Tuple[str, str], int] = {k: 0 for k in final_map}
    fracs: List[Tuple[float, Tuple[str, str]]] = []
    used_extra = 0
    for k, base in final_map.items():
        cap_extra = addable_capacity[k]
        if cap_extra == 0:
            continue
        weight = base if base > 0 else 1
        raw_alloc = deficit * (weight / total_weight)
        alloc_floor = math.floor(raw_alloc)
        if cap_extra is not None:
            alloc_floor = min(alloc_floor, cap_extra)
        if alloc_floor > 0:
            prelim_extra[k] += alloc_floor
            used_extra += alloc_floor
        fracs.append((raw_alloc - alloc_floor, k))

    remaining = deficit - used_extra
    fracs.sort(reverse=True)
    for frac, k in fracs:
        if remaining <= 0:
            break
        cap_extra = addable_capacity[k]
        if cap_extra is not None and prelim_extra[k] >= cap_extra:
            continue
        prelim_extra[k] += 1
        remaining -= 1

    for k, extra in prelim_extra.items():
        final_map[k] += extra

    if remaining > 0:
        expandable: List[Tuple[str, str]] = []
        for k in final_map:
            cap_extra = addable_capacity[k]
            if cap_extra is None:
                expandable.append(k)
            else:
                already = final_map[k] - clipped[k]
                if already < cap_extra:
                    expandable.append(k)

        idx = 0
        n = len(expandable)
        while remaining > 0 and n > 0:
            k = expandable[idx % n]
            cap_extra = addable_capacity[k]
            if cap_extra is not None:
                already = final_map[k] - clipped[k]
                if already >= cap_extra:
                    idx += 1
                    if idx >= n * 2:
                        break
                    continue
            final_map[k] += 1
            remaining -= 1
            idx += 1

    return final_map


def fallback_distribution(
    current_counts: Dict[Tuple[str, str], int],
    gap: int,
    strategy: str = "uniform",
) -> Dict[Tuple[str, str], int]:
    """Simple distribution when LLM output is unavailable."""
    if gap <= 0 or not current_counts:
        return {k: 0 for k in current_counts}

    keys = list(current_counts.keys())
    if not keys:
        return {}

    n = len(keys)

    if strategy == "inverse":
        sorted_keys = sorted(keys, key=lambda k: current_counts[k])
        weights = [1.0 / (i + 1) for i, _ in enumerate(sorted_keys)]
        total_w = sum(weights)
        alloc: Dict[Tuple[str, str], int] = {}
        remain = gap
        for k, w in zip(sorted_keys, weights):
            v = int(gap * (w / total_w))
            v = min(v, remain)
            alloc[k] = v
            remain -= v
        if remain > 0:
            for k in sorted_keys:
                if remain <= 0:
                    break
                alloc[k] += 1
                remain -= 1
        return alloc

    base = gap // n
    leftover = gap - base * n
    alloc = {k: base for k in keys}
    for k in keys:
        if leftover <= 0:
            break
        alloc[k] += 1
        leftover -= 1
    return alloc


def build_prompt(
    target_total: int,
    simplified_stats: List[Dict[str, Any]],
    gap: int,
) -> str:
    """Construct the LLM prompt for count analysis."""
    return (
        f"Target total: {target_total }\n"
        f"Current stats (subtopic → image_count): "
        f"{json .dumps (simplified_stats ,ensure_ascii =False )}\n"
        f"Remaining gap: {gap }\n"
        "Please propose additional images per subtopic. Rules:\n"
        f"1) Sum of allocations ≤ {gap }.\n"
        "2) No negative or fractional numbers.\n"
        "3) Skip subtopics that are already full.\n"
        "4) Use only existing (topic, subtopic) pairs; do not invent.\n"
        "5) Format: JSON array of strings like "
        "'Topic Subtopic: N'.\n"
        "Example: ['Signage exit sign: 120', 'Packaging label: 40']\n"
        "Output the JSON array only."
    )


def run_count_analyse(
    config_path: str,
    agents: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
    enforce_caps: bool = False,
    fallback_strategy: str = "uniform",
    show_stats: bool = False,
    show_lines: bool = False,
    precomputed_stats: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the full count-analysis pipeline."""
    cfg = load_config(config_path)

    if precomputed_stats is not None:
        stats = precomputed_stats
        target_total = int(cfg.get("dataset_num", 10000))
        _log_dir: str = cfg.get("paths", {}).get("log_dir", "")
    else:
        stats, _log_dir, target_total = count_images_in_accepted_dir(config_path)

    simplified_stats = [
        {
            "topic": s["topic"],
            "subtopic": s["subtopic"],
            "image_count": int(s["image_count"]),
        }
        for s in stats
    ]

    if show_stats:
        print("=== Raw Stats ===")
        for r in simplified_stats:
            print(r)

    current_counts = {
        (r["topic"], r["subtopic"]): r["image_count"] for r in simplified_stats
    }
    current_total = sum(current_counts.values())
    gap = max(0, int(target_total) - current_total)

    topics_cfg = cfg.get("topics") or []
    topic_names = [t.get("name") for t in topics_cfg if t.get("name")]

    desired_caps: Dict[Tuple[str, str], int] = {}
    if enforce_caps:
        for t in topics_cfg:
            tp = t.get("name")
            for sub in t.get("subtopics") or []:
                subn = sub.get("name")
                tgt = sub.get("target")
                if tp and subn and isinstance(tgt, int):
                    desired_caps[(tp, subn)] = tgt
        if desired_caps:
            print(f"[Info] Read {len (desired_caps )} target cap(s).")

    if agents is None and use_llm:
        model_name = cfg.get("llm", {}).get("model", "mistral:latest")
        try:
            agents = build_agents(model_name)
        except Exception as e:
            print(f"[Warn] LLM agent build failed, using fallback: {e }")
            agents = {}
            use_llm = False

    prompt = build_prompt(int(target_total), simplified_stats, gap)

    raw_need_map: Dict[Tuple[str, str], int] = {}
    if not use_llm or gap == 0 or agents is None or "CountAnalyse" not in agents:
        pass
    else:
        try:
            raw_lines = agents["CountAnalyse"].run_list(prompt) or []
        except Exception as e:
            print(f"[Error] LLM call failed: {e }")
            raw_lines = []

        if not raw_lines:
            print("(empty)")
        else:
            if show_lines:
                for ln in raw_lines:
                    print(ln)

        raw_need_map = parse_count_analyse_output(raw_lines, topic_names)

    if sum(raw_need_map.values()) == 0 and gap > 0:
        print("[Info] raw_need_map is all zeros; using fallback " "distribution.")
        raw_need_map = fallback_distribution(
            current_counts, gap, strategy=fallback_strategy
        )

    final_need_map = scale_count_needs(
        raw_need_map,
        current_counts,
        gap,
        desired_caps if desired_caps else None,
    )
    final_total = sum(final_need_map.values())

    return {
        "target_total": int(target_total),
        "current_total": int(current_total),
        "gap": int(gap),
        "simplified_stats": simplified_stats,
        "raw_need_map": raw_need_map,
        "final_need_map": final_need_map,
        "accepted_dir": cfg.get("paths", {}).get("accepted_dir"),
    }
