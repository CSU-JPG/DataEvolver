"""Deterministic experience library maintainer (fallback when LLM agent is unavailable)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class HardcodedExperienceMaintainer:
    """Deterministic experience library with sliding window and top-k query ranking."""

    def update(
        self,
        experience: Dict[str, Any],
        adv_entry: Dict[str, Any],
        per_query_pass: Dict[str, Any],
        limits: Dict[str, int],
    ) -> Dict[str, Any]:
        exp = experience or {}
        recent: List[Dict[str, Any]] = list(exp.get("recent_advantages") or [])
        top_q: List[Dict[str, Any]] = list(exp.get("top_queries") or [])

        lib_adv = {
            "round": adv_entry.get("round"),
            "text": adv_entry.get("text"),
            "timestamp": adv_entry.get("timestamp"),
        }
        recent.append(lib_adv)
        semantic_window = int(limits.get("semantic_window", 5) or 5)
        recent = recent[-semantic_window:]

        query_stats: Dict[str, List[int]] = {}
        current_queries = set()

        for item in top_q:
            q = item.get("query")
            if not q:
                continue
            acc = int(item.get("accepted", 0))
            chk = int(item.get("checked", 0))
            query_stats[q] = [acc, chk]

        for q, val in (per_query_pass or {}).items():
            current_queries.add(q)
            if isinstance(val, (tuple, list)) and len(val) >= 2:
                new_acc, new_chk = val[0], val[1]
            else:
                new_acc, new_chk = 0, 0

            new_acc = max(int(new_acc or 0), 0)
            new_chk = max(int(new_chk or 0), 0)

            if q in query_stats:
                query_stats[q][0] += new_acc
                query_stats[q][1] += new_chk
            else:
                query_stats[q] = [new_acc, new_chk]

        ranked: List[Tuple[str, float, int, int, bool]] = []
        for q, (acc, chk) in query_stats.items():
            pr = (acc / chk) if chk > 0 else 0.0
            is_curr = q in current_queries
            ranked.append((q, pr, acc, chk, is_curr))

        ranked.sort(key=lambda x: (x[1], x[4], x[2]), reverse=True)
        top_k = int(limits.get("top_k", 10) or 10)
        selected = ranked[:top_k]

        new_top = []
        for q, pr, acc, chk, _ in selected:
            new_top.append(
                {
                    "query": q,
                    "pass_rate": round(pr, 4),
                    "accepted": acc,
                    "checked": chk,
                }
            )

        return {
            "recent_advantages": recent,
            "top_queries": new_top,
        }
