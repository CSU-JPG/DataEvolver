"""Prompt planning, image generation, and post-processing quality pipeline."""

from __future__ import annotations

import asyncio
import json
import pathlib
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.tools.qwen_image import generate_qwen_images
from src.utils.common import sha1_of_file
from src.worker.writer import save_generated

if TYPE_CHECKING:
    from src.pipeline import OrchestratorPipeline


async def plan_prompts_and_generate(pipeline: OrchestratorPipeline) -> None:
    """Run count analysis, plan prompts, generate images with
    optional critic-driven regeneration loop, and post-process results."""
    log_dir = pathlib.Path(pipeline.cfg.get("paths", {}).get("log_dir", "."))
    prompt_root = log_dir / "prompt_plans"
    prompt_root.mkdir(parents=True, exist_ok=True)

    need_map: Dict[Tuple[str, str], int] = {}
    try:
        from src.tools.count_analyse import run_count_analyse

        ca_res = run_count_analyse(
            config_path=pipeline.config_path,
            agents=pipeline.agents,
            use_llm=("CountAnalyse" in pipeline.agents),
            enforce_caps=False,
            fallback_strategy="uniform",
            show_stats=False,
            show_lines=False,
        )
        need_map = ca_res.get("final_need_map", {})
        print(f"[CountAnalyse] need_items={len(need_map)} gap={ca_res.get('gap')}")
    except Exception as e:
        print(f"[CountAnalyseError] {e}")

    if "PromptPlanner" in pipeline.agents and need_map:
        for (tp, sub), need in need_map.items():
            if need <= 0:
                continue
            pp_prompt = (
                "Based on the following information, generate high-quality English "
                "image generation prompts (only return a JSON array, list of strings):\n"
                f"Theme: {tp}\nSubtheme: {sub}\n"
                f"Number of prompts needed: {int(need)}\n"
                "Generation Requirements:\n"
                "1) The prompt must focus on the theme, listing key textual elements "
                "that must appear in the image.\n"
                "2) Text must be clear, high contrast, watermark-free.\n"
                "3) Layout natural & neat; avoid clutter.\n"
                "4) For complex scenes request simpler layout if needed.\n"
                "5) Output only the JSON array of prompt strings."
            )
            try:
                prompts = pipeline.agents["PromptPlanner"].run_list(pp_prompt) or []
                print(f"[PromptPlanner] {tp}/{sub} -> {len(prompts)} prompts")
                dst_dir = prompt_root / tp
                dst_dir.mkdir(parents=True, exist_ok=True)
                safe_sub = sub.replace("/", "_")
                out_fp = dst_dir / f"{safe_sub}.json"
                with open(out_fp, "w", encoding="utf-8") as f:
                    json.dump(
                        [
                            {"topic": tp, "subtopic": sub, "prompt": p}
                            for p in prompts
                            if isinstance(p, str) and p.strip()
                        ],
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[PromptPlanner] saved {out_fp}")
            except Exception as e:
                print(f"[PromptPlannerError] {tp}/{sub} {e}")

    prompt_files = list(prompt_root.glob("**/*.json"))
    current_prompts: List[Tuple[str, str, str]] = []  # [(topic, sub, prompt_text), ...]
    for pf in prompt_files:
        try:
            recs = json.load(open(pf, "r", encoding="utf-8"))
        except Exception:
            continue
        for rec in recs:
            ptxt = rec.get("prompt")
            if not ptxt:
                continue
            current_prompts.append((rec.get("topic"), rec.get("subtopic"), ptxt))

    if not current_prompts:
        print("[GenPhase] No prompts to generate, skipping generation phase.")
        return

    gen_cfg = pipeline.cfg.get("generation", {}) or {}
    max_rounds = max(int(gen_cfg.get("max_regeneration_rounds", 1)), 1)

    for round_idx in range(max_rounds):
        all_generated: List[Dict[str, Any]] = []
        gen_failures: List[Dict[str, Any]] = []
        for topic, sub, ptxt in current_prompts:
            try:
                res = generate_qwen_images(
                    topic=topic,
                    subtopic=sub,
                    prompts=[ptxt],
                    config_path=pipeline.config_path,
                    cfg=pipeline.cfg,
                )
                gen_results = res.get("results") if res else []
                for img_path, prompt in gen_results:
                    all_generated.append(
                        {
                            "image_path": img_path,
                            "topic": topic,
                            "subtopic": sub,
                            "prompt": prompt,
                        }
                    )
                pipeline.loop_totals["generated"] += len(gen_results or [])
                print(
                    f"[Generate] round={round_idx+1} {topic}/{sub} "
                    f"+{len(gen_results or [])} images"
                )
                for failed_prompt, error_msg in res.get("failures") or []:
                    gen_failures.append(
                        {
                            "image_path": "",
                            "topic": topic,
                            "subtopic": sub,
                            "prompt": failed_prompt,
                            "failure_stage": "generation",
                            "failure_reasons": [error_msg],
                            "quality_details": None,
                            "semantic_details": None,
                            "text_details": None,
                        }
                    )
            except Exception as e:
                print(f"[GenerateError] round={round_idx+1} {topic}/{sub} {e}")

        print(
            f"[GenPhase] round={round_idx+1}/{max_rounds} "
            f"generated={len(all_generated)}"
        )

        failures = await post_process_generated(
            pipeline, all_generated, return_failures=True
        )

        all_failures: List[Dict[str, Any]] = (failures or []) + gen_failures
        if gen_failures:
            print(
                f"[GenPhase] round={round_idx+1} "
                f"gen_failures={len(gen_failures)}"
            )

        if not all_failures or round_idx >= max_rounds - 1:
            break
        if not pipeline.critic or not pipeline.critic.prompt_critic_enabled:
            if gen_failures:
                current_prompts = [
                    (f["topic"], f["subtopic"], f["prompt"])
                    for f in gen_failures
                ]
                print(
                    f"[GenPhase] No critic; re-queued "
                    f"{len(gen_failures)} gen-failed prompts directly."
                )
                continue
            break

        optimized = pipeline.critic.optimize_failed_prompts(all_failures)
        if not optimized:
            print(
                "[GenPhase] No optimized prompts produced, stopping regeneration."
            )
            break

        print(
            f"[GenPhase] Optimized {len(optimized)} prompts for round "
            f"{round_idx+2}."
        )
        current_prompts = [
            (o["topic"], o["subtopic"], o["prompt"]) for o in optimized
        ]

    print(
        f"[GenPhase] total_generated={pipeline.loop_totals['generated']} "
        f"accepted={pipeline.loop_totals['generated_accepted']} "
        f"rejected={pipeline.loop_totals['generated_rejected']}"
    )


async def post_process_generated(
    pipeline: OrchestratorPipeline,
    images: List[Dict[str, Any]],
    return_failures: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Screen generated images through quality, semantic, and text-consistency checks."""
    loop = asyncio.get_running_loop()
    accepted = 0
    failures: List[Dict[str, Any]] = []
    for info in images:
        img_path = pathlib.Path(info.get("image_path"))
        if not img_path.exists():
            continue

        qscore = None
        sem_res = None
        tr = None

        quality_fail_reasons: List[str] = []
        try:
            if hasattr(pipeline.qa, "check_image_quality"):
                quality_fail_reasons = (
                    pipeline.qa.check_image_quality(str(img_path)) or []
                )
                if quality_fail_reasons:
                    reasons = (
                        quality_fail_reasons
                        if isinstance(quality_fail_reasons, list)
                        else ["quality_reject"]
                    )
                    pipeline.rej_tracker.add(str(img_path), reasons)
                    pipeline.loop_totals["generated_rejected"] += 1
                    if return_failures:
                        failures.append(
                            {
                                "image_path": str(img_path),
                                "topic": info.get("topic"),
                                "subtopic": info.get("subtopic"),
                                "prompt": info.get("prompt"),
                                "failure_stage": "quality",
                                "failure_reasons": reasons,
                                "quality_details": None,
                                "semantic_details": None,
                                "text_details": None,
                            }
                        )
                    continue
            else:
                qscore = await loop.run_in_executor(
                    None, pipeline.qa.score, str(img_path), {}
                )
                if hasattr(pipeline.qa, "check"):
                    ok, reasons = pipeline.qa.check(qscore, {})
                else:
                    ok = pipeline.qa.accept(qscore, {})
                    reasons = [] if ok else ["legacy_reject"]
                if not ok:
                    pipeline.rej_tracker.add(str(img_path), reasons)
                    pipeline.loop_totals["generated_rejected"] += 1
                    if return_failures:
                        failures.append(
                            {
                                "image_path": str(img_path),
                                "topic": info.get("topic"),
                                "subtopic": info.get("subtopic"),
                                "prompt": info.get("prompt"),
                                "failure_stage": "quality",
                                "failure_reasons": reasons,
                                "quality_details": qscore,
                                "semantic_details": None,
                                "text_details": None,
                            }
                        )
                    continue
        except Exception as e:
            print(f"[GenQualityError] {img_path.name} {e}")
            pipeline.rej_tracker.add(str(img_path), ["quality_error"])
            pipeline.loop_totals["generated_rejected"] += 1
            if return_failures:
                failures.append(
                    {
                        "image_path": str(img_path),
                        "topic": info.get("topic"),
                        "subtopic": info.get("subtopic"),
                        "prompt": info.get("prompt"),
                        "failure_stage": "quality",
                        "failure_reasons": ["quality_error"],
                        "quality_details": qscore,
                        "semantic_details": None,
                        "text_details": None,
                    }
                )
            continue

        try:
            sem_ok, sem_res = pipeline.semantic.check_relevance(
                str(img_path),
                topic=info.get("topic"),
                subtopic=info.get("subtopic"),
            )
            if not sem_ok:
                rejection_reasons = sem_res.get("rejection_reasons", ["semantic_error"])
                pipeline.rej_tracker.add(str(img_path), rejection_reasons)
                pipeline.loop_totals["generated_rejected"] += 1
                if return_failures:
                    failures.append(
                        {
                            "image_path": str(img_path),
                            "topic": info.get("topic"),
                            "subtopic": info.get("subtopic"),
                            "prompt": info.get("prompt"),
                            "failure_stage": "semantic",
                            "failure_reasons": rejection_reasons,
                            "quality_details": qscore,
                            "semantic_details": sem_res,
                            "text_details": None,
                        }
                    )
                continue
        except Exception as e:
            print(f"[GenSemanticError] {img_path.name} {e}")
            pipeline.rej_tracker.add(str(img_path), ["semantic_error"])
            pipeline.loop_totals["generated_rejected"] += 1
            if return_failures:
                failures.append(
                    {
                        "image_path": str(img_path),
                        "topic": info.get("topic"),
                        "subtopic": info.get("subtopic"),
                        "prompt": info.get("prompt"),
                        "failure_stage": "semantic",
                        "failure_reasons": ["semantic_error"],
                        "quality_details": qscore,
                        "semantic_details": None,
                        "text_details": None,
                    }
                )
            continue

        if pipeline.text_checker.is_available():
            try:
                rec = await loop.run_in_executor(
                    None, pipeline.ocr.run, str(img_path)
                )
                tr = pipeline.text_checker.check(
                    rec, info.get("topic"), info.get("subtopic")
                )
                if not tr.get("passed", True):
                    rejection_reasons = tr.get("rejection_reasons", ["text_error"])
                    pipeline.rej_tracker.add(str(img_path), rejection_reasons)
                    pipeline.loop_totals["generated_rejected"] += 1
                    if return_failures:
                        failures.append(
                            {
                                "image_path": str(img_path),
                                "topic": info.get("topic"),
                                "subtopic": info.get("subtopic"),
                                "prompt": info.get("prompt"),
                                "failure_stage": "text_consistency",
                                "failure_reasons": rejection_reasons,
                                "quality_details": qscore,
                                "semantic_details": sem_res,
                                "text_details": tr,
                            }
                        )
                    continue
            except Exception as e:
                print(f"[GenTextConsistencyError] {img_path.name} {e}")
                pipeline.rej_tracker.add(str(img_path), ["text_error"])
                pipeline.loop_totals["generated_rejected"] += 1
                if return_failures:
                    failures.append(
                        {
                            "image_path": str(img_path),
                            "topic": info.get("topic"),
                            "subtopic": info.get("subtopic"),
                            "prompt": info.get("prompt"),
                            "failure_stage": "text_consistency",
                            "failure_reasons": ["text_error"],
                            "quality_details": qscore,
                            "semantic_details": None,
                            "text_details": None,
                        }
                    )
                continue

        if pipeline.dedup.enabled:
            dedup_topic = (
                "" if pipeline._dedup_scope == "global"
                else info.get("topic", "")
            )
            try:
                is_dup, dup_of = await loop.run_in_executor(
                    None,
                    pipeline.dedup.check_and_add,
                    str(img_path),
                    dedup_topic,
                )
                if is_dup:
                    pipeline.loop_totals["generated_rejected"] += 1
                    pipeline.loop_totals["pHash_rejected"] += 1
                    pipeline.rej_tracker.add(
                        str(img_path), ["pHash_duplicate"]
                    )
                    if return_failures:
                        failures.append(
                            {
                                "image_path": str(img_path),
                                "topic": info.get("topic"),
                                "subtopic": info.get("subtopic"),
                                "prompt": info.get("prompt"),
                                "failure_stage": "pHash_dedup",
                                "failure_reasons": ["pHash_duplicate"],
                                "quality_details": qscore,
                                "semantic_details": sem_res,
                                "text_details": tr,
                            }
                        )
                    continue
            except Exception as e:
                print(f"[GenDedupError] {img_path.name} {e}")

        sha1 = sha1_of_file(img_path)
        meta = {
            "image_path": str(img_path),
            "topic": info.get("topic"),
            "subtopic": info.get("subtopic"),
            "prompt": info.get("prompt"),
            "is_generated": True,
            "content_sha1": sha1,
            "ts": time.time(),
            "source": "qwen_gen",
            "shard": pipeline.shard_id,
        }
        pipeline.writer.write_record(**meta)
        save_generated(pipeline, meta, img_path)
        pipeline.rej_tracker.add(str(img_path), [])
        accepted += 1

    pipeline.loop_totals["generated_accepted"] += accepted
    print(
        f"[GenPostProcess] accepted={accepted} "
        f"rejected={pipeline.loop_totals['generated_rejected']}"
    )

    if return_failures:
        return failures
    return None
