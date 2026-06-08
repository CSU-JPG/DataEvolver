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
    """Run count analysis, plan prompts, generate images, and post-process results."""
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
        print(f"[CountAnalyse] need_items={len (need_map )} gap={ca_res .get ('gap')}")
    except Exception as e:
        print(f"[CountAnalyseError] {e }")

    if "PromptPlanner" in pipeline.agents and need_map:
        for (tp, sub), need in need_map.items():
            if need <= 0:
                continue
            pp_prompt = (
                "Based on the following information, generate high-quality English "
                "image generation prompts (only return a JSON array, list of strings):\n"
                f"Theme: {tp }\nSubtheme: {sub }\n"
                f"Number of prompts needed: {int (need )}\n"
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
                print(f"[PromptPlanner] {tp }/{sub } -> {len (prompts )} prompts")
                dst_dir = prompt_root / tp
                dst_dir.mkdir(parents=True, exist_ok=True)
                safe_sub = sub.replace("/", "_")
                out_fp = dst_dir / f"{safe_sub }.json"
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
                print(f"[PromptPlanner] saved {out_fp }")
            except Exception as e:
                print(f"[PromptPlannerError] {tp }/{sub } {e }")

    prompt_files = list(prompt_root.glob("**/*.json"))
    all_generated: List[Dict[str, Any]] = []
    for pf in prompt_files:
        try:
            recs = json.load(open(pf, "r", encoding="utf-8"))
        except Exception:
            continue
        for rec in recs:
            ptxt = rec.get("prompt")
            if not ptxt:
                continue
            topic = rec.get("topic")
            sub = rec.get("subtopic")
            try:
                res = generate_qwen_images(
                    topic=topic,
                    subtopic=sub,
                    prompts=[ptxt],
                    config_path=pipeline.config_path,
                )
                gen_results = res.get("results") if res else []
                if gen_results:
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
                print(f"[Generate] {topic }/{sub } +{len (gen_results or [])} images")
            except Exception as e:
                print(f"[GenerateError] {topic }/{sub } {e }")
    print(f"[GenPhase] total_generated={len (all_generated )}")
    await post_process_generated(pipeline, all_generated)


async def post_process_generated(
    pipeline: OrchestratorPipeline, images: List[Dict[str, Any]]
) -> None:
    """Screen generated images through quality, semantic, and text-consistency checks."""
    loop = asyncio.get_running_loop()
    accepted = 0
    for info in images:
        img_path = pathlib.Path(info.get("image_path"))
        if not img_path.exists():
            continue

        try:
            if hasattr(pipeline.qa, "check_image_quality"):
                quality_fail_reasons = pipeline.qa.check_image_quality(str(img_path))
                if quality_fail_reasons:
                    pipeline.rej_tracker.add(
                        str(img_path),
                        (
                            quality_fail_reasons
                            if isinstance(quality_fail_reasons, list)
                            else ["quality_reject"]
                        ),
                    )
                    pipeline.loop_totals["generated_rejected"] += 1
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
                    continue
        except Exception as e:
            print(f"[GenQualityError] {img_path .name } {e }")
            pipeline.rej_tracker.add(str(img_path), ["quality_error"])
            pipeline.loop_totals["generated_rejected"] += 1
            continue

        try:
            sem_ok, sem_res = pipeline.semantic.check_relevance(
                str(img_path),
                topic=info.get("topic"),
                subtopic=info.get("subtopic"),
            )
            if not sem_ok:
                pipeline.rej_tracker.add(
                    str(img_path),
                    sem_res.get("rejection_reasons", ["semantic_error"]),
                )
                pipeline.loop_totals["generated_rejected"] += 1
                continue
        except Exception as e:
            print(f"[GenSemanticError] {img_path .name } {e }")
            pipeline.rej_tracker.add(str(img_path), ["semantic_error"])
            pipeline.loop_totals["generated_rejected"] += 1
            continue

        if pipeline.text_checker.is_available():
            try:
                rec = await loop.run_in_executor(None, pipeline.ocr.run, str(img_path))
                tr = pipeline.text_checker.check(
                    rec, info.get("topic"), info.get("subtopic")
                )
                if not tr.get("passed", True):
                    pipeline.rej_tracker.add(
                        str(img_path),
                        tr.get("rejection_reasons", ["text_error"]),
                    )
                    pipeline.loop_totals["generated_rejected"] += 1
                    continue
            except Exception as e:
                print(f"[GenTextConsistencyError] {img_path .name } {e }")
                pipeline.rej_tracker.add(str(img_path), ["text_error"])
                pipeline.loop_totals["generated_rejected"] += 1
                continue

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
        f"[GenPostProcess] accepted={accepted } "
        f"rejected={pipeline .loop_totals ['generated_rejected']}"
    )
