"""Qwen-Image batch generation with multi-GPU support and OOM recovery."""

from __future__ import annotations

import gc
import hashlib
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

try:
    from diffusers import DiffusionPipeline as _DP

    _DIFFUSERS_IMPORT_ERR: Optional[Exception] = None
except Exception as _e:
    _DP = None
    _DIFFUSERS_IMPORT_ERR = _e

_PIPELINE: Any = None

NEGATIVE_PROMPT_FOR_CLEAR_TEXT = (
    "blurry, out of focus, motion blur, low resolution, low quality, "
    "unreadable text, garbled text, distorted letters, wrong spelling, "
    "compression artifacts, jpeg artifacts, watermark, logo, signature, "
    "overexposed, underexposed, strange colors, unnatural artifacts, "
    "extra limbs, deformed"
)

__all__ = ["generate_qwen_images"]


def _select_dtype() -> torch.dtype:
    """Choose the best available dtype for the current GPU."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _is_oom_error(e: Exception) -> bool:
    """Return True for any flavour of CUDA out-of-memory error."""
    if torch.cuda.is_available() and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return True
    return False


def _free_cuda_memory() -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _try_enable_memory_optimizations(pipe: Any) -> None:
    """Apply all available memory-saving knobs on the pipeline."""
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)
        print("[QwenImage][OOM] enable_attention_slicing=1")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        print("[QwenImage][OOM] enable_vae_slicing")
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
        print("[QwenImage][OOM] enable_vae_tiling")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[QwenImage][OOM] enable_xformers_memory_efficient_attention")
    except Exception:
        pass


def _load_pipeline(
    model_name: str = "Qwen/Qwen-Image",
) -> Any:
    """Load the Qwen-Image diffusion pipeline with multi-GPU strategy fallback."""
    if _DP is None:
        raise RuntimeError(
            f"diffusers not installed or failed to load: " f"{_DIFFUSERS_IMPORT_ERR }"
        )

    dtype = _select_dtype()
    _free_cuda_memory()
    t0 = time.time()

    strategy = os.getenv("QWEN_DEVICE_MAP_STRATEGY", "sequential")
    if strategy not in ("sequential", "balanced", "cuda"):
        print(
            f"[QwenImage][Warn] Invalid strategy '{strategy }'; "
            f"falling back to sequential"
        )
        strategy = "sequential"

    accel_ok = False
    if torch.cuda.is_available():
        try:
            import accelerate

            accel_ok = True
        except Exception:
            accel_ok = False

    pipe = None

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        vram_info = [
            f"GPU{i }: "
            f"{torch .cuda .get_device_properties (i ).total_memory //1024 **3 }GB"
            for i in range(gpu_count)
        ]
        print(f"[QwenImage] Detected {gpu_count } GPU(s): " f"{', '.join (vram_info )}")

        if gpu_count > 1 and accel_ok:
            print(
                f"[QwenImage] Attempting multi-GPU load "
                f"device_map='{strategy }' ..."
            )
            try:
                pipe = _DP.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=strategy,
                    low_cpu_mem_usage=True,
                )
                print(
                    f"[QwenImage] Multi-GPU load OK "
                    f"strategy={strategy } GPUs={gpu_count }"
                )
            except Exception as e:
                print(
                    f"[QwenImage][Warn] Multi-GPU load failed "
                    f"({strategy }), trying alternate: {e }"
                )
                pipe = None

            if pipe is None and strategy == "sequential":
                try:
                    pipe = _DP.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        device_map="balanced",
                        low_cpu_mem_usage=True,
                    )
                    print(
                        "[QwenImage] Multi-GPU load OK " "strategy=balanced (fallback)"
                    )
                except Exception as e2:
                    print(
                        f"[QwenImage][Warn] balanced also failed, "
                        f"falling back to single GPU: {e2 }"
                    )
                    pipe = None

        if pipe is None:
            print("[QwenImage] Single-GPU load (cuda:0)...")
            pipe = _DP.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            pipe.to("cuda:0")
            if gpu_count > 1 and not accel_ok:
                print(
                    "[QwenImage][Info] Multiple GPUs available but "
                    "accelerate not installed, using cuda:0 only.\n"
                    "  Install: pip install accelerate"
                )
    else:
        pipe = _DP.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        print("[QwenImage] CPU load (debug mode)")

    _try_enable_memory_optimizations(pipe)

    try:
        _name = pipe.config.get("_name_or_path", "Unknown")
        print(f"[QwenImage] Model: {_name }")
    except Exception:
        pass

    pipe.set_progress_bar_config(disable=True)
    print(
        f"[QwenImage] Pipeline loaded dtype={dtype } "
        f"elapsed={time .time ()-t0 :.1f}s"
    )
    return pipe


def _get_pipeline() -> Any:
    """Return the global pipeline singleton, loading on first access."""
    global _PIPELINE
    if _PIPELINE is None:
        model_name = os.getenv("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
        _PIPELINE = _load_pipeline(model_name)
    return _PIPELINE


def _infer_with_oom_retry(pipe: Any, params: Dict[str, Any]) -> Any:
    """Run inference with up to 3 OOM-retry attempts."""
    last_exc: Optional[Exception] = None

    for attempt in range(3):
        try:
            with torch.inference_mode():
                return pipe(**params)

        except Exception as e:
            if not _is_oom_error(e):
                raise

            last_exc = e
            print(f"[QwenImage][OOM] Attempt {attempt +1 }: " f"{str (e )[:120 ]}")
            _free_cuda_memory()

            if attempt == 1:
                print("[QwenImage][OOM] Forcing finest slicing ...")
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing(1)
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()

    print("[QwenImage][OOM] All 3 retries exhausted; giving up.")
    raise last_exc


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration, ensuring src/ is on sys.path."""
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in os.sys.path:
        os.sys.path.insert(0, str(root))
    from src.utils.config import load_config

    return load_config(config_path)


def _resolve_gen_dir(cfg: Dict[str, Any]) -> pathlib.Path:
    """Resolve the output directory for generated images."""
    paths = cfg.get("paths", {}) or {}
    base = paths.get("gen_dir") or paths.get("images_generated") or "./generated"
    p = pathlib.Path(base).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_qwen_images(
    topic: str,
    subtopic: str,
    prompts: List[str],
    config_path: str = "config.yaml",
    strict: bool = False,
) -> Dict[str, Any]:
    """Generate images using Qwen-Image for a (topic, subtopic) pair."""
    start_time = time.time()

    prompts = [p for p in prompts if p and p.strip()]
    if not prompts:
        return {
            "summary": {
                "topic": topic,
                "subtopic": subtopic,
                "total_prompts": 0,
                "success": 0,
                "failed": 0,
            },
            "results": [],
            "failures": [],
        }

    cfg = _load_config(config_path)
    qcfg = cfg.get("qwen", {}) or {}
    gen_dir = _resolve_gen_dir(cfg)

    negative_prompt = NEGATIVE_PROMPT_FOR_CLEAR_TEXT
    true_cfg_scale = float(qcfg.get("true_cfg_scale", 4.0))
    steps = int(qcfg.get("num_inference_steps", 15))
    base_seed = qcfg.get("seed")
    if isinstance(base_seed, str) and base_seed.isdigit():
        base_seed = int(base_seed)
    elif not isinstance(base_seed, int):
        base_seed = None

    safe_sub = subtopic.replace("/", "_")
    out_dir = gen_dir / topic / safe_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = _get_pipeline()

    results: List[Tuple[str, str]] = []
    failures: List[Tuple[str, str]] = []
    per_prompt_timings: List[float] = []

    for idx, prompt in enumerate(prompts):
        g = None
        if base_seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            g = torch.Generator(device=device).manual_seed(base_seed + idx)

        params: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "generator": g,
        }

        try:
            t_start = time.time()
            output = _infer_with_oom_retry(pipe, params)
            img = output.images[0]

            run_tag = os.getenv("QWEN_IMAGE_WORKER_ID", f"pid{os .getpid ()}")
            safe_tag = "".join(
                c if (c.isalnum() or c in "-_") else "_" for c in str(run_tag)
            )
            fp = out_dir / f"gen_{safe_tag }_{idx :05d}.png"
            if fp.exists():
                short = hashlib.sha1(
                    f"{safe_tag }_{idx }_{time .time_ns ()}".encode()
                ).hexdigest()[:8]
                fp = out_dir / f"gen_{safe_tag }_{idx :05d}_{short }.png"

            img.save(fp, format="PNG", compress_level=0)
            elapsed = time.time() - t_start
            per_prompt_timings.append(elapsed)
            results.append((str(fp), prompt))
            print(
                f"[QwenImage][Generated] idx={idx } " f"time={elapsed :.2f}s path={fp }"
            )

        except Exception as e:
            print(
                f"[QwenImage][Error] idx={idx } " f"prompt='{prompt [:60 ]}' err={e }"
            )
            failures.append((prompt, str(e)))
            if strict:
                raise
        finally:
            _free_cuda_memory()

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_images = len(results)
    images_per_sec = (
        total_images / elapsed_time if elapsed_time > 0 and total_images > 0 else 0.0
    )
    avg_time_per_image = elapsed_time / total_images if total_images > 0 else 0.0

    summary = {
        "topic": topic,
        "subtopic": subtopic,
        "total_prompts": len(prompts),
        "success": len(results),
        "failed": len(failures),
        "output_dir": str(out_dir),
        "steps": steps,
        "true_cfg_scale": true_cfg_scale,
        "seed_base": base_seed,
        "elapsed_time": elapsed_time,
        "images_per_sec": images_per_sec,
        "avg_time_per_image_sec": avg_time_per_image,
        "per_prompt_timings_sec": per_prompt_timings,
    }
    print(
        f"[QwenImage][Summary] topic={topic } subtopic={subtopic } "
        f"prompts={len (prompts )} success={len (results )} "
        f"failed={len (failures )} dir={out_dir } "
        f"time={elapsed_time :.2f}s "
        f"throughput={images_per_sec :.2f} img/s "
        f"avg={avg_time_per_image :.2f}s/img"
    )
    if failures:
        print(
            f"[QwenImage][Failures] {len (failures )} entries; "
            f"first error: {failures [0 ][1 ][:160 ]}"
        )

    return {"summary": summary, "results": results, "failures": failures}


def _cli() -> None:
    """Command-line entry point."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Qwen Image Batch Generator")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--subtopic", required=True)
    ap.add_argument("--prompt", action="append", help="Repeatable")
    ap.add_argument("--prompts-file", help="Text file, one prompt per line")
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    prompts: List[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompts_file and pathlib.Path(args.prompts_file).exists():
        prompts.extend(
            [
                line.strip()
                for line in open(args.prompts_file, encoding="utf-8")
                if line.strip()
            ]
        )
    if not prompts:
        raise SystemExit("Need at least one --prompt or --prompts-file")
    if args.repeat and len(prompts) == 1:
        prompts = [prompts[0]] * args.repeat

    res = generate_qwen_images(
        topic=args.topic,
        subtopic=args.subtopic,
        prompts=prompts,
        config_path=args.config,
        strict=args.strict,
    )
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
