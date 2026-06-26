# DataEvolver: Self-Evolving Multi-Agent Data Construction for Text-Rich Image Generation

<p align="center">
  <a href="https://sgysy.github.io/">Siyu Yan</a><sup>1,2,*</sup>&nbsp;&nbsp;
  <a href="#">Yizhen Gao</a><sup>1,*</sup>&nbsp;&nbsp;
  <a href="#">Yilin Wang</a><sup>1</sup>&nbsp;&nbsp;
  <a href="#">Dongxing Mao</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://fingerrec.github.io/">Alex Jinpeng Wang</a><sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>Central South University&nbsp;&nbsp;&nbsp;
  <sup>2</sup>The Hong Kong University of Science and Technology<br>
  <sup>*</sup>Equal contribution&nbsp;&nbsp;&nbsp;
  <sup>†</sup>Corresponding author
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
  <a href="https://sgysy.github.io/dataevolver/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-orange" alt="Python 3.9+">
</p>

<p align="center">
  <a href="https://sgysy.github.io/dataevolver/">Project Page</a>
</p>

---

<p align="center">
  <img src="dataevolver-teaser.png" alt="DataEvolver overview" width="80%">
</p>

## Overview

**DataEvolver** is a self-evolving multi-agent data construction framework for text-rich image generation. Instead of following a static *crawl → filter → freeze* pipeline, DataEvolver treats rejected samples as feedback signals and uses them to improve subsequent construction rounds.

The key idea is simple: **rejected samples should not be discarded silently; they should guide the next round of retrieval, filtering, and generation.** Construction-time failures, including OCR errors, semantic mismatches, duplicates, and topic coverage gaps, are converted into actionable feedback for policy revision.

## Framework

DataEvolver is organized as a closed-loop construction process with four cooperative agents.

<table>
  <thead>
    <tr>
      <th width="24%">Agent</th>
      <th width="48%">Role</th>
      <th width="28%">Backend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="24%">🔍&nbsp;<b>Retriever</b></td>
      <td width="48%">Discovers candidate images with optimized search queries and updates query strategies using feedback from previous rounds.</td>
      <td width="28%"><code>llm.model</code> via Ollama, default <code>mistral:latest</code></td>
    </tr>
    <tr>
      <td width="24%">✅&nbsp;<b>Verifier</b></td>
      <td width="48%">Filters candidates through OCR, image quality, watermark, deduplication, CLIP semantic relevance, and text-consistency checks.</td>
      <td width="28%"><code>PaddleOCR</code>/<code>Tesseract</code> + <code>CLIP</code> + <code>Sentence-Transformers</code> + pHash</td>
    </tr>
    <tr>
      <td width="24%">📊&nbsp;<b>Critic</b></td>
      <td width="48%">Summarizes rejection patterns into semantic feedback, maintains an experience library, and revises construction policies.</td>
      <td width="28%"><code>llm.cir_model</code> via Ollama, default <code>qwen3.5:4b</code></td>
    </tr>
    <tr>
      <td width="24%">🎨&nbsp;<b>Generator</b></td>
      <td width="48%">Uses CountAnalyse and PromptPlanner to identify coverage gaps, write prompts, synthesize text-aware images, and re-filter generated samples.</td>
      <td width="28%"><code>mistral:latest</code> + <code>qwen3.5:4b</code> agents, <code>Qwen/Qwen-Image</code> T2I, optional <code>qwen3-vl:latest</code> annotation</td>
    </tr>
  </tbody>
</table>

```text
Round t:
  Retriever → Verifier → Accepted Dataset
                    ↓
              Rejected Samples
                    ↓
          Critic → Semantic Feedback → Policy Update
                    ↓
Round t+1:
  Revised Queries / Prompts / Thresholds → Retriever → ...
```

<p align="center">
  <img src="dataevolver-pipeline.png" alt="DataEvolver pipeline" width="80%">
</p>

## Highlights

- **Feedback-driven construction:** rejection causes are converted into semantic feedback rather than being discarded.
- **Closed-loop policy revision:** retrieval queries, generation prompts, and filtering thresholds are updated across construction rounds.
- **Targeted data completion:** the Generator synthesizes samples for under-represented text-rich image categories.
- **Traceable construction process:** accepted samples preserve query, caption, OCR, quality, semantic, and filtering metadata.

## Results

On **PixArt-α** at the **0.75M** data scale, DataEvolver improves OCR-F1 over the strongest matched-budget baseline.

| Evaluation Set | Relative OCR-F1 Improvement |
| --- | ---: |
| TextScenesHQ | +85.3% |
| LongTextBench | +35.3% |

Ablation studies show that both the **Critic** and the **Generator** contribute to the final performance, indicating that feedback-based policy revision and targeted completion are both necessary for effective text-rich data construction.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/CSU-JPG/DataEvolver.git
cd DataEvolver
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

For an existing environment, run the main entrypoint directly:

```bash
python main.py --config config.yaml --shard-id host-a --keep-rejects
```

`--shard-id` is required and should be unique for each concurrent machine or shard. Use `--keep-rejects` if you want rejected samples retained for diagnosis.

For a full CUDA/Ollama bootstrap, use:

```bash
bash run.sh
```

The script creates a Python 3.9 Conda environment, installs CUDA-oriented dependencies, starts eight local Ollama servers on ports `11437`-`11444`, pulls the default Ollama models, and launches `main.py`.

## Configuration

All major settings are controlled by `config.yaml`.

<details>
<summary><b>Paths</b> — data, logs, and output directories</summary>

```yaml
paths:
  images_crawled: "/path/to/crawled/images"
  images_generated: "/path/to/generated/images"
  ann_dir: "/path/to/annotations"
  log_dir: "/path/to/logs"
  accepted_dir: "/path/to/accepted"
  gen_dir: "/path/to/generated/raw"
  gen_accepted_dir: "/path/to/gen/accepted"
```

</details>

<details>
<summary><b>LLM & Agent Models</b> — Ollama-compatible local models</summary>

```yaml
llm:
  model: "mistral:latest"          # Retriever / QueryGenerator / QueryPlannerAgent / CountAnalyse
  cir_model: "qwen3.5:4b"          # Critic agents / PromptPlanner / prompt rewriting
```

All LLM agents are served through local Ollama instances. `src/agents/local_agents.py` round-robins requests over ports `11437`-`11444`.

</details>

<details>
<summary><b>Pipeline Control</b> — crawl/generate phases and feedback loops</summary>

```yaml
dataset_num: 1000000

generation:
  enabled: true
  max_regeneration_rounds: 3
  prompt_critic_enabled: true

critic:
  enabled: true
  strategy:
    enabled: true
    queries_enabled: true
    warmup_rounds: 3
  experience:
    use_agent: true

dynamic_query:
  enabled: true
  low_watermark: 30

feedback_query:
  enabled: true
  min_samples: 20
```

The main run first performs asynchronous retrieval and filtering, then enters the optional generation phase when `generation.enabled` is true.

</details>

<details>
<summary><b>Retrieval, OCR, and Filtering</b> — Bing, OCR, quality, semantic, and dedup checks</summary>

```yaml
bing:
  per_subtopic_images: 1500
  mkt: "auto"
  safesearch: "moderate"

ocr:
  engine: "paddle"
  lang: "ch"
  use_multi_gpu: true
  use_multiprocess: true

quality:
  min_ocr_coverage: 0.25
  min_legibility: 0.5
  min_side: 384
  max_side: 4096

semantic:
  enabled: true
  model_name: "openai/clip-vit-base-patch32"

text_consistency:
  enabled: true
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

hash_deduplication:
  enabled: true
  max_distance: 3
  scope: "global"
```

The Verifier combines OCR-derived quality signals, watermark detection, pHash deduplication, CLIP similarity, and sentence-transformer text consistency before writing accepted samples.

</details>

<details>
<summary><b>Generation and Annotation Models</b> — Qwen-Image and Qwen-VL</summary>

```yaml
qwen:
  true_cfg_scale: 4.5
  num_inference_steps: 25
  seed: 42

annotation:
  qwen_vl:
    enabled: true
    model: "qwen3-vl:latest"
    verbose: false
```

Image synthesis uses the Diffusers pipeline `Qwen/Qwen-Image` by default. Override it with the `QWEN_IMAGE_MODEL` environment variable if needed. The Qwen-VL setting is used for optional vision-language caption/annotation, not for text-to-image generation.

</details>

<details>
<summary><b>Topics & Seed Queries</b> — target domains for data construction</summary>

```yaml
topics:
  - name: Store_Signs_and_Shopfronts
    seed_queries:
      - "store signboard shop front"
      - "restaurant sign food street sign"
      - "Chinese shop plaque traditional signage"

  - name: Book_Covers
    seed_queries:
      - "book cover title author text"
      - "novel cover book jacket design"
```

Seed queries define the initial retrieval space. During construction, the Retriever and Critic refine these queries based on acceptance and rejection patterns.

</details>

## Quick Customization

1. Edit `paths` in `config.yaml`.
2. Define target `topics` and seed queries.
3. Choose Ollama-compatible LLM backends.
4. Start Ollama instances and make sure `mistral:latest`, `qwen3.5:4b`, and `qwen3-vl:latest` are available, or replace them in `config.yaml`.
5. Adjust OCR, semantic, text-consistency, deduplication, and quality thresholds.
6. Enable or disable generation with `generation.enabled`.
7. Run `python main.py --config config.yaml --shard-id <unique-shard-id>`.

## Citation

If you find DataEvolver useful, please cite:

```bibtex
@article{yan2026dataevolver,
  title   = {DataEvolver: Self-Evolving Multi-Agent Data Construction for Text-Rich Image Generation},
  author  = {Yan, Siyu and Gao, Yizhen and Wang, Yilin and Mao, Dongxing and Wang, Alex Jinpeng},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgments

This project builds on several open-source tools, including PaddleOCR, Tesseract OCR, Ollama, Hugging Face Diffusers, Qwen-Image, Qwen-VL, CLIP, and Sentence-Transformers.
