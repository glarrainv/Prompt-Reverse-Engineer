---
title: Prompt Reverse Engineer
emoji: 🔁
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
suggested_hardware: zero-a10g
models:
  - Qwen/Qwen3-VL-8B-Instruct
  - black-forest-labs/FLUX.2-klein-4B
  - black-forest-labs/FLUX.1-schnell
  - facebook/dinov2-base
library_name: transformers
tags:
  - image-to-prompt
  - computer-vision
  - captioning
  - qwen
  - flux
  - dino
  - vqa
  - zero-gpu
short_description: Turn any image into a verified text-to-image prompt.
---

# Prompt Reverse Engineer

Turn an image into a verified text-to-image prompt. A closed-loop pipeline extracts a constrained scene graph, grounds every fact on the original, renders with a proxy model, scores each predicate with VQAScore and layout with DINOv2, and rewrites only what fails — until both gates clear or iterations run out.

[![License](https://img.shields.io/github/license/glarrainv/Prompt-Reverse-Engineer?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/glarrainv/Prompt-Reverse-Engineer?style=for-the-badge)](https://github.com/glarrainv/Prompt-Reverse-Engineer/stargazers)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/glarr/Prompt-Reverse-Engineer)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.31-orange?style=for-the-badge&logo=gradio)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.22-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/docs/transformers)
[![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green?style=for-the-badge)](LICENSE)

---

## Overview

**Prompt Reverse Engineer** inverts text-to-image generation: instead of prompting a model and hoping the output matches your vision, you provide an image and get back a prompt whose claims are checked, one by one, against a proxy render of that prompt. The pipeline runs four stages in a loop:

1. **Extract** a Davidsonian Scene Graph (DSG) — entities, attributes, relations, and medium/optics on a 3×3 grid
2. **Ground** every predicate on the original image; drop or re-localize facts the scorer rejects
3. **Render** the linearized prompt with a fast proxy model (FLUX.2-klein-4B by default; FLUX.1-schnell is supported)
4. **Verify** with VQAScore (predicate agreement) and DINOv2 (layout similarity); **rewrite** only failed facts

**What makes this different:**

- Grounding on the original: facts the scorer rejects on the original image are removed (or, for grid positions, re-localized) before rendering, and rewrites cannot introduce era/style buzzwords that the grounded prompt does not already contain. Grounding is only as good as the scorer: with the local backend the same model extracts and scores, so shared blind spots remain (use `SCORER_BACKEND=api` for an independent check)
- Targeted diff rewrites: only failed predicates are edited, preserving the rest
- Dual-gate stopping: both predicate agreement (S_VQA) and layout similarity (S_DINO) must clear configurable thresholds; if they never do, the best iteration is returned and the report says `max_iters`
- Two deployment pathways: local GPUs (Qwen3-VL + FLUX + DINOv2) or API-only (OpenRouter + FAL + gpt-4o)
- Gradio UI for Hugging Face Spaces (ZeroGPU-ready) and Python CLI for batch runs
- Full JSON report with every iteration, scores, predicates, and proxy renders

---

## Quickstart

### Option A: Hugging Face Space (zero setup)

[Open the Space](https://huggingface.co/spaces/glarr/Prompt-Reverse-Engineer) → upload an image → adjust gates if needed → click **Invert image**. The verified prompt, proxy renders, per-fact scores, and a downloadable JSON report stream in live.

### Option B: Local (Windows / macOS / Linux)

```bash
# 1. Clone
git clone https://github.com/glarrainv/Prompt-Reverse-Engineer.git
cd Prompt-Reverse-Engineer

# 2. Create venv and install (requirements pin torch 2.13.0; on Windows, install the CUDA build
#    of torch 2.13.0 from pytorch.org first, since the PyPI wheel there is CPU-only)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Nothing is gated with the default models. Only if you set
#    PROXY_MODEL_ID=black-forest-labs/FLUX.1-schnell: accept its license on the model page, then log in
hf auth login  # or export HF_TOKEN=hf_...

# 4. Run the Gradio UI
python app.py
# → opens at http://localhost:7860
```

> **GPU memory (local pathway).** All three models stay loaded. In bf16 the default weights are ~33.9 GB (Qwen3-VL-8B 17.5 GB, FLUX.2-klein-4B 16.0 GB, DINOv2-B 0.35 GB), plus activations, so a 48 GB GPU is the practical minimum. Ways to fit smaller cards, or to use the larger proxy (estimates from weight sizes, not benchmarks):
>
> | Setting | Weights on GPU | Fits |
> | --- | --- | --- |
> | default (klein-4B, bf16) | ~33.9 GB | 48 GB GPU |
> | `PROXY_FP8=1` | ~30 GB | 40 GB GPU |
> | `PROXY_CPU_OFFLOAD=1` | ~18 GB + the active FLUX component (≤ 8 GB) | 24 GB GPU; slower |
> | `PROXY_BACKEND=fal` | ~18 GB | 24 GB GPU; FLUX runs on fal.ai (paid, needs `FAL_KEY`) |
> | `PROXY_MODEL_ID=black-forest-labs/FLUX.1-schnell` | ~51.6 GB | 80 GB GPU |
> | schnell + `PROXY_FP8=1` | ~40 GB | 48 GB GPU |
>
> `PROXY_FP8=1` stores the FLUX transformer in float8 and computes in bf16; proxy renders change slightly.

### Choosing the proxy model

The proxy renderer is the forward model the loop optimizes the prompt against, so the verified prompt is specific to it. `PROXY_MODEL_ID` selects it; the pipeline class, step count and guidance scale are read from the checkpoint:

| `PROXY_MODEL_ID` | Weights | Steps / guidance | License and access | Needs |
| --- | --- | --- | --- | --- |
| `black-forest-labs/FLUX.2-klein-4B` (default) | 16.0 GB (transformer 7.75, Qwen3 text encoder 8.0) | 4 / 1.0 | Apache-2.0, ungated | diffusers ≥ 0.37 |
| `black-forest-labs/FLUX.1-schnell` | 33.7 GB (transformer 23.8, T5-XXL 9.5) | 4 / 0.0 | Apache-2.0, gated (accept on the model page) | `HF_TOKEN` |

- Only step-distilled checkpoints work. `FLUX.2-klein-base-*` are rejected at startup (their `model_index.json` lacks `is_distilled: true`), and other pipelines raise an error listing the supported ones.
- With the default klein-4B no model in the stack is gated, so no `HF_TOKEN` is needed; FLUX.1-schnell requires one.
- Every run report records the models and proxy settings used under `"models"`. Keep one proxy for a whole stimulus bank: scores and prompts are not comparable across proxies.
- klein-4B is verified structurally (pipeline selection, 4 steps, guidance 1.0, fp8 casting, deterministic seeds) on a small random-weight pipeline; render quality on your stimuli has not been measured. Compare S_VQA and S_DINO against schnell on a few images before relying on either in a study.

### Option C: Python CLI (batch processing)

```bash
# Invert a folder of images
python cli.py invert --input stimuli/ --out runs/ --max-iters 5 --vqa-gate 0.90 --dino-gate 0.80

# Select DINO medoid render per condition folder
python cli.py medoid --input renders/ --out medoids.json --key patch --copy-to medoids/
```

| Flag | Effect | Default |
|------|--------|---------|
| `--input PATH` | Image file or directory | required |
| `--out DIR` | Output root for runs/ | `runs/` |
| `--max-iters N` | Max rewrite iterations | `5` |
| `--vqa-gate F` | S_VQA convergence gate | `0.90` |
| `--dino-gate F` | S_DINO convergence gate (0 = disable) | `0.80` |
| `--fail-threshold F` | Predicate failure threshold for rewrite | `0.75` |
| `--w-vqa F` | Weight on S_VQA for best-iteration selection | `0.7` |
| `--seed N` | Base seed (proxy seed = base + t) | `42` |
| `--no-ground` | Skip grounding on original (not recommended) | off |

### Option D: Docker (GPU)

The repo has no Dockerfile; this one is a starting point and has not been tested.

```dockerfile
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
CMD ["python", "app.py"]
```

```bash
docker build -t prompt-reverse-engineer .
docker run --gpus all -p 7860:7860 -e HF_TOKEN=hf_... prompt-reverse-engineer
```

The local pathway is not practical on CPU (an 8B VLM plus a 12B diffusion transformer); without a GPU, use the API pathway below.

### Option E: API-only pathway (no local GPU)

Set these environment variables before launching:

```bash
export VLM_BACKEND=api
export VLM_API_KEY=your_openrouter_key       # or OPENAI_API_KEY compatible endpoint
export VLM_API_MODEL=qwen/qwen3-vl-32b-instruct
export VLM_API_BASE_URL=https://openrouter.ai/api/v1

export SCORER_BACKEND=api
export SCORER_API_KEY=your_openai_key        # gpt-4o for VQAScore logprobs
export SCORER_API_MODEL=gpt-4o

export PROXY_BACKEND=fal
export FAL_KEY=your_fal_key
export FAL_ENDPOINT=fal-ai/flux/schnell

export DINO_MODEL_ID=facebook/dinov2-small   # small, CPU-friendly
```

Then run `python app.py` or `python cli.py ...` as above.

---

## Configuration

All model backends and runtime knobs are controlled via environment variables (see `inversion/backends.py`). The Gradio UI also exposes the most common gates as sliders.

| Variable | Meaning | Default | Notes |
|----------|---------|---------|-------|
| `VLM_BACKEND` | `local` or `api` | `local` | VLM for DSG extraction + rewrite |
| `VLM_MODEL_ID` | HF repo (local) | `Qwen/Qwen3-VL-8B-Instruct` | ~17.5 GB in bf16 (no quantization) |
| `VLM_API_MODEL` | Model name (api) | `qwen/qwen3-vl-32b-instruct` | OpenRouter format |
| `VLM_API_BASE_URL` | API base URL | `https://openrouter.ai/api/v1` | |
| `VLM_API_KEY` | API key | — | Required if `VLM_BACKEND=api` |
| `SCORER_BACKEND` | `local` or `api` | auto (matches VLM) | VQAScore backend |
| `SCORER_API_MODEL` | API model for scoring | `gpt-4o` | Needs logprobs |
| `SCORER_API_KEY` | API key for scorer | `$OPENAI_API_KEY` | |
| `SCORER_API_BASE_URL` | API base URL for scorer | OpenAI | |
| `PROXY_BACKEND` | `flux` or `fal` | `flux` | Proxy renderer |
| `PROXY_MODEL_ID` | HF repo (flux) | `black-forest-labs/FLUX.2-klein-4B` | Also supports `black-forest-labs/FLUX.1-schnell`; see "Choosing the proxy model" |
| `PROXY_CPU_OFFLOAD` | Keep FLUX on CPU and move each component to GPU only while it runs | `0` | Saves ~10–24 GB of GPU memory; slower. Local GPUs only, not ZeroGPU |
| `PROXY_FP8` | Store the FLUX transformer in float8, compute in bf16 | `0` | Saves ~3.9 GB of resident weights with klein-4B (~33.9 → ~30 GB) and ~12 GB with schnell (~51.6 → ~40 GB); slightly changes proxy renders |
| `ZEROGPU_SIZE` | `auto`, `large` or `xlarge` | `auto` | Size of every `@spaces.GPU` call; `auto` picks `xlarge` when resident weights exceed 42 GB (logged at startup) |
| `FAL_ENDPOINT` | FAL endpoint | `fal-ai/flux/schnell` | If `PROXY_BACKEND=fal` |
| `DINO_MODEL_ID` | DINO model | `facebook/dinov2-base` | DINOv2-B (ungated). Any DINOv2/DINOv3 checkpoint works; `facebook/dinov3-*` is gated with manual approval, and the 0.80 gate is not transferable between backbones |
| `DINO_DEVICE` | `cuda` or `cpu` | auto (CUDA if available) | |
| `VQA_MODE` | `normalized` or `yes` | `normalized` | `normalized` = P(yes) / (P(yes) + P(no)); `yes` = the original VQAScore, P("Yes") only (thresholds not comparable) |

**EngineConfig** (Python API / CLI flags) — runtime gates per run:

| Key | Meaning | Default | UI range |
|-----|---------|---------|-------|
| `max_iters` | Max proxy renders (render → verify → rewrite rounds) | `5` | 1–8 |
| `vqa_gate` | S_VQA convergence threshold | `0.90` | 0.5–1.0 |
| `dino_gate` | S_DINO convergence threshold (`0` = stop on S_VQA alone, the research plan's rule) | `0.80` | 0.0–1.0 |
| `fail_threshold` | Predicates scoring below this are sent to the rewriter | `0.75` | 0.3–0.95 |
| `parent_gate` | DSG rule: children of an entity scoring below this score 0 | `0.5` | — |
| `ground` | Enable grounding on original | `True` | — |
| `grounding_threshold` | P(yes) on the original below which a predicate is pruned; also the minimum for re-localizing a grid cell | `0.5` | — |
| `w_vqa` / `w_dino` | Weights of the joint score used to pick the best iteration | `0.7` / `0.3` | 0.0–1.0 |
| `base_seed` | Base seed (proxy seed = base + t) | `42` | — |
| `proxy_long_side` | Proxy render long side (px) | `1024` | — |

---

## Usage / API

### Python API (single image)

```python
from PIL import Image
from inversion import EngineConfig, invert_image_to_verified_prompt
from inversion.backends import BackendSettings, load_components

# Load models (respects env vars)
settings = BackendSettings()
components = load_components(settings)
backends = components.backends()

# Configure run
cfg = EngineConfig(
    max_iters=5,
    vqa_gate=0.90,
    dino_gate=0.80,
    fail_threshold=0.75,
    w_vqa=0.7,
    base_seed=42,
    ground=True,
)

# Run
og = Image.open("my_image.png").convert("RGB")
run = invert_image_to_verified_prompt(og, backends, cfg)

# Results
print("Verified prompt:", run.best_prompt)
print(f"Best iteration: t={run.best.t}")
print(f"S_VQA: {run.best.mean_vqa:.3f}, S_DINO: {run.best.dino_patch:.3f}, Joint: {run.best.joint:.3f}")

# Save full report
from inversion.engine import save_run
save_run(run, "runs/my_image", og)
```

### Gradio UI (app.py)

The Space UI exposes:

- Image upload (drag-drop, file picker, clipboard paste)
- Stopping rule sliders: max iterations, S_VQA gate, S_DINO gate
- Advanced accordion: failure threshold, S_VQA weight, base seed, grounding toggle
- Live streaming: stage progress bar, per-iteration scores, predicate table, scene graph JSON, grounding log
- Outputs: verified prompt (copy button), proxy render gallery, downloadable JSON report

---

## Output / Data Flow

```
Input image
    │
    ▼
┌─────────────────────────────────────┐
│ Module 1: Constrained DSG extraction│  → DSGSceneSchema (JSON)
│   Qwen3-VL → entities, attrs, rels  │     • medium_and_optics[]
│   3×3 grid, ≤12 nodes               │     • background
└─────────────────────────────────────┘     • color_palette[]
    │                                       • nodes[] (id, grid_cell, entity, attrs, rels)
    ▼                                       • extra_checks[]
┌─────────────────────────────────────┐
│ Module 2: Grounding on original     │  → predicates[] (linearized)
│   VQAScore each predicate on OG     │     • dropped[] (hallucinations removed)
│   Parent-gate dependency blocking   │     • relocalized[] (grid cell corrections)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Module 2b: Canonical linearization  │  → canonical_prompt (string)
│   Deterministic template from graph │
└─────────────────────────────────────┘
    │
    ▼ (loop max_iters times)
┌─────────────────────────────────────┐
│ Module 3: Proxy render (FLUX)       │  → proxy_image (PIL)
│   seed = base_seed + t              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Module 4: Dual verification         │  → scored[] (Predicate + raw/effective)
│   VQAScore each predicate on proxy  │     • mean_vqa (S_VQA)
│   DINOv2 patch/cls similarity       │     • dino_patch (S_DINO), dino_cls
│   Joint = w_vqa·S_VQA + w_dino·S_DINO│
└─────────────────────────────────────┘
    │
    ├─ Gates met? → STOP (converged)
    │
    ▼
┌─────────────────────────────────────┐
│ Targeted diff rewrite               │  → RewritePatch (edits + additions)
│   VLM rewrites only failed facts    │     Applied → new prompt for next iter
└─────────────────────────────────────┘
```

**Report schema** (`report.json` per run):
```json
{
  "config": { ...EngineConfig... },
  "models": { "vlm_model", "scorer", "vqa_mode", "dino_model", "proxy_model", "proxy_pipeline", "proxy_steps", "proxy_guidance_scale", "proxy_fp8" },
  "stop_reason": "converged" | "max_iters",
  "best_prompt": "verified prompt string",
  "best_iteration": 3,
  "best_scores": { "S_VQA": 0.92, "S_DINO_patch": 0.84, "joint": 0.88 },
  "canonical_prompt": "initial linearization",
  "proxy_size": [1024, 768],
  "raw_graph": { ...DSGSceneSchema as extracted... },
  "graph": { ...DSGSceneSchema after grounding... },
  "predicates": [ { "id", "kind", "statement", "parents", "ref" }, ... ],
  "grounding_dropped": [ { "id", "kind", "statement", "parents", "p_yes", "effective", "blocked_by" }, ... ],
  "grounding_relocalized": [ { "node", "entity", "from", "to", "p_yes" }, ... ],
  "iterations": [ { "t", "seed", "prompt", "S_VQA", "S_DINO_patch", "S_DINO_cls", "joint",
                    "failed", "rewrite_notes", "patch", "rewrite_error", "predicates" }, ... ]
}
```

---

## Performance

No runtime benchmarks have been measured yet. What is known:

- **Downloads.** The local pathway fetches ~33.9 GB of weights on first run with the default proxy (Qwen3-VL-8B 17.5 GB, FLUX.2-klein-4B diffusers components 16.0 GB, DINOv2-B 0.35 GB), or ~51.6 GB with FLUX.1-schnell (33.7 GB); later runs use the cache. A fresh Space container downloads them at every cold start.
- **GPU memory.** See the table under Option B.
- **Calls per image.** One extraction, one grounding pass on the original (plus one re-localization pass if a grid cell is rejected), then per iteration one render, one VQA pass, two DINO embeddings, and a rewrite unless it is the last iteration. On ZeroGPU each of these is its own `@spaces.GPU` allocation.

### Hugging Face Space (ZeroGPU)

ZeroGPU copies all loaded weights onto the GPU at the start of every `@spaces.GPU` call, so they must fit the slice with room for activations. With the default FLUX.2-klein-4B the weights are ~33.9 GB, which fits a `large` slice (50.87 GB) in bf16, so no variables are needed. `ZEROGPU_SIZE=auto` reads the loaded weight size at startup and requests `xlarge` (the full GPU, double quota cost per second) only above 42 GB; the chosen size is logged as `[startup] resident model weights: … -> ZeroGPU size '…'`, and with the defaults it should read `'large'`. Nothing in the default stack is gated, so no `HF_TOKEN` is needed.

To run the Space on FLUX.1-schnell instead, set the variables `PROXY_MODEL_ID=black-forest-labs/FLUX.1-schnell` and `PROXY_FP8=1` (~40 GB, fits `large`; without fp8 it is ~51.6 GB and `auto` requests `xlarge`), accept its license on the model page, and add `HF_TOKEN` as a Space secret. `PROXY_FP8=1` also applies to klein-4B (~30 GB), so remove it to render in bf16.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` | VRAM exhausted | Set `PROXY_FP8=1` or `PROXY_CPU_OFFLOAD=1`; reduce `proxy_long_side`; use `PROXY_BACKEND=fal` or the API pathway |
| ZeroGPU: `memory allocation failed with OOM ... total: 50868518912` then `NVML_SUCCESS == r INTERNAL ASSERT FAILED` in `pack_to_cuda` | ZeroGPU copies all resident weights onto the GPU at the start of every call; with FLUX.1-schnell in bf16 they are ~51.6 GB and a `large` slice holds 50.87 GB | Use the default FLUX.2-klein-4B (~33.9 GB), or schnell with `PROXY_FP8=1` (~40 GB, fits `large`), or keep bf16 with `ZEROGPU_SIZE=auto`, which then requests `xlarge`; check the `[startup] ... ZeroGPU size` log line |
| `VLM_API_KEY not set` | API pathway selected without key | Export `VLM_API_KEY` (OpenRouter) or switch to `VLM_BACKEND=local` |
| `gpt-4o logprobs unavailable` | Scorer API key lacks logprobs access | Use OpenAI key with logprobs; or set `SCORER_BACKEND=local` |
| `Access denied` / 401 / 403 for FLUX.1-schnell (only if you set it as `PROXY_MODEL_ID`) | Gated repo | Accept the license on the model page, then `hf auth login` locally or add `HF_TOKEN` as a Space secret; or use the ungated default FLUX.2-klein-4B |
| `Access denied` for `facebook/dinov3-*` | Gated with manual approval | Use the default `facebook/dinov2-base` until access is granted |
| Gradio UI won't load | Port 7860 in use | `GRADIO_SERVER_PORT=7861 python app.py` |
| Space slow to start | Cold start downloads ~34 GB of weights (~52 GB with schnell), then ZeroGPU packs them | Normal; the first run after startup also waits for a GPU allocation |

**Logs:** Gradio console output shows per-stage status. CLI prints iteration status to stdout. Full run data in `runs/<stem>/report.json`.

---

## Attribution & License

**Author:** Gaspar Dario Larrain Varas  
**Space:** https://huggingface.co/spaces/glarr/Prompt-Reverse-Engineer  
**Repository:** https://github.com/glarrainv/Prompt-Reverse-Engineer  
**License:** Apache-2.0

This work builds on the following research:

```bibtex
@inproceedings{cho2024dsg,
  title={Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation},
  author={Cho, Jaemin and Hu, Yushi and Garg, Roopal and Anderson, Peter and Krishna, Ranjay and Baldridge, Jason and Bansal, Mohit and Pont-Tuset, Jordi and Wang, Su},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@inproceedings{lin2024vqascore,
  title={Evaluating Text-to-Visual Generation with Image-to-Text Generation},
  author={Lin, Zhiqiu and Pathak, Deepak and Li, Baiqi and Li, Jiayao and Xia, Xide and Neubig, Graham and Zhang, Pengchuan and Ramanan, Deva},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}

@article{oquab2024dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024}
}

@article{simeoni2025dinov3,
  title={DINOv3},
  author={Sim{\'e}oni, Oriane and others},
  journal={arXiv preprint arXiv:2508.10104},
  year={2025}
}

@article{bai2025qwen25vl,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and others},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}

@misc{flux2024,
  title={FLUX},
  author={{Black Forest Labs}},
  year={2024},
  howpublished={https://github.com/black-forest-labs/flux}
}

@software{larrain2026promptreverse,
  author = {Larrain Varas, Gaspar Dario},
  title = {Prompt Reverse Engineer},
  year = {2026},
  url = {https://github.com/glarrainv/Prompt-Reverse-Engineer},
  license = {Apache-2.0}
}
```

**Model credits:** Qwen Team (Qwen3-VL), Black Forest Labs (FLUX.1), Meta AI (DINOv2/DINOv3), OpenAI (gpt-4o for VQAScore API pathway).
