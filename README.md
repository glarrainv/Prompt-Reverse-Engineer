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
  - black-forest-labs/FLUX.1-schnell
  - facebook/dinov3-vitb16-pretrain-lvd1689m
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

Turn an image into a verified text-to-image prompt. A closed-loop pipeline extracts a constrained scene graph, grounds every fact on the original, renders with a proxy model, scores each predicate with VQAScore and layout with DINOv3, and rewrites only what fails — until both gates clear or iterations run out.

[![License](https://img.shields.io/github/license/glarrainv/Prompt-Reverse-Engineer?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/glarrainv/Prompt-Reverse-Engineer?style=for-the-badge)](https://github.com/glarrainv/Prompt-Reverse-Engineer/stargazers)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/glarr/Prompt-Reverse-Engineer)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-orange?style=for-the-badge&logo=gradio)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.22-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/docs/transformers)

---

## Overview

**Prompt Reverse Engineer** inverts text-to-image generation: instead of prompting a model and hoping the output matches your vision, you provide an image and get back a prompt whose every claim is verified against a proxy render. The pipeline runs four stages in a loop:

1. **Extract** a Davidsonian Scene Graph (DSG) — entities, attributes, relations, and medium/optics on a 3×3 grid
2. **Ground** every predicate on the original image; drop or re-localize facts the scorer rejects
3. **Render** the linearized prompt with a fast proxy model (FLUX.1-schnell by default)
4. **Verify** with VQAScore (predicate agreement) and DINOv3 (layout similarity); **rewrite** only failed facts

**What makes this different:**
- Zero-hallucination grounding: facts false on the original never reach the prompt
- Targeted diff rewrites: only failed predicates are edited, preserving the rest
- Dual-gate stopping: both predicate agreement (S_VQA) and layout similarity (S_DINO) must clear configurable thresholds
- Two deployment pathways: local GPUs (Qwen3-VL + FLUX + DINOv3) or API-only (OpenRouter + FAL + gpt-4o)
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

# 2. Create venv and install (PyTorch + CUDA 12.1 assumed; adjust for your hardware)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Accept gated model licenses and set HF token
huggingface-cli login  # or export HF_TOKEN=hf_...

# 4. Run the Gradio UI
python app.py
# → opens at http://localhost:7860
```

> **GPU requirements (local pathway):** 24 GB VRAM recommended for Qwen3-VL-8B + FLUX.1-schnell + DINOv3-B. With 16 GB, enable CPU offload: `PROXY_CPU_OFFLOAD=1`.

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

### Option D: Docker (CPU/GPU)

```dockerfile
# Dockerfile (add to repo root if needed)
FROM pytorch/pytorch:2.13.0-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

```bash
docker build -t prompt-reverse-engineer .
# GPU:
docker run --gpus all -p 7860:7860 prompt-reverse-engineer
# CPU (slow):
docker run -p 7860:7860 prompt-reverse-engineer
```

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
| `VLM_MODEL_ID` | HF repo (local) | `Qwen/Qwen3-VL-8B-Instruct` | Requires ~16 GB VRAM in 4-bit |
| `VLM_API_MODEL` | Model name (api) | `qwen/qwen3-vl-32b-instruct` | OpenRouter format |
| `VLM_API_BASE_URL` | API base URL | `https://openrouter.ai/api/v1` | |
| `VLM_API_KEY` | API key | — | Required if `VLM_BACKEND=api` |
| `SCORER_BACKEND` | `local` or `api` | auto (matches VLM) | VQAScore backend |
| `SCORER_API_MODEL` | API model for scoring | `gpt-4o` | Needs logprobs |
| `SCORER_API_KEY` | API key for scorer | `$OPENAI_API_KEY` | |
| `PROXY_BACKEND` | `flux` or `fal` | `flux` | Proxy renderer |
| `PROXY_MODEL_ID` | HF repo (flux) | `black-forest-labs/FLUX.1-schnell` | |
| `PROXY_CPU_OFFLOAD` | Offload FLUX to CPU | `0` | Set `1` for <24 GB VRAM |
| `FAL_ENDPOINT` | FAL endpoint | `fal-ai/flux/schnell` | If `PROXY_BACKEND=fal` |
| `DINO_MODEL_ID` | DINO model | `facebook/dinov3-vitb16-pretrain-lvd1689m` | DINOv3-B default |
| `DINO_DEVICE` | `cuda`, `cpu`, `mps` | auto | |
| `VQA_MODE` | `normalized` or `raw` | `normalized` | VQAScore normalization |

**EngineConfig** (Python API / CLI flags) — runtime gates per run:

| Key | Meaning | Default | Range |
|-----|---------|---------|-------|
| `max_iters` | Max rewrite iterations | `5` | 1–8 |
| `vqa_gate` | S_VQA convergence threshold | `0.90` | 0.5–1.0 |
| `dino_gate` | S_DINO convergence threshold | `0.80` | 0.0–1.0 |
| `fail_threshold` | Predicate failure threshold | `0.75` | 0.3–0.95 |
| `parent_gate` | DSG parent gate for dependency blocking | `0.5` | — |
| `ground` | Enable grounding on original | `True` | — |
| `grounding_threshold` | Re-localization acceptance gate | `0.5` | — |
| `w_vqa` | Weight on S_VQA for best pick | `0.7` | 0.0–1.0 |
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
- Image upload (drag-drop, clipboard, webcam)
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
│   DINOv3 patch/cls similarity       │     • dino_patch (S_DINO), dino_cls
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
  "stop_reason": "converged" | "max_iters",
  "best_prompt": "verified prompt string",
  "best_iteration": 3,
  "best_scores": { "S_VQA": 0.92, "S_DINO_patch": 0.84, "joint": 0.88 },
  "canonical_prompt": "initial linearization",
  "proxy_size": [1024, 768],
  "graph": { ...DSGSceneSchema... },
  "predicates": [ { "id", "kind", "statement", "ref" }, ... ],
  "grounding_dropped": [ ... ],
  "grounding_relocalized": [ { "node", "entity", "from", "to", "p_yes" }, ... ],
  "iterations": [ { "t", "seed", "prompt", "S_VQA", "S_DINO_patch", "joint", "failed_ids", "patch" }, ... ]
}
```

---

## Performance

| Backend | Hardware | Time/Image (5 iters) | VRAM (peak) | Notes |
|---------|----------|----------------------|-------------|-------|
| Local (Qwen3-VL-8B + FLUX-schnell + DINOv3-B) | RTX 4090 24 GB | ~180 s | ~22 GB | Default pathway |
| Local + CPU offload (FLUX) | RTX 3080 16 GB | ~240 s | ~14 GB | `PROXY_CPU_OFFLOAD=1` |
| API (OpenRouter Qwen3-VL-32B + FAL FLUX + gpt-4o) | CPU only | ~90 s | <2 GB | Network-bound |
| Local (Qwen3-VL-8B 4-bit + FLUX-schnell 8-bit) | RTX 4090 24 GB | ~150 s | ~16 GB | Quantization WIP |

> First run downloads model weights (~35 GB total for local pathway). Subsequent runs are cached.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` | VRAM exhausted | Set `PROXY_CPU_OFFLOAD=1`; reduce `proxy_long_side`; use API pathway |
| `VLM_API_KEY not set` | API pathway selected without key | Export `VLM_API_KEY` (OpenRouter) or switch to `VLM_BACKEND=local` |
| `gpt-4o logprobs unavailable` | Scorer API key lacks logprobs access | Use OpenAI key with logprobs; or set `SCORER_BACKEND=local` |
| `FLUX model not found` | HF token/gated repo | `huggingface-cli login` or set `PROXY_MODEL_ID` to public variant |
| `DINOv3 model not found` | Wrong model ID | Use `facebook/dinov3-vitb16-pretrain-lvd1689m` (default) or `facebook/dinov2-base` |
| Gradio UI won't load | Port 7860 in use | `GRADIO_SERVER_PORT=7861 python app.py` |
| First iteration very slow | Model loading + compilation | Normal; torch.compile + model init on first forward |

**Logs:** Gradio console output shows per-stage status. CLI prints iteration status to stdout. Full run data in `runs/<stem>/report.json`.

---

## Attribution & License

**Author:** Gaspar Dario Larrain Varas  
**Space:** https://huggingface.co/spaces/glarr/Prompt-Reverse-Engineer  
**Repository:** https://github.com/glarrainv/Prompt-Reverse-Engineer  
**License:** Apache-2.0

This work builds on the following research:

```bibtex
@article{cho2024dsg,
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
