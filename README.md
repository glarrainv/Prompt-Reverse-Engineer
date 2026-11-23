---
title: Prompt Reverse Engineer
emoji: 🔁
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12.12"
suggested_hardware: zero-a10g
app_file: app.py
pinned: true
short_description: Image to prompt reverse enginner, turn any image back.
license: apache-2.0
models:
  - Qwen/Qwen3-VL-8B-Instruct
  - black-forest-labs/FLUX.1-schnell
  - facebook/dinov2-base
library_name: transformers
tags:
  - image-to-prompt
  - computer vision
  - captioning
  - qwen
---

# Prompt Reverse Engineer: closed-loop DSG + VQA inversion

The original pipeline was one step: an image went into Qwen3-VL-8B, which returned unconstrained prose. This version wraps that same VLM in a self-correcting loop with four modules:

```
OG image S_k
  │
  ▼  Module 1  Constrained DSG extractor   (Qwen3-VL + Pydantic schema, 3×3 grid, medium/palette)
  │            └─ grounding: every predicate is checked on S_k itself; facts the scorer rejects
  │               on the original are pruned as hallucinations (they could never be satisfied),
  │               including medium/optics facets, which are removed from the prompt as well;
  │               a rejected grid cell is re-localized by scoring all 9 cells on S_k
  ▼  Module 2  Predicate compiler + canonical linearizer
  │            G_k → dependency-ordered yes/no predicates Q_DSG (global → entity → attribute
  │            → position → relation) and a deterministic canonical prompt P_k^(0)
  ▼  Module 3  Proxy T2I renderer           (FLUX.1-schnell, 4 steps, seed = 42 + t)
  ▼  Module 4  Dual verifier
  │            4A VQAScore: P(yes | proxy, 'Does this image show "{q}"? …') per predicate,
  │               DSG rule: children of an absent parent entity score 0
  │            4B DINOv2 patch-aligned cosine between S_k and the proxy (layout)
  │
  ├─ S_VQA ≥ 0.90 AND S_DINO ≥ 0.80    → lock P_k*
  ├─ t = max_iters                      → return best prompt by 0.7·S_VQA + 0.3·S_DINO
  └─ otherwise: targeted diff rewrite   → only spans tied to failed predicates are edited
                (the VLM sees original + proxy; edits must quote the prompt verbatim and are applied
                 deterministically; unmatched edits are skipped and recorded, never guessed; edits that
                 would inject an era/style buzzword absent from the grounded prompt, e.g. "vintage",
                 "cinematic", "1970s", "Bauhaus", are skipped as unverified)
```

## Code map

| Path | Role |
| --- | --- |
| `inversion/schema.py` | Module 1 schema (`DSGSceneSchema`, `SpatialNode`, `Relation`) and the `RewritePatch` diff schema |
| `inversion/compiler.py` | Module 2: `compile_predicates`, `apply_dependencies`, `prune_hallucinations`, `linearize`, `apply_patch` |
| `inversion/vlm.py` | `LocalQwenVLM` (one set of weights for extraction, VQAScore, and rewriting) and `OpenAICompatVLM` (API) |
| `inversion/proxy.py` | Module 3: `FluxSchnellRenderer` (diffusers) and `FalRenderer` (fal.ai) |
| `inversion/dino.py` | Module 4B: `DinoEmbedder` (DINOv2-B default), `dino_similarity`, `medoid_index` |
| `inversion/engine.py` | The state machine: `iter_inversion` (streams stages), `invert_image_to_verified_prompt`, `save_run` |
| `inversion/backends.py` | Builds all components from environment variables |
| `app.py` | Gradio Space UI. Each model call gets its own `@spaces.GPU` allocation |
| `cli.py` | Batch `invert` over a stimulus folder, and DINO `medoid` selection over seed renders |
| `tests/` | Model-free tests of the schema, compiler, dependency rule, grounding, diff, and loop |

## Setup

**Prerequisites (both pathways).** `black-forest-labs/FLUX.1-schnell` is gated on Hugging Face. Accept its license with your account, then create a read token (`HF_TOKEN`). On the Space, add `HF_TOKEN` as a secret, or the Space will fail at startup. The default DINOv2 backbone is ungated; a gated `facebook/dinov3-*` checkpoint can be used through `DINO_MODEL_ID` once access is granted.

### Pathway B: local GPU, HPC, or the HF Space (default, no API cost)

```bash
conda create -n dsg_vqa_inv python=3.12 -y && conda activate dsg_vqa_inv
pip install -r requirements.txt
export HF_TOKEN=hf_...
python app.py                                     # UI
python cli.py invert --input stimuli/ --out runs/ # batch: runs/<stem>/report.json, best_prompt.txt, proxies
```

Loading Qwen3-VL-8B, FLUX.1-schnell, and DINOv2-B together needs about 52 GB. On a 24 GB card, set `PROXY_CPU_OFFLOAD=1`, or set `PROXY_BACKEND=fal` to move proxy rendering to the API.

### Pathway A: no local GPU (API)

```bash
export VLM_BACKEND=api VLM_API_KEY=sk-or-...         # OpenRouter, qwen/qwen3-vl-32b-instruct by default
export SCORER_BACKEND=api OPENAI_API_KEY=sk-...       # VQAScore via gpt-4o logprobs
export PROXY_BACKEND=fal FAL_KEY=...                  # fal-ai/flux/schnell
export DINO_MODEL_ID=facebook/dinov2-small DINO_DEVICE=cpu
python cli.py invert --input stimuli/ --out runs/
```

The API scorer reads P("Yes") from the top-20 logprobs of a one-token answer, so `SCORER_API_MODEL` must be a model that returns logprobs for image inputs (gpt-4o does). Many OpenRouter providers do not, which is why the scorer is configured separately from the VLM.

| Variable | Default |
| --- | --- |
| `VLM_BACKEND` | `local` (`api`) |
| `VLM_MODEL_ID` | `Qwen/Qwen3-VL-8B-Instruct` (e.g. `Qwen/Qwen3-VL-32B-Instruct`) |
| `VLM_API_MODEL` / `VLM_API_BASE_URL` / `VLM_API_KEY` | `qwen/qwen3-vl-32b-instruct` / OpenRouter / — |
| `SCORER_BACKEND` | same as VLM; reuses the VLM's weights when both are local |
| `SCORER_API_MODEL` / `SCORER_API_BASE_URL` / `SCORER_API_KEY` | `gpt-4o` / OpenAI / `OPENAI_API_KEY` |
| `PROXY_BACKEND` | `flux` (`fal`) |
| `PROXY_MODEL_ID`, `PROXY_CPU_OFFLOAD`, `FAL_ENDPOINT` | `black-forest-labs/FLUX.1-schnell`, `0`, `fal-ai/flux/schnell` |
| `DINO_MODEL_ID`, `DINO_DEVICE` | `facebook/dinov2-base` (Pivot 1 backbone), auto |
| `VQA_MODE` | `normalized` (`yes` = original VQAScore, P("Yes") only) |

## Stimulus-bank workflow

1. **Positive control.** Run `python cli.py invert --input 000DAH.png --out runs/` and inspect `runs/000DAH/report.json`. Check whether the camera, the "PALACE" marquee position, and the subject's orientation survived grounding as predicates, and whether S_VQA ≥ 0.90.
2. **Lock P\*_1…P\*_9.** Run `python cli.py invert --input stimuli/ --out runs/`. `runs/summary.csv` lists each image's stop reason, best scores, and best prompt.
3. **Render target conditions** in the target models, outside this repo. Save them as `renders/<condition>/<seed>.png`, then pick one medoid per condition: `python cli.py medoid --input renders/ --out medoids.json --copy-to medoids/`.

## Notes and limitations

- **VQAScore normalization.** The original VQAScore is P("Yes") over the full vocabulary. Qwen3-VL answers mostly with lowercase "yes". In testing, a fact it affirmed with probability 0.99 scored P("Yes") = 0.009, and batched and single runs disagreed by up to 0.05. By default the scorer therefore uses P(yes-variants) / (P(yes-variants) + P(no-variants)). `VQA_MODE=yes` restores the original definition, but its thresholds are then not comparable.

- **Structured output.** The API backend requests `json_schema` output. The local backend puts the schema in the prompt and validates with Pydantic, with up to two repair retries. It does not use token-level constrained decoding.
- **Self-verification bias.** With the local backend, one model extracts the predicates and also scores them. Grounding removes facts the model rejects on the original, but shared blind spots stay. For an independent check, use a different scorer (`SCORER_BACKEND=api`, or a larger `VLM_MODEL_ID` for scoring).
- **DINO gate.** S_DINO compares patch (i, j) of the original with patch (i, j) of the proxy at 448×448. It responds to layout and to left/right flips. Treat the 0.80 default as a strong target, not a calibrated threshold, and calibrate it on your own stimuli. The 0.80 value is not transferable between backbones (DINOv2 vs DINOv3), and it is an addition to the plan's stop rule (S_VQA ≥ 0.90 only); set `--dino-gate 0` to apply the plan's rule exactly.
- **Prompt length.** The canonical prompt can exceed the 77-token CLIP limit. FLUX reads up to 512 T5 tokens, so the proxy still sees the full prompt.

Author: Larrain, G
