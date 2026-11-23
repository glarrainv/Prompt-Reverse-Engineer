"""Build model components from environment variables (one place for all deployment knobs).

Pathway B (local / HPC / HF Space GPU) is the default:
    VLM_BACKEND=local, PROXY_BACKEND=flux  -> Qwen3-VL-8B + FLUX.1-schnell + DINOv2-B, all local.
Pathway A (no local GPU):
    VLM_BACKEND=api   VLM_API_KEY=...  (OpenRouter by default, Qwen3-VL-32B-Instruct)
    SCORER_BACKEND=api OPENAI_API_KEY=... (gpt-4o logprobs for VQAScore)
    PROXY_BACKEND=fal  FAL_KEY=...
    DINO_MODEL_ID=facebook/dinov2-small (small, CPU-friendly)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from PIL import Image

from .dino import DEFAULT_DINO_MODEL
from .engine import Backends

if TYPE_CHECKING:
    from .dino import DinoEmbedder
    from .vlm import VLMBase


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class BackendSettings:
    vlm_backend: str = field(default_factory=lambda: _env("VLM_BACKEND", "local"))
    vlm_model_id: str = field(default_factory=lambda: _env("VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"))
    vlm_api_model: str = field(default_factory=lambda: _env("VLM_API_MODEL", "qwen/qwen3-vl-32b-instruct"))
    vlm_api_base_url: str | None = field(
        default_factory=lambda: _env("VLM_API_BASE_URL", "https://openrouter.ai/api/v1"))
    vlm_api_key: str | None = field(default_factory=lambda: _env("VLM_API_KEY"))

    # Defaults to sharing the VLM's weights when local (Module 1/4A/rewriter on one model).
    scorer_backend: str | None = field(default_factory=lambda: _env("SCORER_BACKEND"))
    scorer_api_model: str = field(default_factory=lambda: _env("SCORER_API_MODEL", "gpt-4o"))
    scorer_api_base_url: str | None = field(default_factory=lambda: _env("SCORER_API_BASE_URL"))
    scorer_api_key: str | None = field(
        default_factory=lambda: _env("SCORER_API_KEY", _env("OPENAI_API_KEY")))

    proxy_backend: str = field(default_factory=lambda: _env("PROXY_BACKEND", "flux"))
    proxy_model_id: str = field(default_factory=lambda: _env("PROXY_MODEL_ID", "black-forest-labs/FLUX.1-schnell"))
    proxy_cpu_offload: bool = field(default_factory=lambda: _env("PROXY_CPU_OFFLOAD", "0") == "1")
    fal_endpoint: str = field(default_factory=lambda: _env("FAL_ENDPOINT", "fal-ai/flux/schnell"))

    vqa_mode: str = field(default_factory=lambda: _env("VQA_MODE", "normalized"))

    dino_model_id: str = field(
        default_factory=lambda: _env("DINO_MODEL_ID", DEFAULT_DINO_MODEL))
    dino_device: str | None = field(default_factory=lambda: _env("DINO_DEVICE"))


@dataclass
class Components:
    vlm: VLMBase
    scorer: VLMBase
    renderer: Callable[[str, int, int, int], Image.Image]
    dino: DinoEmbedder

    def backends(self) -> Backends:
        return Backends(
            extract_dsg=self.vlm.extract_dsg,
            vqa_scores=self.scorer.vqa_scores,
            rewrite=self.vlm.rewrite,
            render=self.renderer,
            embed=self.dino.embed,
        )


def load_components(settings: BackendSettings | None = None) -> Components:
    s = settings or BackendSettings()
    from .dino import DinoEmbedder
    from .vlm import LocalQwenVLM, OpenAICompatVLM

    if s.vlm_backend == "local":
        vlm = LocalQwenVLM(s.vlm_model_id, vqa_mode=s.vqa_mode)
    elif s.vlm_backend == "api":
        vlm = OpenAICompatVLM(s.vlm_api_model, base_url=s.vlm_api_base_url, api_key=s.vlm_api_key,
                              vqa_mode=s.vqa_mode)
    else:
        raise ValueError(f"VLM_BACKEND must be 'local' or 'api', got {s.vlm_backend!r}")

    scorer_backend = s.scorer_backend or ("local" if s.vlm_backend == "local" else "api")
    if scorer_backend == "local":
        scorer = vlm if isinstance(vlm, LocalQwenVLM) else LocalQwenVLM(s.vlm_model_id, vqa_mode=s.vqa_mode)
    elif scorer_backend == "api":
        scorer = OpenAICompatVLM(s.scorer_api_model, base_url=s.scorer_api_base_url, api_key=s.scorer_api_key,
                                 vqa_mode=s.vqa_mode)
    else:
        raise ValueError(f"SCORER_BACKEND must be 'local' or 'api', got {scorer_backend!r}")

    if s.proxy_backend == "flux":
        from .proxy import FluxSchnellRenderer
        renderer = FluxSchnellRenderer(s.proxy_model_id, cpu_offload=s.proxy_cpu_offload)
    elif s.proxy_backend == "fal":
        from .proxy import FalRenderer
        renderer = FalRenderer(s.fal_endpoint)
    else:
        raise ValueError(f"PROXY_BACKEND must be 'flux' or 'fal', got {s.proxy_backend!r}")

    dino = DinoEmbedder(s.dino_model_id, device=s.dino_device)
    return Components(vlm=vlm, scorer=scorer, renderer=renderer, dino=dino)
