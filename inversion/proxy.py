"""Module 3: fast proxy text-to-image renderer (the internal forward model)."""

from __future__ import annotations

import io
import urllib.request

from PIL import Image


def proxy_size(og_size: tuple[int, int], long_side: int = 1024, multiple: int = 16) -> tuple[int, int]:
    """Match the OG aspect ratio at ~`long_side`, snapped to the model's latent multiple."""
    w, h = og_size
    scale = long_side / max(w, h)

    def snap(x: float) -> int:
        return max(multiple, int(round(x * scale / multiple)) * multiple)

    return snap(w), snap(h)


# Step-distilled proxies the renderer knows how to drive: diffusers pipeline class -> (steps, guidance_scale).
# The pipeline defaults (50 steps, guidance 4.0 for FLUX.2) are wrong for distilled weights, so both are explicit.
SUPPORTED_PROXY_PIPELINES: dict[str, tuple[int, float]] = {
    "FluxPipeline": (4, 0.0),  # black-forest-labs/FLUX.1-schnell
    "Flux2KleinPipeline": (4, 1.0),  # black-forest-labs/FLUX.2-klein-4B (step-distilled; needs diffusers >= 0.37)
}


def proxy_defaults(class_name: str, is_distilled: bool | None = None) -> tuple[int, float]:
    """(steps, guidance_scale) for a proxy pipeline, rejecting ones this renderer cannot drive correctly."""
    if class_name not in SUPPORTED_PROXY_PIPELINES:
        raise ValueError(f"PROXY_MODEL_ID uses pipeline {class_name!r}; supported: "
                         f"{', '.join(SUPPORTED_PROXY_PIPELINES)} (FLUX.1-schnell, FLUX.2-klein-4B).")
    if class_name == "Flux2KleinPipeline" and is_distilled is not True:
        # The klein *base* models are not step-distilled: they need ~50 steps and classifier-free guidance.
        raise ValueError("This FLUX.2 klein checkpoint is not step-distilled (`is_distilled` is not true in its "
                         "model_index.json), so 4-step rendering would be poor. Use black-forest-labs/FLUX.2-klein-4B.")
    return SUPPORTED_PROXY_PIPELINES[class_name]


class FluxRenderer:
    """Step-distilled FLUX proxy via diffusers: FLUX.1-schnell (default) or FLUX.2-klein-4B.

    The pipeline class, step count and guidance scale come from the checkpoint's model_index.json, so
    `PROXY_MODEL_ID` is the only switch.
    """

    def __init__(self, model_id: str = "black-forest-labs/FLUX.2-klein-4B", device: str | None = None,
                 cpu_offload: bool = False, steps: int | None = None, max_sequence_length: int = 512,
                 fp8: bool = False):
        import torch
        from diffusers import DiffusionPipeline

        config = DiffusionPipeline.load_config(model_id)  # model_index.json only; no weights yet
        self.pipeline_class = config["_class_name"]
        default_steps, self.guidance_scale = proxy_defaults(self.pipeline_class, config.get("is_distilled"))
        self.steps = default_steps if steps is None else steps
        self.max_sequence_length = max_sequence_length
        self.torch = torch
        self.pipe = DiffusionPipeline.from_pretrained(model_id, dtype=torch.bfloat16)
        if fp8:
            # Store the transformer in float8 and upcast each layer to bf16 when it runs (FLUX.1-schnell:
            # ~24 GB -> ~12 GB). Norms and positional embeddings stay bf16. Opt-in: it slightly changes renders.
            self.pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                                           compute_dtype=torch.bfloat16)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.info = {"proxy_model": model_id, "proxy_pipeline": self.pipeline_class, "proxy_steps": self.steps,
                     "proxy_guidance_scale": self.guidance_scale, "proxy_fp8": bool(fp8)}

    def __call__(self, prompt: str, seed: int, width: int, height: int) -> Image.Image:
        generator = self.torch.Generator(device="cpu").manual_seed(int(seed))
        # `prompt=` by keyword: Flux2KleinPipeline's first positional parameter is `image`, not `prompt`.
        return self.pipe(
            prompt=prompt, width=width, height=height, num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale, max_sequence_length=self.max_sequence_length, generator=generator,
        ).images[0]


FluxSchnellRenderer = FluxRenderer  # backwards-compatible name


class FalRenderer:
    """fal-ai/flux/schnell over the fal.ai API (reads FAL_KEY from the environment)."""

    def __init__(self, endpoint: str = "fal-ai/flux/schnell", steps: int = 4):
        import fal_client

        self.fal, self.endpoint, self.steps = fal_client, endpoint, steps
        self.info = {"proxy_model": endpoint, "proxy_pipeline": "fal", "proxy_steps": steps}

    def __call__(self, prompt: str, seed: int, width: int, height: int) -> Image.Image:
        result = self.fal.subscribe(self.endpoint, arguments={
            "prompt": prompt, "seed": int(seed), "num_inference_steps": self.steps,
            "image_size": {"width": width, "height": height}, "num_images": 1,
        })
        url = result["images"][0]["url"]
        with urllib.request.urlopen(url, timeout=60) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
