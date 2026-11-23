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


class FluxSchnellRenderer:
    """black-forest-labs/FLUX.1-schnell via diffusers (4-step distilled, guidance-free)."""

    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-schnell", device: str | None = None,
                 cpu_offload: bool = False, steps: int = 4, max_sequence_length: int = 512):
        import torch
        from diffusers import FluxPipeline

        self.torch = torch
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.steps, self.max_sequence_length = steps, max_sequence_length

    def __call__(self, prompt: str, seed: int, width: int, height: int) -> Image.Image:
        generator = self.torch.Generator(device="cpu").manual_seed(int(seed))
        return self.pipe(
            prompt, width=width, height=height, num_inference_steps=self.steps, guidance_scale=0.0,
            max_sequence_length=self.max_sequence_length, generator=generator,
        ).images[0]


class FalRenderer:
    """fal-ai/flux/schnell over the fal.ai API (reads FAL_KEY from the environment)."""

    def __init__(self, endpoint: str = "fal-ai/flux/schnell", steps: int = 4):
        import fal_client

        self.fal, self.endpoint, self.steps = fal_client, endpoint, steps

    def __call__(self, prompt: str, seed: int, width: int, height: int) -> Image.Image:
        result = self.fal.subscribe(self.endpoint, arguments={
            "prompt": prompt, "seed": int(seed), "num_inference_steps": self.steps,
            "image_size": {"width": width, "height": height}, "num_images": 1,
        })
        url = result["images"][0]["url"]
        with urllib.request.urlopen(url, timeout=60) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
