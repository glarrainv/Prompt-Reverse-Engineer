"""Module 4B: DINO spatial-layout verifier and embedding medoid selection.

The default backbone is DINOv2 ViT-B/14 (ungated), as specified in the research plan (Pivot 1).
Any DINOv2/DINOv3 checkpoint works via DINO_MODEL_ID; patch tokens are always the last N
tokens, after CLS and any register tokens.

Both images are resized to the same square resolution so patch (i, j) of the
OG image is compared against patch (i, j) of the proxy. The mean per-patch
cosine is therefore sensitive to *where* things are (layout, left/right,
figure-ground), unlike a pooled CLS cosine, which is also reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


DEFAULT_DINO_MODEL = "facebook/dinov2-base"


@dataclass
class DinoEmbedding:
    cls: torch.Tensor  # [D], CPU float32
    patches: torch.Tensor  # [N, D], CPU float32, row-major patch grid


class DinoEmbedder:
    def __init__(self, model_id: str = DEFAULT_DINO_MODEL, device: str | None = None,
                 resolution: int = 448):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        proc = AutoImageProcessor.from_pretrained(model_id)
        self.mean = torch.tensor(getattr(proc, "image_mean", None) or [0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor(getattr(proc, "image_std", None) or [0.229, 0.224, 0.225]).view(3, 1, 1)
        self.patch = int(getattr(self.model.config, "patch_size", 16))
        self.resolution = resolution - resolution % self.patch
        self.n_patches = (self.resolution // self.patch) ** 2

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> DinoEmbedding:
        img = image.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - self.mean) / self.std).unsqueeze(0).to(self.device, next(self.model.parameters()).dtype)
        hidden = self.model(pixel_values=x).last_hidden_state[0].float().cpu()
        # Layout: [CLS, register tokens..., patch tokens]; patch tokens are always the last N.
        return DinoEmbedding(cls=hidden[0], patches=hidden[-self.n_patches:])


def dino_similarity(a: DinoEmbedding, b: DinoEmbedding) -> dict[str, float]:
    patch = torch.nn.functional.cosine_similarity(a.patches, b.patches, dim=-1).mean().item()
    cls = torch.nn.functional.cosine_similarity(a.cls, b.cls, dim=-1).item()
    return {"patch": patch, "cls": cls}


def medoid_index(embeddings: list[DinoEmbedding], key: str = "patch") -> tuple[int, list[float]]:
    """Index of the embedding with the highest mean similarity to all others."""
    n = len(embeddings)
    if n == 1:
        return 0, [1.0]
    means = []
    for i in range(n):
        sims = [dino_similarity(embeddings[i], embeddings[j])[key] for j in range(n) if j != i]
        means.append(sum(sims) / len(sims))
    return int(np.argmax(means)), means
