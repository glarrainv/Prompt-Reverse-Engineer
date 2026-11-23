"""Closed-loop DSG + VQA image-to-prompt inversion engine."""

from .compiler import apply_dependencies, apply_patch, compile_predicates, linearize, prune_hallucinations
from .engine import (Backends, EngineConfig, InversionRun, invert_image_to_verified_prompt, iter_inversion,
                     save_run)
from .schema import DSGSceneSchema, RewritePatch, SpatialNode

__all__ = [
    "Backends", "DSGSceneSchema", "EngineConfig", "InversionRun", "RewritePatch", "SpatialNode",
    "apply_dependencies", "apply_patch", "compile_predicates", "invert_image_to_verified_prompt",
    "iter_inversion", "linearize", "prune_hallucinations", "save_run",
]
