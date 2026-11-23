"""Closed-loop inversion engine: extract -> ground -> linearize -> (render -> verify -> diff-rewrite)*.

The engine is a generator that yields the same `InversionRun` after every stage,
so UIs can stream progress. Model work goes through the `Backends` callables,
which lets the Space wrap each one in its own ZeroGPU allocation and lets tests
substitute stubs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image

from .compiler import (PatchResult, Predicate, ScoredPredicate, apply_dependencies, apply_patch,
                       compile_predicates, linearize, position_statement, prune_hallucinations)
from .dino import DinoEmbedding, dino_similarity
from .proxy import proxy_size
from .schema import GRID_ORDER, DSGSceneSchema, RewritePatch
from .vlm import VQA_TEMPLATE


@dataclass
class EngineConfig:
    max_iters: int = 5
    vqa_gate: float = 0.90  # stop when mean effective VQAScore >= this ...
    dino_gate: float = 0.80  # ... AND DINO patch cosine >= this
    fail_threshold: float = 0.75  # predicates below this are sent to the rewriter
    parent_gate: float = 0.5  # DSG: children of a parent below this are blocked (scored 0)
    ground: bool = True  # verify extracted predicates on the OG image and prune hallucinations
    grounding_threshold: float = 0.5
    w_vqa: float = 0.7
    w_dino: float = 0.3
    base_seed: int = 42  # proxy seed at iteration t is base_seed + t
    proxy_long_side: int = 1024
    question_template: str = VQA_TEMPLATE


@dataclass
class Backends:
    extract_dsg: Callable[[Image.Image], DSGSceneSchema]
    vqa_scores: Callable[[Image.Image, list[str], str], list[float]]
    rewrite: Callable[[Image.Image, Image.Image, str, list[str]], RewritePatch]
    render: Callable[[str, int, int, int], Image.Image]
    embed: Callable[[Image.Image], DinoEmbedding]
    meta: dict = field(default_factory=dict)  # which models produced the run; copied into the report


@dataclass
class IterationRecord:
    t: int
    seed: int
    prompt: str
    proxy_image: Image.Image
    scored: list[ScoredPredicate]
    mean_vqa: float
    dino_patch: float
    dino_cls: float
    joint: float
    failed: list[ScoredPredicate]
    rewrite_notes: list[str] = field(default_factory=list)
    patch: PatchResult | None = None
    rewrite_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "t": self.t, "seed": self.seed, "prompt": self.prompt,
            "S_VQA": round(self.mean_vqa, 4), "S_DINO_patch": round(self.dino_patch, 4),
            "S_DINO_cls": round(self.dino_cls, 4), "joint": round(self.joint, 4),
            "failed": [sp.predicate.id for sp in self.failed],
            "rewrite_notes": self.rewrite_notes,
            "patch": asdict(self.patch) if self.patch else None,
            "rewrite_error": self.rewrite_error,
            "predicates": [sp.to_dict() for sp in self.scored],
        }


@dataclass
class InversionRun:
    config: EngineConfig
    models: dict = field(default_factory=dict)
    status: str = ""
    raw_graph: DSGSceneSchema | None = None
    graph: DSGSceneSchema | None = None
    predicates: list[Predicate] = field(default_factory=list)
    grounding: list[ScoredPredicate] = field(default_factory=list)
    dropped: list[ScoredPredicate] = field(default_factory=list)
    relocalized: list[dict] = field(default_factory=list)
    canonical_prompt: str = ""
    proxy_wh: tuple[int, int] | None = None
    iterations: list[IterationRecord] = field(default_factory=list)
    best_index: int | None = None
    stop_reason: str | None = None

    @property
    def best(self) -> IterationRecord | None:
        return None if self.best_index is None else self.iterations[self.best_index]

    @property
    def best_prompt(self) -> str:
        return self.best.prompt if self.best else self.canonical_prompt

    def to_dict(self) -> dict:
        best = self.best
        return {
            "config": asdict(self.config),
            "models": self.models,
            "stop_reason": self.stop_reason,
            "best_prompt": self.best_prompt,
            "best_iteration": best.t if best else None,
            "best_scores": ({"S_VQA": round(best.mean_vqa, 4), "S_DINO_patch": round(best.dino_patch, 4),
                             "joint": round(best.joint, 4)} if best else None),
            "canonical_prompt": self.canonical_prompt,
            "proxy_size": self.proxy_wh,
            "raw_graph": self.raw_graph.model_dump() if self.raw_graph else None,
            "graph": self.graph.model_dump() if self.graph else None,
            "predicates": [p.to_dict() for p in self.predicates],
            "grounding_dropped": [sp.to_dict() for sp in self.dropped],
            "grounding_relocalized": self.relocalized,
            "iterations": [it.to_dict() for it in self.iterations],
        }


def _relocalize(run: InversionRun, og: Image.Image, backends: Backends, cfg: EngineConfig) -> None:
    """A rejected grid cell would leave a wrong position in the prompt, so instead of only
    dropping it, ask the scorer about all 9 cells and move the node to the best-supported one."""
    alive = {n.id for n in run.graph.nodes}
    pending = [sp for sp in run.dropped if sp.predicate.kind == "position" and sp.predicate.ref["node"] in alive]
    nodes = [run.graph.node(sp.predicate.ref["node"]) for sp in pending]
    statements = [position_statement(n.entity, cell) for n in nodes for cell in GRID_ORDER]
    scores = backends.vqa_scores(og, statements, cfg.question_template)
    k = len(GRID_ORDER)
    for i, (sp, node) in enumerate(zip(pending, nodes)):
        cell_scores = scores[i * k:(i + 1) * k]
        best = max(range(k), key=cell_scores.__getitem__)
        if cell_scores[best] >= cfg.grounding_threshold and GRID_ORDER[best] != node.grid_cell:
            run.relocalized.append({"node": node.id, "entity": node.entity, "from": node.grid_cell,
                                    "to": GRID_ORDER[best], "p_yes": round(cell_scores[best], 4)})
            node.grid_cell = GRID_ORDER[best]
            run.dropped.remove(sp)
    if run.relocalized:
        dropped_ids = {sp.predicate.id for sp in run.dropped}
        run.predicates = [p for p in compile_predicates(run.graph) if p.id not in dropped_ids]


def iter_inversion(og_image: Image.Image, backends: Backends,
                   config: EngineConfig | None = None) -> Iterator[InversionRun]:
    cfg = config or EngineConfig()
    if cfg.max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    og = og_image.convert("RGB")
    run = InversionRun(config=cfg, models=dict(backends.meta))

    # ---- Module 1: constrained DSG extraction
    run.status = "Module 1 · extracting constrained scene graph"
    yield run
    run.raw_graph = backends.extract_dsg(og)
    predicates = compile_predicates(run.raw_graph)

    # ---- Anti-hallucination grounding: facts false on the OG itself are unfixable
    if cfg.ground:
        run.status = f"Module 2 · grounding {len(predicates)} predicates on the original image"
        yield run
        og_scores = backends.vqa_scores(og, [p.statement for p in predicates], cfg.question_template)
        run.grounding = apply_dependencies(predicates, og_scores, cfg.parent_gate)
        run.graph, run.predicates, run.dropped = prune_hallucinations(
            run.raw_graph, run.grounding, cfg.grounding_threshold)
        if any(sp.predicate.kind == "position" for sp in run.dropped):
            run.status = "Module 2 · re-localizing entities whose grid cell was rejected on the original"
            yield run
            _relocalize(run, og, backends, cfg)
    else:
        run.graph, run.predicates = run.raw_graph, predicates

    # ---- Module 2: canonical linearization
    prompt = run.canonical_prompt = linearize(run.graph)
    statements = [p.statement for p in run.predicates]
    og_emb = backends.embed(og)
    run.proxy_wh = width, height = proxy_size(og.size, cfg.proxy_long_side)

    for t in range(cfg.max_iters):
        seed = cfg.base_seed + t
        # ---- Module 3: proxy render
        run.status = f"t={t + 1}/{cfg.max_iters} · Module 3 · rendering proxy (seed {seed})"
        yield run
        proxy = backends.render(prompt, seed, width, height)

        # ---- Module 4: dual verification
        run.status = f"t={t + 1}/{cfg.max_iters} · Module 4 · scoring {len(statements)} predicates + DINO layout"
        yield run
        raw = backends.vqa_scores(proxy, statements, cfg.question_template)
        scored = apply_dependencies(run.predicates, raw, cfg.parent_gate)
        mean_vqa = sum(sp.effective for sp in scored) / max(1, len(scored))
        sim = dino_similarity(og_emb, backends.embed(proxy))
        joint = cfg.w_vqa * mean_vqa + cfg.w_dino * sim["patch"]
        # Root-cause failures only: blocked children are fixed by fixing their parent.
        failed = [sp for sp in scored if sp.raw < cfg.fail_threshold and not sp.blocked_by]

        rec = IterationRecord(t + 1, seed, prompt, proxy, scored, mean_vqa, sim["patch"], sim["cls"], joint, failed)
        run.iterations.append(rec)
        if run.best is None or joint > run.best.joint:
            run.best_index = len(run.iterations) - 1

        if mean_vqa >= cfg.vqa_gate and sim["patch"] >= cfg.dino_gate:
            run.stop_reason = "converged"
            break
        if t == cfg.max_iters - 1:
            run.stop_reason = "max_iters"
            break

        # ---- Targeted diff rewrite (only failed predicates)
        notes = [sp.predicate.statement for sp in failed]
        if sim["patch"] < cfg.dino_gate:
            notes.append(f"the overall layout: every shape occupies the same position, size, and proportion of the "
                         f"frame as in the original (layout similarity {sim['patch']:.2f} < {cfg.dino_gate:.2f})")
        rec.rewrite_notes = notes
        run.status = f"t={t + 1}/{cfg.max_iters} · targeted rewrite of {len(failed)} failed predicate(s)"
        yield run
        try:
            rec.patch = apply_patch(prompt, backends.rewrite(og, proxy, prompt, notes))
            prompt = rec.patch.prompt
        except Exception as err:  # keep the loop alive; next iteration re-renders with a new seed
            rec.rewrite_error = f"{type(err).__name__}: {err}"

    best = run.best
    run.status = (f"Done · {run.stop_reason} · best t={best.t}: S_VQA={best.mean_vqa:.3f}, "
                  f"S_DINO={best.dino_patch:.3f}, joint={best.joint:.3f}")
    yield run


def invert_image_to_verified_prompt(og_image: Image.Image, backends: Backends,
                                    config: EngineConfig | None = None) -> InversionRun:
    run = None
    for run in iter_inversion(og_image, backends, config):
        pass
    return run


def save_run(run: InversionRun, out_dir: str | Path, og_image: Image.Image | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    (out / "best_prompt.txt").write_text(run.best_prompt + "\n", encoding="utf-8")
    if og_image is not None:
        og_image.convert("RGB").save(out / "original.png")
    for it in run.iterations:
        it.proxy_image.save(out / f"proxy_t{it.t}_seed{it.seed}.png")
    return out
