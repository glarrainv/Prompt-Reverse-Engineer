"""Model-free tests: schema validation, DSG compilation/dependencies, grounding, diff patches, and the loop."""

import json

import pytest
import torch
from PIL import Image

from inversion import (Backends, DSGSceneSchema, EngineConfig, RewritePatch, apply_dependencies, apply_patch,
                       compile_predicates, invert_image_to_verified_prompt, linearize, prune_hallucinations)
from inversion.dino import DinoEmbedding, medoid_index
from inversion.proxy import proxy_size
from inversion.vlm import extract_json

GRAPH = {
    "medium_and_optics": "flat gouache on matte paper; zero drop shadows; deep depth of field",
    "background": "a plain off-white background",
    "color_palette": ["cobalt blue", "crimson red", "Cobalt Blue", "ochre yellow"],
    "nodes": [
        {"id": "Red Circle", "grid_cell": "middle left", "entity": "a large circle",
         "attributes": ["color: crimson red", "sharp edges"],
         "relations": [{"predicate": "is left of", "target_id": "blue_square"},
                       {"predicate": "touches", "target_id": "ghost"}]},
        {"id": "blue_square", "grid_cell": "Mid-Right", "entity": "a small square",
         "attributes": ["cobalt blue", "rotated 45 degrees"], "relations": []},
        {"id": "sign", "grid_cell": "top center", "entity": 'a sign reading "PALACE"', "attributes": []},
    ],
    "extra_checks": ["exactly one circle"],
}


def graph() -> DSGSceneSchema:
    return DSGSceneSchema.model_validate(GRAPH)


def test_schema_normalizes_and_cleans():
    g = graph()
    assert [n.id for n in g.nodes] == ["red_circle", "blue_square", "sign"]
    assert [n.grid_cell for n in g.nodes] == ["Mid-Left", "Mid-Right", "Top-Center"]
    assert g.color_palette == ["cobalt blue", "crimson red", "ochre yellow"]  # case-insensitive dedupe
    assert g.nodes[0].attributes == ["crimson red", "sharp edges"]  # 'key: value' prefix stripped
    assert g.medium_and_optics == ["flat gouache on matte paper", "zero drop shadows", "deep depth of field"]
    commas = DSGSceneSchema.model_validate({**GRAPH, "medium_and_optics": "a flat, matte image"})
    assert commas.medium_and_optics == ["a flat, matte image"]  # commas never split a facet
    # Dangling relation target is dropped.
    assert [r.target_id for r in g.nodes[0].relations] == ["blue_square"]


def test_schema_rejects_bad_grid_and_empty_nodes():
    with pytest.raises(Exception):
        DSGSceneSchema.model_validate({**GRAPH, "nodes": [{**GRAPH["nodes"][0], "grid_cell": "upstairs"}]})
    with pytest.raises(Exception):
        DSGSceneSchema.model_validate({**GRAPH, "nodes": []})


def test_compile_is_dependency_ordered():
    preds = compile_predicates(graph())
    kinds = [p.kind for p in preds]
    order = ["global", "entity", "attribute", "position", "relation"]
    assert kinds == sorted(kinds, key=order.index)
    rel = next(p for p in preds if p.kind == "relation")
    assert rel.parents == ["entity:red_circle", "entity:blue_square"]
    assert rel.statement == "a large circle that is left of the small square"
    assert len({p.id for p in preds}) == len(preds)


def test_dependencies_block_children_of_absent_parent():
    preds = compile_predicates(graph())
    raw = [0.1 if p.id == "entity:blue_square" else 0.95 for p in preds]
    scored = {sp.predicate.id: sp for sp in apply_dependencies(preds, raw, parent_gate=0.5)}
    assert scored["rel:red_circle:is_left_of:blue_square"].effective == 0.0
    assert scored["rel:red_circle:is_left_of:blue_square"].blocked_by == ["entity:blue_square"]
    assert scored["pos:blue_square"].effective == 0.0
    assert scored["pos:red_circle"].effective == 0.95


def test_grounding_prunes_hallucinated_content_with_stable_ids():
    g = graph()
    preds = compile_predicates(g)
    bad = {"attr:red_circle:crimson_red", "palette:ochre_yellow", "entity:sign", "medium:deep_depth_of_field"}
    raw = [0.1 if p.id in bad else 0.9 for p in preds]
    pruned, surviving, dropped = prune_hallucinations(g, apply_dependencies(preds, raw, 0.5), 0.5)
    ids = {p.id for p in surviving}
    assert "attr:red_circle:sharp_edges" in ids  # sibling of the dropped attribute survives
    assert not ids & bad
    assert "pos:sign" not in ids  # child of dropped entity
    assert [n.id for n in pruned.nodes] == ["red_circle", "blue_square"]
    assert pruned.nodes[0].attributes == ["sharp edges"]
    assert pruned.color_palette == ["cobalt blue", "crimson red"]
    assert pruned.medium_and_optics == ["flat gouache on matte paper", "zero drop shadows"]  # unverified facet leaves
    assert {sp.predicate.id for sp in dropped} >= bad


def test_grounding_never_empties_graph():
    g = graph()
    preds = compile_predicates(g)
    pruned, surviving, _ = prune_hallucinations(g, apply_dependencies(preds, [0.0] * len(preds), 0.5), 0.5)
    assert len(pruned.nodes) == 3


def test_linearize_is_deterministic_and_grid_ordered():
    text = linearize(graph())
    assert text == linearize(graph())
    assert text.startswith("Flat gouache on matte paper")
    assert text.index("top-center") < text.index("middle-left") < text.index("middle-right")
    assert "it is left of the small square" in text


def test_apply_patch_only_touches_verbatim_spans():
    prompt = "A red circle on the left. A blue square on the right."
    patch = RewritePatch(edits=[{"original": "A red circle", "revised": "A large crimson-red circle"},
                                {"original": "nonexistent span", "revised": "x"}],
                         additions=["A sign reading PALACE sits top center"])
    res = apply_patch(prompt, patch)
    assert res.prompt == ("A large crimson-red circle on the left. A blue square on the right. "
                          "A sign reading PALACE sits top center.")
    assert len(res.applied) == 1 and len(res.skipped) == 1


def test_grounding_removes_rejected_medium_facet_from_prompt():
    # Pivot 1: an unverified optics claim (the Figure 1 "soft bokeh" failure) must not reach the T2I prompt.
    g = DSGSceneSchema.model_validate({**GRAPH, "medium_and_optics": "flat gouache; shallow depth of field with soft bokeh"})
    preds = compile_predicates(g)
    raw = [0.1 if "bokeh" in p.statement else 0.9 for p in preds]
    pruned, _, _ = prune_hallucinations(g, apply_dependencies(preds, raw, 0.5), 0.5)
    assert pruned.medium_and_optics == ["flat gouache"]
    assert "bokeh" not in linearize(pruned)


@pytest.mark.parametrize("text, expected", [
    ("a vintage orange bus", {"vintage"}),
    ("1970s cinematic film look", {"1970s", "cinematic"}),
    ("the '70s, in the style of Bauhaus", {"'70s", "in the style of", "bauhaus"}),
    ("a Cubist, Surrealist and minimalist composition", {"cubist", "surrealist", "minimalist"}),
    ("a red circle rotated 45 degrees, sharp edges, deep depth of field", set()),
    ("modernity and cubes", set()),
])
def test_buzzword_lint(text, expected):
    from inversion.compiler import buzzwords
    assert buzzwords(text) == expected


def test_apply_patch_rejects_injected_buzzwords():
    prompt = "Flat gouache on paper. A red circle on the left. A sign reading \"RETRO\" at the top."
    patch = RewritePatch(edits=[{"original": "A red circle", "revised": "A vintage red circle"},
                                {"original": "on the left", "revised": "on the far left, retro"}],
                         additions=["Cinematic lighting", "A small blue square at the bottom right"])
    res = apply_patch(prompt, patch)
    assert "vintage" not in res.prompt and "Cinematic" not in res.prompt
    assert "on the far left, retro" in res.prompt  # 'retro' already present (visible text), so allowed
    assert res.added == ["A small blue square at the bottom right."]
    assert sum("unverified" in s["reason"] for s in res.skipped) == 2


def test_choose_gpu_size():
    from inversion.backends import choose_gpu_size
    assert choose_gpu_size("auto", int(51.6e9)) == "xlarge"  # Qwen3-VL-8B + FLUX bf16 + DINOv2: OOM on 'large'
    assert choose_gpu_size("auto", int(39.7e9)) == "large"  # with PROXY_FP8=1
    assert choose_gpu_size("large", int(80e9)) == "large"  # explicit override wins
    with pytest.raises(ValueError):
        choose_gpu_size("huge", 0)


def test_resident_bytes_counts_shared_weights_once():
    from types import SimpleNamespace
    from inversion.backends import Components
    vlm = SimpleNamespace(model=torch.nn.Linear(10, 10))  # 110 fp32 params = 440 bytes
    pipe = SimpleNamespace(components={"transformer": torch.nn.Linear(4, 4, dtype=torch.bfloat16), "tok": "x"})
    comps = Components(vlm=vlm, scorer=vlm, renderer=SimpleNamespace(pipe=pipe),
                       dino=SimpleNamespace(model=torch.nn.Embedding(2, 3)))
    assert comps.resident_bytes() == 440 + 20 * 2 + 6 * 4


def test_proxy_defaults_and_rejections():
    from inversion.proxy import proxy_defaults
    assert proxy_defaults("FluxPipeline") == (4, 0.0)  # FLUX.1-schnell
    assert proxy_defaults("Flux2KleinPipeline", True) == (4, 1.0)  # FLUX.2-klein-4B (model_index: is_distilled true)
    for distilled in (None, False):  # klein *base* checkpoints have no is_distilled key
        with pytest.raises(ValueError, match="not step-distilled"):
            proxy_defaults("Flux2KleinPipeline", distilled)
    with pytest.raises(ValueError, match="supported"):
        proxy_defaults("StableDiffusionXLPipeline")


def test_run_report_records_the_models_used():
    world = StubWorld()
    backends = world.backends()
    backends.meta = {"proxy_model": "black-forest-labs/FLUX.2-klein-4B", "proxy_steps": 4, "proxy_guidance_scale": 1.0}
    report = invert_image_to_verified_prompt(world.og, backends, EngineConfig(max_iters=3)).to_dict()
    assert report["models"] == backends.meta
    assert invert_image_to_verified_prompt(world.og, world.backends(), EngineConfig(max_iters=1)).to_dict()["models"] == {}


def test_extract_json_handles_fences_and_think():
    raw = "<think>hmm</think>```json\n{\"a\": {\"b\": 1}}\n```"
    assert json.loads(extract_json(raw)) == {"a": {"b": 1}}


def test_proxy_size_keeps_aspect_and_multiple_of_16():
    w, h = proxy_size((3000, 2000))
    assert (w, h) == (1024, 688) and w % 16 == 0 and h % 16 == 0


def test_medoid():
    def emb(v):
        return DinoEmbedding(cls=torch.tensor(v, dtype=torch.float32), patches=torch.tensor([v], dtype=torch.float32))
    idx, _ = medoid_index([emb([1, 0]), emb([1, 0.1]), emb([1, 0.2])])
    assert idx == 1


# ---------------------------------------------------------------- full loop with stub backends

class StubWorld:
    """Proxy 'renders' encode the prompt; the scorer says yes iff the statement's key phrase is in it."""

    def __init__(self, fix_on_rewrite=True):
        self.fix_on_rewrite = fix_on_rewrite
        self.rendered: dict[int, str] = {}
        self.rewrites = 0
        self.og = Image.new("RGB", (640, 480), "white")

    def extract(self, image):
        return graph()

    def vqa(self, image, statements, template):
        assert "{}" in template
        if id(image) not in self.rendered:  # OG grounding: all true except the hallucinated palette color
            return [0.1 if "ochre" in s else 0.95 for s in statements]
        prompt = self.rendered[id(image)]
        # 18 predicates: one failure at 0.0 pulls the mean to ~0.87 (< 0.90 gate).
        return [0.92 if ("rotated" not in s or "rotated 45 degrees, clearly" in prompt) else 0.0 for s in statements]

    def rewrite(self, og, proxy, prompt, failed):
        self.rewrites += 1
        assert any("rotated 45 degrees" in f for f in failed)
        if not self.fix_on_rewrite:
            return RewritePatch(edits=[{"original": "not in prompt", "revised": "x"}])
        return RewritePatch(edits=[{"original": "rotated 45 degrees", "revised": "rotated 45 degrees, clearly"}])

    def render(self, prompt, seed, w, h):
        img = Image.new("RGB", (w, h), "white")
        self.rendered[id(img)] = prompt
        return img

    def embed(self, image):
        return DinoEmbedding(cls=torch.ones(4), patches=torch.ones(9, 4))

    def backends(self):
        return Backends(self.extract, self.vqa, self.rewrite, self.render, self.embed)


def test_loop_converges_after_targeted_rewrite():
    world = StubWorld()
    run = invert_image_to_verified_prompt(world.og, world.backends(), EngineConfig(max_iters=5))
    assert run.stop_reason == "converged"
    assert len(run.iterations) == 2 and world.rewrites == 1
    assert "ochre" not in run.canonical_prompt  # pruned during grounding
    assert run.best.t == 2 and "clearly" in run.best_prompt
    assert run.best.mean_vqa >= 0.9 and run.best.dino_patch == pytest.approx(1.0)
    json.dumps(run.to_dict())  # report is serializable


@pytest.mark.parametrize("sign_really_at, expect_move", [("bottom-center", True), (None, False)])
def test_grounding_relocalizes_rejected_grid_cell(sign_really_at, expect_move):
    world = StubWorld()
    base_vqa = world.vqa

    def vqa(image, statements, template):
        scores = base_vqa(image, statements, template)
        if id(image) in world.rendered:
            return scores
        # On the original, the sign is NOT top-center (the extractor's claim) but at `sign_really_at`.
        return [(0.95 if sign_really_at and sign_really_at in s else 0.05)
                if ("PALACE" in s and "located in" in s) else sc for s, sc in zip(statements, scores)]

    world.vqa = vqa
    run = invert_image_to_verified_prompt(world.og, world.backends(), EngineConfig(max_iters=1))
    if expect_move:
        assert run.relocalized == [{"node": "sign", "entity": 'a sign reading "PALACE"', "from": "Top-Center",
                                    "to": "Bot-Center", "p_yes": 0.95}]
        assert 'In the bottom-center of the frame, a sign reading "PALACE"' in run.canonical_prompt
        assert "pos:sign" in {p.id for p in run.predicates}
        assert "bottom-center" in next(p for p in run.predicates if p.id == "pos:sign").statement
    else:
        assert run.relocalized == []
        assert "pos:sign" in {sp.predicate.id for sp in run.dropped}
        assert "pos:sign" not in {p.id for p in run.predicates}


def test_loop_hits_max_iters_and_keeps_best():
    world = StubWorld(fix_on_rewrite=False)
    run = invert_image_to_verified_prompt(world.og, world.backends(), EngineConfig(max_iters=3))
    assert run.stop_reason == "max_iters"
    assert len(run.iterations) == 3 and world.rewrites == 2  # no rewrite after the final render
    assert run.iterations[0].patch.skipped and run.best_prompt == run.canonical_prompt
