"""Module 2: dependency predicate compiler, canonical linearizer, and diff applier.

Predicates follow the Davidsonian Scene Graph (DSG) dependency rule: an
attribute, position, or relation question is only credited when every parent
entity it depends on is itself judged present. Children of a failed parent
are "blocked" and score 0, so a relation is never rewarded when its objects
are missing.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field

from .schema import GRID_ORDER, GRID_PHRASE, DSGSceneSchema, RewritePatch

KIND_ORDER = ("global", "entity", "attribute", "position", "relation")


@dataclass
class Predicate:
    id: str
    kind: str  # one of KIND_ORDER
    statement: str  # noun phrase / clause inserted into the VQA question template
    parents: list[str] = field(default_factory=list)
    # Where in the graph this fact came from, used to prune hallucinations.
    ref: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredPredicate:
    predicate: Predicate
    raw: float  # P("Yes") from the scorer
    effective: float  # raw, or 0.0 when a parent failed
    blocked_by: list[str] = field(default_factory=list)

    def passed(self, threshold: float) -> bool:
        return self.effective >= threshold

    def to_dict(self) -> dict:
        return {
            "id": self.predicate.id,
            "kind": self.predicate.kind,
            "statement": self.predicate.statement,
            "parents": self.predicate.parents,
            "p_yes": round(self.raw, 4),
            "effective": round(self.effective, 4),
            "blocked_by": self.blocked_by,
        }


# --------------------------------------------------------------------------- text helpers

_ARTICLE = re.compile(r"^(a|an|one|some)\s+", re.IGNORECASE)


def definite(np: str) -> str:
    """'a red circle' -> 'the red circle' (leaves already-definite phrases alone)."""
    text = np.strip()
    if re.match(r"^the\s", text, re.IGNORECASE):
        return text
    if _ARTICLE.match(text):
        return _ARTICLE.sub("the ", text, count=1)
    return f"the {text}"


def _sentence(text: str) -> str:
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def _join_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------- compiler

def position_statement(entity: str, grid_cell: str) -> str:
    return f"{entity} located in the {GRID_PHRASE[grid_cell]} region of the image"


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def compile_predicates(graph: DSGSceneSchema) -> list[Predicate]:
    """Compile the scene graph into dependency-ordered binary VQA predicates.

    Ids are keyed on content (not list position) so they stay stable when the
    graph is pruned or recompiled.
    """
    preds: list[Predicate] = []

    for facet in graph.medium_and_optics:
        preds.append(Predicate(f"medium:{_key(facet)}", "global", facet, ref={"field": "medium"}))
    if graph.background:
        preds.append(Predicate("background", "global", graph.background, ref={"field": "background"}))
    for color in graph.color_palette:
        preds.append(Predicate(f"palette:{_key(color)}", "global", f"the color {color}",
                               ref={"field": "palette", "value": color}))
    for check in graph.extra_checks:
        preds.append(Predicate(f"check:{_key(check)}", "global", check,
                               ref={"field": "extra_checks", "value": check}))

    entity_id = {n.id: f"entity:{n.id}" for n in graph.nodes}
    for n in graph.nodes:
        preds.append(Predicate(entity_id[n.id], "entity", n.entity, ref={"node": n.id}))
    for n in graph.nodes:
        for attr in n.attributes:
            preds.append(Predicate(f"attr:{n.id}:{_key(attr)}", "attribute", f"{n.entity} that is {attr}",
                                   parents=[entity_id[n.id]], ref={"node": n.id, "attribute": attr}))
    for n in graph.nodes:
        preds.append(Predicate(f"pos:{n.id}", "position", position_statement(n.entity, n.grid_cell),
                               parents=[entity_id[n.id]], ref={"node": n.id, "grid_cell": n.grid_cell}))
    for n in graph.nodes:
        for rel in n.relations:
            target = graph.node(rel.target_id)
            preds.append(Predicate(
                f"rel:{n.id}:{_key(rel.predicate)}:{target.id}", "relation",
                f"{n.entity} that {rel.predicate} {definite(target.entity)}",
                parents=[entity_id[n.id], entity_id[target.id]],
                ref={"node": n.id, "relation": rel.model_dump()},
            ))
    unique: dict[str, Predicate] = {}
    for p in preds:
        unique.setdefault(p.id, p)
    return list(unique.values())


def apply_dependencies(predicates: list[Predicate], raw_scores: list[float],
                       parent_gate: float) -> list[ScoredPredicate]:
    """DSG rule: a child's score is zeroed when any parent scores below `parent_gate`."""
    raw = {p.id: s for p, s in zip(predicates, raw_scores)}
    out = []
    for p, s in zip(predicates, raw_scores):
        blocked = [par for par in p.parents if raw.get(par, 0.0) < parent_gate]
        out.append(ScoredPredicate(p, float(s), 0.0 if blocked else float(s), blocked))
    return out


def prune_hallucinations(graph: DSGSceneSchema, scored: list[ScoredPredicate],
                         threshold: float) -> tuple[DSGSceneSchema, list[Predicate], list[ScoredPredicate]]:
    """Ground the graph against the OG image itself.

    Facts the scorer rejects *on the original image* can never be satisfied by a
    proxy render, so they are removed before the loop starts. Returns the pruned
    graph, the surviving predicates, and the dropped (scored) predicates.
    """
    g = copy.deepcopy(graph)
    dropped = [sp for sp in scored if sp.effective < threshold]
    dropped_ids = {sp.predicate.id for sp in dropped}

    dead_nodes = {sp.predicate.ref["node"] for sp in dropped if sp.predicate.kind == "entity"}
    if len(dead_nodes) == len(g.nodes):  # never prune the graph to nothing
        dead_nodes = set()
        dropped = [sp for sp in dropped if sp.predicate.kind != "entity"]
        dropped_ids = {sp.predicate.id for sp in dropped}

    bad_attrs = {(sp.predicate.ref["node"], sp.predicate.ref["attribute"])
                 for sp in dropped if sp.predicate.kind == "attribute"}
    bad_rels = {(sp.predicate.ref["node"], sp.predicate.ref["relation"]["predicate"],
                 sp.predicate.ref["relation"]["target_id"])
                for sp in dropped if sp.predicate.kind == "relation"}
    bad_colors = {sp.predicate.ref["value"] for sp in dropped if sp.predicate.ref.get("field") == "palette"}
    bad_checks = {sp.predicate.ref["value"] for sp in dropped if sp.predicate.ref.get("field") == "extra_checks"}
    bad_medium = {sp.predicate.statement for sp in dropped if sp.predicate.ref.get("field") == "medium"}

    g.nodes = [n for n in g.nodes if n.id not in dead_nodes]
    for n in g.nodes:
        n.attributes = [a for a in n.attributes if (n.id, a) not in bad_attrs]
        n.relations = [r for r in n.relations
                       if r.target_id not in dead_nodes and (n.id, r.predicate, r.target_id) not in bad_rels]
    g.color_palette = [c for c in g.color_palette if c not in bad_colors]
    g.extra_checks = [c for c in g.extra_checks if c not in bad_checks]
    # Medium/optics facets rejected on the original are unverified style claims (e.g. an
    # injected "shallow depth of field with soft bokeh"), so they leave the prompt too.
    g.medium_and_optics = [m for m in g.medium_and_optics if m not in bad_medium]
    # Background and grid position stay in the prompt (they are still the best available
    # description; positions are re-localized by the engine); only their predicates are dropped.

    surviving = [p for p in compile_predicates(g) if p.id not in dropped_ids]
    return g, surviving, dropped


# --------------------------------------------------------------------------- linearizer

def linearize(graph: DSGSceneSchema) -> str:
    """Deterministically serialize the graph into the canonical prompt P_k^(0)."""
    parts = [_sentence(", ".join(graph.medium_and_optics))]
    if graph.background:
        parts.append(_sentence(f"Background: {graph.background}"))
    if graph.color_palette:
        parts.append(_sentence(f"Color palette limited to {_join_list(graph.color_palette)}"))

    by_id = {n.id: n for n in graph.nodes}
    for cell in GRID_ORDER:
        for n in (n for n in graph.nodes if n.grid_cell == cell):
            desc = n.entity
            if n.attributes:
                desc += f" ({'; '.join(n.attributes)})"
            text = f"In the {GRID_PHRASE[cell]} of the frame, {desc}"
            rels = [f"{r.predicate} {definite(by_id[r.target_id].entity)}" for r in n.relations]
            if rels:
                text += f"; it {_join_list(rels)}"
            parts.append(_sentence(text))
    for check in graph.extra_checks:
        parts.append(_sentence(f"The image shows {check}"))
    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------- targeted diff

@dataclass
class PatchResult:
    prompt: str
    applied: list[dict]
    skipped: list[dict]
    added: list[str]


# Era / art-historical / aesthetic-hype vocabulary. Rewrites are not re-verified on the original,
# so a term from this list may only enter the prompt if the (grounded) prompt already contains it.
_BUZZWORDS = re.compile(
    r"(?<![\w'])(?:"
    r"(?:1[0-9]|20)\d0s|'\d0s|\d{1,2}(?:st|nd|rd|th)[- ]century|mid[- ]century|"
    r"vintage|retro|old[- ]fashioned|antique|nostalgic|victorian|edwardian|"
    r"cinematic|film still|masterpiece|award[- ]winning|trending|artstation|hyper[- ]?realistic|photorealistic|"
    r"8k|4k|ultra[- ]detailed|epic|in the style of|style of|"
    r"art deco|art nouveau|bauhaus|de stijl|dada(?:ism|ist)?|pop art|op art|renaissance|baroque|rococo|"
    r"(?:cub|surreal|expression|impression|constructiv|supremat|futur|modern|minimal|fauv|orph|vortic)"
    r"(?:ism|ist|istic)"
    r")(?!\w)",
    re.IGNORECASE,
)


def buzzwords(text: str) -> set[str]:
    """Lower-cased era/style buzzwords found in `text`."""
    return {m.group(0).lower() for m in _BUZZWORDS.finditer(text)}


def apply_patch(prompt: str, patch: RewritePatch) -> PatchResult:
    """Apply a targeted diff; spans not found verbatim are skipped, never guessed.

    Edits and additions that would introduce an era/style buzzword absent from the
    current prompt are skipped (unverified on the original image).
    """
    new, applied, skipped = prompt, [], []
    for edit in patch.edits:
        orig, rev = edit.original.strip(), edit.revised.strip()
        injected = buzzwords(rev) - buzzwords(new)
        if not orig or orig == rev:
            skipped.append({**edit.model_dump(), "reason": "empty or unchanged"})
        elif injected:
            skipped.append({**edit.model_dump(), "reason": f"unverified style/era terms: {sorted(injected)}"})
        elif orig in new:
            new = new.replace(orig, rev, 1)
            applied.append(edit.model_dump())
        else:
            skipped.append({**edit.model_dump(), "reason": "original span not found verbatim"})
    added = []
    for sentence in (_sentence(a) for a in patch.additions):
        if not sentence or sentence in new:
            continue
        injected = buzzwords(sentence) - buzzwords(new)
        if injected:
            skipped.append({"addition": sentence, "reason": f"unverified style/era terms: {sorted(injected)}"})
        else:
            added.append(sentence)
    if added:
        new = new.rstrip() + " " + " ".join(added)
    return PatchResult(new, applied, skipped, added)
