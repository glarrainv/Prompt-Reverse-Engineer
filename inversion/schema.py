"""Module 1 schema: typed Davidsonian-style scene graph extracted from the OG image.

The VLM must emit JSON that validates against `DSGSceneSchema`. Validators are
lenient on surface form (grid-cell spelling, list lengths) but strict on
structure, so the downstream compiler always receives a well-formed graph.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

GridCell = Literal[
    "Top-Left", "Top-Center", "Top-Right",
    "Mid-Left", "Center", "Mid-Right",
    "Bot-Left", "Bot-Center", "Bot-Right",
]
GRID_ORDER: tuple[str, ...] = GridCell.__args__  # reading order: left-to-right, top-to-bottom

GRID_PHRASE = {
    "Top-Left": "top-left", "Top-Center": "top-center", "Top-Right": "top-right",
    "Mid-Left": "middle-left", "Center": "center", "Mid-Right": "middle-right",
    "Bot-Left": "bottom-left", "Bot-Center": "bottom-center", "Bot-Right": "bottom-right",
}

MAX_NODES = 12
MAX_ATTRIBUTES = 4
MAX_RELATIONS = 3
MAX_PALETTE = 8
MAX_EXTRA_CHECKS = 6


def normalize_grid_cell(value: str) -> str:
    """Map spellings like 'top left', 'middle-right', 'bottom center' onto GridCell."""
    s = re.sub(r"[\s_]+", "-", str(value).strip().lower())
    s = s.replace("middle", "mid").replace("bottom", "bot").replace("centre", "center")
    if s in {"center", "mid-center", "center-center", "mid-mid", "mid"}:
        return "Center"
    for cell in GRID_ORDER:
        if s == cell.lower():
            return cell
    raise ValueError(f"unknown grid cell {value!r}; use one of {', '.join(GRID_ORDER)}")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "node"


_KEY_PREFIX = re.compile(r"^[A-Za-z][A-Za-z /_-]{1,24}:\s+")  # 'color: dark red' -> 'dark red'


def _clean_list(items: list[str], limit: int, strip_keys: bool = False) -> list[str]:
    if isinstance(items, str):
        items = [items]
    seen, out = set(), []
    for item in items:
        text = " ".join(str(item).split()).strip(" .;,")
        if strip_keys:
            text = _KEY_PREFIX.sub("", text)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out[:limit]


class Relation(BaseModel):
    predicate: str = Field(
        description="Spatial or physical relation phrased so '<subject> <predicate> <target>' reads as a sentence, "
        "e.g. 'is left of', 'overlaps', 'is above', 'is partially behind', 'touches'."
    )
    target_id: str = Field(description="`id` of another node in this graph.")


class SpatialNode(BaseModel):
    id: str = Field(description="Unique snake_case identifier, e.g. 'red_circle'.")
    grid_cell: GridCell = Field(
        description="Cell of the 3x3 grid holding the node's visual center. One of: " + ", ".join(GRID_ORDER)
    )
    entity: str = Field(
        description="Objective visual form/object as an indefinite noun phrase (e.g. 'a large circle', "
        "'a sign reading \"PALACE\"'), without unverified era or style buzzwords."
    )
    attributes: list[str] = Field(
        default_factory=list,
        description=f"Up to {MAX_ATTRIBUTES} exact visual properties as bare phrases (e.g. 'crimson red', "
        "'sharp edges', 'rotated 45 degrees', 'about one third of the frame width'), never 'key: value'. "
        "Do not repeat what the entity noun already says.",
    )
    relations: list[Relation] = Field(
        default_factory=list,
        description=f"Up to {MAX_RELATIONS} spatial/physical relations to other nodes.",
    )

    @field_validator("grid_cell", mode="before")
    @classmethod
    def _grid(cls, v):
        return normalize_grid_cell(v)

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, v):
        return _slug(str(v))

    @field_validator("entity", mode="before")
    @classmethod
    def _entity(cls, v):
        text = " ".join(str(v).split()).strip(" .")
        if not text:
            raise ValueError("entity must be non-empty")
        return text

    @field_validator("attributes", mode="before")
    @classmethod
    def _attrs(cls, v):
        return _clean_list(v or [], MAX_ATTRIBUTES, strip_keys=True)

    @field_validator("relations", mode="after")
    @classmethod
    def _rels(cls, v):
        return v[:MAX_RELATIONS]


MAX_MEDIUM_FACETS = 4


class DSGSceneSchema(BaseModel):
    medium_and_optics: list[str] = Field(
        description=f"1-{MAX_MEDIUM_FACETS} short, separately verifiable facets of the physical medium, surface "
        "and optics, e.g. ['flat gouache on matte paper', 'zero drop shadows', 'deep depth of field']. "
        "Observable facts only; no guesses about how the image was made."
    )
    background: str = Field(description="The background/ground as a noun phrase, e.g. 'a plain off-white background'.")
    color_palette: list[str] = Field(description=f"3-{MAX_PALETTE} dominant colors as precise color names.")
    nodes: list[SpatialNode] = Field(
        description=f"1-{MAX_NODES} distinct visible entities, each anchored to one 3x3 grid cell."
    )
    extra_checks: list[str] = Field(
        default_factory=list,
        description=f"0-{MAX_EXTRA_CHECKS} additional atomic, binary-verifiable visual facts not captured by nodes "
        "(e.g. counts: 'exactly three circles'; visible text). Noun-phrase or short-clause form.",
    )

    @field_validator("medium_and_optics", mode="before")
    @classmethod
    def _medium(cls, v):
        # A plain string is split on semicolons only: commas often join adjectives ("a flat, matte image").
        if isinstance(v, str):
            v = v.split(";")
        out = _clean_list(v or [], MAX_MEDIUM_FACETS)
        if not out:
            raise ValueError("medium_and_optics must contain at least one facet")
        return out

    @field_validator("background", mode="before")
    @classmethod
    def _text(cls, v):
        return " ".join(str(v).split()).strip(" .")

    @field_validator("color_palette", mode="before")
    @classmethod
    def _palette(cls, v):
        return _clean_list(v or [], MAX_PALETTE)

    @field_validator("extra_checks", mode="before")
    @classmethod
    def _extra(cls, v):
        return _clean_list(v or [], MAX_EXTRA_CHECKS)

    @model_validator(mode="after")
    def _graph_integrity(self):
        if not self.nodes:
            raise ValueError("nodes must contain at least one entity")
        self.nodes = self.nodes[:MAX_NODES]
        # Make ids unique, then drop relations whose target is missing or self-referential.
        seen: dict[str, int] = {}
        for node in self.nodes:
            base = node.id
            if base in seen:
                seen[base] += 1
                node.id = f"{base}_{seen[base]}"
            else:
                seen[base] = 1
        ids = {n.id for n in self.nodes}
        for node in self.nodes:
            node.relations = [r for r in node.relations if r.target_id in ids and r.target_id != node.id]
        return self

    def node(self, node_id: str) -> SpatialNode:
        return next(n for n in self.nodes if n.id == node_id)


class TextEdit(BaseModel):
    original: str = Field(description="An exact, verbatim substring of the current prompt to replace.")
    revised: str = Field(description="Replacement text for that span.")


class RewritePatch(BaseModel):
    """Targeted diff: only spans tied to failed predicates are touched."""

    edits: list[TextEdit] = Field(default_factory=list, description="At most 6 span replacements.")
    additions: list[str] = Field(
        default_factory=list,
        description="At most 3 short new sentences for failed facts that no existing span covers.",
    )

    @field_validator("edits", mode="after")
    @classmethod
    def _cap_edits(cls, v):
        return v[:6]

    @field_validator("additions", mode="before")
    @classmethod
    def _cap_additions(cls, v):
        return _clean_list(v or [], 3)
