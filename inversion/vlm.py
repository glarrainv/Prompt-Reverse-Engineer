"""VLM backends: Module 1 (DSG extraction), Module 4A (VQAScore), and the targeted rewriter.

`LocalQwenVLM` keeps a single set of Qwen3-VL weights in memory and serves all
three roles. `OpenAICompatVLM` targets any OpenAI-compatible endpoint
(OpenAI, OpenRouter, Together, Fireworks) for the no-local-GPU pathway.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from PIL import Image
from pydantic import BaseModel, ValidationError

from .schema import GRID_ORDER, DSGSceneSchema, RewritePatch

T = TypeVar("T", bound=BaseModel)

VQA_TEMPLATE = 'Does this image show "{}"? Please answer yes or no.'

# VQAScore modes:
#   "normalized" (default): P(yes-variants) / (P(yes-variants) + P(no-variants)). Qwen3-VL puts most of
#                 its mass on lowercase "yes", so the exact-"Yes" probability badly under-reads true facts
#                 (e.g. 0.009 for a fact answered "yes" with p=0.99) and is unstable across batching.
#   "yes":        the original VQAScore definition, P("Yes") over the full vocabulary.
VQA_MODES = ("normalized", "yes")
_YES, _NO = ("Yes", "yes", "YES"), ("No", "no", "NO")


def _combine(p_yes: float, p_no: float, p_exact_yes: float, mode: str) -> float:
    if mode == "yes":
        return p_exact_yes
    return p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.0

EXTRACTION_INSTRUCTION = f"""You are building a verifiable scene graph of this image so that a text-to-image model can reproduce it exactly without seeing it.

Procedure:
1. Partition the image into a 3x3 grid: {", ".join(GRID_ORDER)}.
2. List every distinct visible entity (shapes, objects, figures, marks, visible text) as a node anchored to the grid cell that contains its visual center.
3. For each node give up to 4 exact visual attributes (precise color name, geometry, size relative to the frame, orientation/facing direction, edge sharpness, texture) and up to 3 spatial or physical relations to other nodes, referenced by node id.
4. Describe the medium, surface, and optics as observable facts; describe the background; list the dominant color palette.
5. Optionally add a few extra atomic checks (counts, visible text) not already covered by nodes.

Rules:
- Describe only what is visible. No artist names, art movements, eras, or style buzzwords unless they are written in the image. No interpretation of meaning.
- Every attribute and relation must be independently verifiable by looking at the image.
- Entities are indefinite noun phrases ("a large circle"). Visible text is quoted verbatim inside an entity (a sign reading "PALACE").
- Attributes are bare phrases ("crimson red", "rotated 45 degrees"), never "key: value", and never restate the entity noun.
- A node's grid cell is where its center falls when the width and the height are each split into thirds.
- Left/right always mean the viewer's left/right."""

REWRITE_INSTRUCTION = """Image 1 is the ORIGINAL target image. Image 2 is a PROXY rendered by a text-to-image model from the prompt below.

PROMPT:
<<<
{prompt}
>>>

These visual facts are TRUE in the original image but FAILED verification in the proxy:
{failed}

Produce a minimal targeted diff to the prompt so that a text-to-image model will render the failed facts correctly.
Rules:
- Only touch text related to the failed facts; leave everything else exactly as it is.
- Each edit's "original" must be a short span (one clause or sentence) copied verbatim, character for character, from the prompt; "revised" replaces it. Make failed facts more explicit, concrete, and prominent (exact color names, position in the frame, size, count, orientation).
- Use "additions" (at most 3 short sentences) only for failed facts that no existing span covers.
- Describe only what is visible in image 1. No artist names, art movements, eras, or style buzzwords."""


def fit(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale (never upscale) so the longest side is at most `max_side`."""
    image = image.convert("RGB")
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return image
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def extract_json(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return text[start:end + 1]


class VLMBase:
    """Shared structured-output logic; subclasses implement `chat` and `vqa_scores`."""

    extract_tokens = 3072
    rewrite_tokens = 1024

    def chat(self, images: list[Image.Image], text: str, max_new_tokens: int = 512,
             json_schema: dict | None = None) -> str:
        raise NotImplementedError

    def vqa_scores(self, image: Image.Image, statements: list[str], template: str = VQA_TEMPLATE) -> list[float]:
        raise NotImplementedError

    def structured(self, images: list[Image.Image], instruction: str, schema_cls: type[T],
                   max_new_tokens: int, retries: int = 2) -> T:
        """Schema-constrained generation: schema in prompt (+ API json_schema), Pydantic
        validation, and repair retries that feed the validation error back to the model."""
        schema = schema_cls.model_json_schema()
        base = (f"{instruction}\n\nReturn ONLY one JSON object that validates against this JSON Schema "
                f"(no markdown, no commentary):\n{json.dumps(schema)}")
        prompt, last_err = base, None
        for _ in range(retries + 1):
            raw = self.chat(images, prompt, max_new_tokens=max_new_tokens, json_schema=schema)
            try:
                return schema_cls.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as err:
                last_err = err
                prompt = (f"{base}\n\nYour previous answer was invalid:\n{str(err)[:1500]}\n\n"
                          f"Previous answer (truncated):\n{raw[:3000]}\n\nReturn the corrected JSON object only.")
        raise RuntimeError(f"{schema_cls.__name__} extraction failed after {retries + 1} attempts: {last_err}")

    def extract_dsg(self, image: Image.Image) -> DSGSceneSchema:
        return self.structured([image], EXTRACTION_INSTRUCTION, DSGSceneSchema, self.extract_tokens)

    def rewrite(self, og_image: Image.Image, proxy_image: Image.Image, prompt: str,
                failed: list[str]) -> RewritePatch:
        bullet = "\n".join(f"- {f}" for f in failed)
        return self.structured([og_image, proxy_image], REWRITE_INSTRUCTION.format(prompt=prompt, failed=bullet),
                               RewritePatch, self.rewrite_tokens)


class LocalQwenVLM(VLMBase):
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", device: str | None = None,
                 max_side: int = 1024, vqa_max_side: int = 768, vqa_batch_size: int = 8,
                 vqa_mode: str = "normalized"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if vqa_mode not in VQA_MODES:
            raise ValueError(f"vqa_mode must be one of {VQA_MODES}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype="auto").to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"  # last position = next-token logits in a batch
        self.max_side, self.vqa_max_side, self.vqa_batch_size = max_side, vqa_max_side, vqa_batch_size
        self.vqa_mode = vqa_mode
        tok = self.processor.tokenizer
        self.yes_id = tok.convert_tokens_to_ids("Yes")
        self.yes_ids, self.no_ids = self._single_token_ids(tok, _YES), self._single_token_ids(tok, _NO)

    @staticmethod
    def _single_token_ids(tok, words) -> list[int]:
        ids = set()
        for w in words:
            for variant in (w, " " + w):
                enc = tok.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
        return sorted(ids)

    def chat(self, images, text, max_new_tokens=512, json_schema=None):
        content = [{"type": "image", "image": fit(img, self.max_side)} for img in images]
        content.append({"type": "text", "text": text})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def vqa_scores(self, image, statements, template=VQA_TEMPLATE):
        """VQAScore from the next-token distribution (see VQA_MODES)."""
        img = fit(image, self.vqa_max_side)
        scores: list[float] = []
        for i in range(0, len(statements), self.vqa_batch_size):
            batch = statements[i:i + self.vqa_batch_size]
            texts = [
                self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"},
                                                  {"type": "text", "text": template.replace("{}", s)}]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for s in batch
            ]
            inputs = self.processor(text=texts, images=[img] * len(batch), padding=True,
                                    return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, :].float()
            probs = self.torch.softmax(logits, dim=-1)
            p_yes = probs[:, self.yes_ids].sum(-1).tolist()
            p_no = probs[:, self.no_ids].sum(-1).tolist()
            p_exact = probs[:, self.yes_id].tolist()
            scores.extend(_combine(y, n, e, self.vqa_mode) for y, n, e in zip(p_yes, p_no, p_exact))
        return scores


class OpenAICompatVLM(VLMBase):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_side: int = 1024, workers: int = 8, vqa_mode: str = "normalized"):
        from openai import OpenAI

        if vqa_mode not in VQA_MODES:
            raise ValueError(f"vqa_mode must be one of {VQA_MODES}")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.max_side, self.workers, self.vqa_mode = model, max_side, workers, vqa_mode

    def _image_part(self, image: Image.Image) -> dict:
        buf = io.BytesIO()
        fit(image, self.max_side).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return {"type": "image_url", "image_url": {"url": url}}

    def _create(self, content: list[dict], **kwargs):
        return self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}], temperature=0, **kwargs)

    def chat(self, images, text, max_new_tokens=512, json_schema=None):
        import openai

        content = [self._image_part(img) for img in images] + [{"type": "text", "text": text}]
        kwargs = {"max_tokens": int(max_new_tokens)}
        if json_schema is not None:
            fmt = {"type": "json_schema", "json_schema": {"name": "output", "schema": json_schema}}
            try:
                return self._create(content, response_format=fmt, **kwargs).choices[0].message.content or ""
            except openai.BadRequestError:
                pass  # provider does not support json_schema; fall back to prompt + validation
        return self._create(content, **kwargs).choices[0].message.content or ""

    def vqa_scores(self, image, statements, template=VQA_TEMPLATE):
        part = self._image_part(image)

        def score(statement: str) -> float:
            resp = self._create([part, {"type": "text", "text": template.replace("{}", statement)}],
                                max_tokens=1, logprobs=True, top_logprobs=20)
            lp = resp.choices[0].logprobs
            if lp is None or not lp.content:
                raise RuntimeError(f"{self.model} returned no logprobs; VQAScore needs a logprob-capable scorer "
                                   "(e.g. gpt-4o) or the local backend.")
            top = lp.content[0].top_logprobs
            p_yes = sum(math.exp(t.logprob) for t in top if t.token.strip() in _YES)
            p_no = sum(math.exp(t.logprob) for t in top if t.token.strip() in _NO)
            p_exact = sum(math.exp(t.logprob) for t in top if t.token.strip() == "Yes")
            return _combine(p_yes, p_no, p_exact, self.vqa_mode)

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(score, statements))
