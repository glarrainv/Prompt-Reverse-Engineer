import html
import json
import tempfile
from pathlib import Path

import gradio as gr
import spaces

from inversion import EngineConfig, iter_inversion
from inversion.backends import BackendSettings, choose_gpu_size, load_components
from inversion.engine import Backends

SETTINGS = BackendSettings()
C = load_components(SETTINGS)

# ZeroGPU copies ALL resident weights onto the GPU at the start of every @spaces.GPU call, so the slice
# must hold all of them plus activations. With the default FLUX.2-klein-4B the weights are ~33.9 GB and fit a
# "large" slice (50.87 GB). Qwen3-VL-8B + FLUX.1-schnell + DINOv2 in bf16 is ~51.6 GB, which fails with a CUDA
# OOM before any code runs, so choose_gpu_size() requests "xlarge" for it (or set PROXY_FP8=1 to fit "large").
RESIDENT_BYTES = C.resident_bytes()
GPU_SIZE = choose_gpu_size(SETTINGS.zerogpu_size, RESIDENT_BYTES)
print(f"[startup] resident model weights: {RESIDENT_BYTES / 1e9:.1f} GB -> ZeroGPU size '{GPU_SIZE}'", flush=True)


# Each model call gets its own (short) ZeroGPU allocation instead of one long one
# for the whole loop. Outside Spaces/ZeroGPU, @spaces.GPU is a no-op.
@spaces.GPU(duration=180, size=GPU_SIZE)
def gpu_extract(image):
    return C.vlm.extract_dsg(image)


@spaces.GPU(duration=90, size=GPU_SIZE)
def gpu_vqa(image, statements, template):
    return C.scorer.vqa_scores(image, statements, template)


@spaces.GPU(duration=120, size=GPU_SIZE)
def gpu_rewrite(og, proxy, prompt, failed):
    return C.vlm.rewrite(og, proxy, prompt, failed)


@spaces.GPU(duration=60, size=GPU_SIZE)
def gpu_render(prompt, seed, width, height):
    return C.renderer(prompt, seed, width, height)


@spaces.GPU(duration=30, size=GPU_SIZE)
def gpu_embed(image):
    return C.dino.embed(image)


BACKENDS = Backends(extract_dsg=gpu_extract, vqa_scores=gpu_vqa, rewrite=gpu_rewrite,
                    render=gpu_render, embed=gpu_embed, meta=C.info)


# --------------------------------------------------------------------------- look

def _short(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


MODEL_CHIP = " · ".join([
    _short(SETTINGS.vlm_model_id if SETTINGS.vlm_backend == "local" else SETTINGS.vlm_api_model),
    _short(SETTINGS.proxy_model_id if SETTINGS.proxy_backend == "flux" else SETTINGS.fal_endpoint),
    _short(SETTINGS.dino_model_id),
])

BRAND = gr.themes.Color(
    c50="#EEF3FD", c100="#DCE6FA", c200="#B9CDF5", c300="#8AABEC", c400="#5A85DD",
    c500="#3A68CF", c600="#2B59C3", c700="#234AA3", c800="#1E4297", c900="#1A3570", c950="#0F2048",
    name="brand",
)

THEME = gr.themes.Base(
    primary_hue=BRAND,
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F5F6F8",
    block_background_fill="#FFFFFF",
    block_border_color="#E3E6EB",
    block_border_width="1px",
    block_radius="12px",
    block_shadow="none",
    block_label_text_color="#5B6270",
    block_title_text_color="#14171F",
    block_title_text_weight="500",
    border_color_primary="#E3E6EB",
    body_text_color_subdued="#5B6270",
    input_background_fill="#FFFFFF",
    input_border_color="#D3D7DE",
    slider_color="#2B59C3",
    color_accent="#2B59C3",
    color_accent_soft="#EAF0FC",
    link_text_color="#2B59C3",
    button_primary_background_fill="#2B59C3",
    button_primary_background_fill_hover="#1E4297",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_border_color="#D3D7DE",
    button_large_radius="10px",
    button_large_text_size="15px",
    checkbox_background_color_selected="#2B59C3",
    table_even_background_fill="#FFFFFF",
    table_odd_background_fill="#FAFBFC",
    section_header_text_weight="600",
)

CSS_PATH = Path(__file__).parent / "app" / "theme.css"  # custom styles; loaded via launch(css_paths=...)

LOGO = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 0 1 15.5-6.2L21 8"/><path d="M21 3v5h-5"/>'
        '<path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/><path d="M3 21v-5h5"/></svg>')
CHECK = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" '
         'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>')

HEADER_HTML = f"""
<div class="pre-header">
  <div class="pre-brand">
    <div class="pre-logo">{LOGO}</div>
    <div><div class="pre-title">Prompt Reverse Engineer</div>
    <div class="pre-sub">Image → verified text-to-image prompt</div></div>
  </div>
  <nav class="pre-nav">
    <a href="#method">How it works</a>
    <a href="#cite">Cite</a>
    <span class="pre-chip"><span class="pre-dot"></span>{html.escape(MODEL_CHIP)}</span>
  </nav>
</div>
"""

INTRO_HTML = """
<div class="pre-card pre-intro" id="method">
  <h1>Turn an image back into a prompt you can verify</h1>
  <p>Instead of one free-form caption, the image is broken into checkable facts. A fast model renders the
  prompt, every fact is checked against the render, and only the failed parts are rewritten.</p>
  <ol class="pre-how">
    <li><span class="pre-n">01</span><b>Extract a scene graph</b>
      <span>Shapes, colors, medium and relations, each placed on a 3 × 3 grid.</span></li>
    <li><span class="pre-n">02</span><b>Ground on the original</b>
      <span>Facts the original itself fails are removed, so no unverified style words reach the prompt.</span></li>
    <li><span class="pre-n">03</span><b>Render and verify</b>
      <span>A proxy model renders the prompt; VQAScore checks each fact and DINO checks the layout.</span></li>
    <li><span class="pre-n">04</span><b>Rewrite only what failed</b>
      <span>Targeted edits repeat until both scores clear their gates or iterations run out.</span></li>
  </ol>
</div>
<div class="pre-placeholder"><b>Results appear here</b>
  <span>The verified prompt, proxy renders and per-fact scores stream in as each step finishes.</span></div>
"""

STAGES = ["Extract scene graph", "Ground on original", "Render proxy", "Verify", "Rewrite failures"]
ITER_HEADERS = ["Iteration", "Seed", "S_VQA", "S_DINO", "Joint", "Failed", "Rewrite"]
PRED_HEADERS = ["Kind", "Statement", "P(yes)", "Effective", "Status"]
GROUND_HEADERS = ["Fact", "Detail", "Action"]


# --------------------------------------------------------------------------- rendering

def _stage(run) -> int:
    """Index of the active pipeline stage (5 = finished)."""
    s = run.status
    if s.startswith("Done"):
        return 5
    for i, key in enumerate(("Module 1", "Module 2", "Module 3", "Module 4", "targeted rewrite")):
        if key in s:
            return i
    return 0


def _meter(value, gate=None):
    fill = f'<div class="pre-meter-fill" style="width:{max(0.0, min(1.0, value)) * 100:.0f}%"></div>'
    mark = "" if gate is None else f'<div class="pre-meter-gate" style="left:{gate * 100:.0f}%"></div>'
    return f'<div class="pre-meter">{fill}{mark}</div>'


def _tile(label, value, note, fraction, gate=None):
    return (f'<div class="pre-tile"><div class="pre-tile-label">{label}</div>'
            f'<div><span class="pre-tile-value">{value}</span> <span class="pre-muted" style="font-size:12px">'
            f'{note}</span></div>{_meter(fraction, gate)}</div>')


def _status_html(run) -> str:
    cfg, best, stage = run.config, run.best, _stage(run)
    if stage == 5 and run.stop_reason == "converged":
        badge = f'<span class="pre-badge ok">{CHECK}Converged</span>'
        line = f"Both gates met at iteration {best.t} of {cfg.max_iters}"
    elif stage == 5:
        badge = '<span class="pre-badge warn">Gates not met</span>'
        line = f"Returning the best of {len(run.iterations)} iterations (t = {best.t})"
    else:
        badge = '<span class="pre-badge run">Running</span>'
        line = html.escape(run.status)
    stages = "".join(
        f'<li class="{"done" if i < stage else "active" if i == stage else "pending"}"><div class="pre-bar"></div>'
        f'<div class="pre-n">{i + 1:02d}</div><div class="pre-stage-label">{name}</div></li>'
        for i, name in enumerate(STAGES))
    if best:
        tiles = (_tile("Predicate agreement · S_VQA", f"{best.mean_vqa:.2f}", f"gate {cfg.vqa_gate:.2f}",
                       best.mean_vqa, cfg.vqa_gate)
                 + _tile("Layout similarity · S_DINO", f"{best.dino_patch:.2f}", f"gate {cfg.dino_gate:.2f}",
                         best.dino_patch, cfg.dino_gate)
                 + _tile("Joint score", f"{best.joint:.3f}", f"best of {len(run.iterations)}", best.joint))
    else:
        tiles = (_tile("Predicate agreement · S_VQA", "—", f"gate {cfg.vqa_gate:.2f}", 0, cfg.vqa_gate)
                 + _tile("Layout similarity · S_DINO", "—", f"gate {cfg.dino_gate:.2f}", 0, cfg.dino_gate)
                 + _tile("Joint score", "—", "", 0))
    weights = f"Best = {cfg.w_vqa:g} · S_VQA + {cfg.w_dino:g} · S_DINO"
    return (f'<div class="pre-card"><div class="pre-row"><div class="pre-row" style="justify-content:flex-start">'
            f'{badge}<span class="pre-muted">{line}</span></div><span class="pre-muted" style="font-size:12px">'
            f'{weights}</span></div><ol class="pre-stages">{stages}</ol><div class="pre-tiles">{tiles}</div></div>')


def _iteration_rows(run):
    rows = []
    for it in run.iterations:
        if it.rewrite_error:
            rewrite = "Rewrite error"
        elif it.patch is not None:
            rewrite = f"{len(it.patch.applied)} edits, {len(it.patch.added)} added, {len(it.patch.skipped)} skipped"
        elif it is run.iterations[-1] and run.stop_reason:
            rewrite = "Stopped: gates met" if run.stop_reason == "converged" else "Stopped: max iterations"
        else:
            rewrite = "…"
        label = f"{it.t} · best" if it is run.best else str(it.t)
        rows.append([label, it.seed, round(it.mean_vqa, 3), round(it.dino_patch, 3), round(it.joint, 3),
                     len(it.failed), rewrite])
    return rows


def _predicate_rows(run, fail_threshold):
    best = run.best
    if best is None:
        return [[p.kind, p.statement, None, None, "Pending"] for p in run.predicates]
    rows = []
    for sp in best.scored:
        status = ("Blocked by " + ", ".join(sp.blocked_by)) if sp.blocked_by else (
            "Pass" if sp.effective >= fail_threshold else "Fail")
        rows.append([sp.predicate.kind, sp.predicate.statement, round(sp.raw, 3), round(sp.effective, 3), status])
    return rows


def _grounding_rows(run):
    rows = [[sp.predicate.statement, f"{sp.predicate.kind} · P(yes) {sp.raw:.2f} on the original", "Removed"]
            for sp in run.dropped]
    rows += [[m["entity"], f"position · {m['from']} → {m['to']} · P(yes) {m['p_yes']:.2f}", "Re-localized"]
             for m in run.relocalized]
    return rows


def _gallery(og, run):
    items = [(og, "Original")]
    items += [(it.proxy_image, f"t = {it.t} · VQA {it.mean_vqa:.2f} · DINO {it.dino_patch:.2f}"
               + (" · best" if it is run.best else "")) for it in run.iterations]
    return items


def _outputs(og, run, fail_threshold, report_path=None):
    return (
        gr.update(visible=False),  # intro
        gr.update(visible=True),  # results
        _status_html(run),
        run.best_prompt,
        gr.update(value=report_path, interactive=report_path is not None),
        _gallery(og, run),
        _iteration_rows(run),
        _predicate_rows(run, fail_threshold),
        run.graph.model_dump() if run.graph else None,
        _grounding_rows(run),
    )


def invert(image, max_iters, vqa_gate, dino_gate, fail_threshold, w_vqa, base_seed, ground):
    if image is None:
        raise gr.Error("Upload an image to start.")
    cfg = EngineConfig(max_iters=int(max_iters), vqa_gate=vqa_gate, dino_gate=dino_gate,
                       fail_threshold=fail_threshold, w_vqa=w_vqa, w_dino=round(1 - w_vqa, 4),
                       base_seed=int(base_seed), ground=bool(ground))
    run = None
    for run in iter_inversion(image, BACKENDS, cfg):
        yield _outputs(image, run, fail_threshold)
    assert run is not None
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="inversion_report_", delete=False,
                                     encoding="utf-8") as f:
        json.dump(run.to_dict(), f, indent=2)
    yield _outputs(image, run, fail_threshold, f.name)


# --------------------------------------------------------------------------- layout

with gr.Blocks(title="Prompt Reverse Engineer") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            with gr.Group(elem_classes="pre-group"):
                gr.HTML('<p class="pre-step-title">1 · Original image</p>', elem_classes="pre-group-title")
                image = gr.Image(type="pil", label="Original image", show_label=False,
                                 sources=["upload", "clipboard"], height=320, elem_classes="pre-image")
            with gr.Group(elem_classes="pre-group"):
                gr.HTML('<p class="pre-step-title">2 · Stopping rule</p>', elem_classes="pre-group-title")
                max_iters = gr.Slider(1, 8, value=5, step=1, label="Max iterations")
                vqa_gate = gr.Slider(0.5, 1.0, value=0.90, step=0.01, label="Min. predicate agreement (S_VQA)")
                dino_gate = gr.Slider(0.0, 1.0, value=0.80, step=0.01, label="Min. layout similarity (S_DINO)",
                                      info="Set to 0 to stop on S_VQA alone.")
            with gr.Accordion("Advanced settings", open=False):
                fail_threshold = gr.Slider(0.3, 0.95, value=0.75, step=0.01, label="Failure threshold",
                                           info="Facts scoring below this are sent to the rewriter.")
                w_vqa = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Weight on S_VQA",
                                  info="Picks the best iteration; the rest of the weight goes to S_DINO.")
                base_seed = gr.Number(value=42, precision=0, label="Base seed", info="Proxy seed = base + t.")
                ground = gr.Checkbox(value=True, label="Ground predicates on the original",
                                     info="Removes facts the scorer rejects on the original image.")
            run_button = gr.Button("Invert image", variant="primary", size="lg")

        with gr.Column(scale=3, min_width=560):
            intro = gr.HTML(INTRO_HTML)
            with gr.Column(visible=False, elem_classes="pre-results") as results:
                status = gr.HTML()
                best_prompt = gr.Textbox(label="Verified prompt", lines=3, max_lines=16, interactive=False,
                                         buttons=["copy"])
                report = gr.DownloadButton("Download report (JSON)", value=None, interactive=False,
                                           variant="secondary", size="sm", elem_classes="pre-dl")
                gallery = gr.Gallery(label="Original vs. proxy renders", columns=3, rows=1, height=310,
                                     object_fit="contain", allow_preview=True)
                with gr.Column(elem_classes="pre-panel"), gr.Tabs():
                    with gr.Tab("Iterations"):
                        iterations = gr.Dataframe(headers=ITER_HEADERS, interactive=False, show_label=False)
                    with gr.Tab("Predicates"):
                        predicates = gr.Dataframe(headers=PRED_HEADERS, interactive=False, wrap=True,
                                                  show_label=False, max_height=480)
                    with gr.Tab("Scene graph"):
                        graph_json = gr.JSON(show_label=False)
                    with gr.Tab("Grounding"):
                        gr.Markdown("Facts the scorer rejected on the original image. They are removed from the "
                                    "prompt, or moved to the grid cell the scorer supports, before rendering.")
                        grounding = gr.Dataframe(headers=GROUND_HEADERS, interactive=False, wrap=True,
                                                 show_label=False)

    with gr.Accordion("Cite", open=False, elem_id="cite"):
        gr.Markdown("""Author: Larrain, G

```bibtex
@misc{qwen3technicalreport,
  title={Qwen3 Technical Report}, author={Qwen Team}, year={2025},
  eprint={2505.09388}, archivePrefix={arXiv}, primaryClass={cs.CL}, url={https://arxiv.org/abs/2505.09388}
}

@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
  journal={arXiv preprint arXiv:2502.13923}, year={2025}
}

@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191}, year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966}, year={2023}
}

@inproceedings{cho2024dsg,
  title={Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation},
  author={Cho, Jaemin and Hu, Yushi and Garg, Roopal and Anderson, Peter and Krishna, Ranjay and Baldridge, Jason and Bansal, Mohit and Pont-Tuset, Jordi and Wang, Su},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2024}
}

@inproceedings{lin2024vqascore,
  title={Evaluating Text-to-Visual Generation with Image-to-Text Generation},
  author={Lin, Zhiqiu and Pathak, Deepak and Li, Baiqi and Li, Jiayao and Xia, Xide and Neubig, Graham and Zhang, Pengchuan and Ramanan, Deva},
  booktitle={European Conference on Computer Vision (ECCV)}, year={2024}
}

@article{oquab2024dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\\'e}e and Moutakanni, Th{\\'e}o and others},
  journal={Transactions on Machine Learning Research (TMLR)}, year={2024}
}

@article{simeoni2025dinov3,
  title={DINOv3}, author={Sim{\\'e}oni, Oriane and others},
  journal={arXiv preprint arXiv:2508.10104}, year={2025}
}

@misc{flux2024,
  title={FLUX}, author={{Black Forest Labs}}, year={2024},
  howpublished={https://github.com/black-forest-labs/flux}
}
```""")

    run_button.click(
        invert,
        [image, max_iters, vqa_gate, dino_gate, fail_threshold, w_vqa, base_seed, ground],
        [intro, results, status, best_prompt, report, gallery, iterations, predicates, graph_json, grounding],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css_paths=[CSS_PATH])
