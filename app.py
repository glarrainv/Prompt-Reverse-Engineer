import json
import tempfile

import gradio as gr
import spaces

from inversion import EngineConfig, iter_inversion
from inversion.backends import BackendSettings, load_components
from inversion.engine import Backends

SETTINGS = BackendSettings()
C = load_components(SETTINGS)


# Each model call gets its own (short) ZeroGPU allocation instead of one long one
# for the whole loop. Outside Spaces/ZeroGPU, @spaces.GPU is a no-op.
@spaces.GPU(duration=180)
def gpu_extract(image):
    return C.vlm.extract_dsg(image)


@spaces.GPU(duration=90)
def gpu_vqa(image, statements, template):
    return C.scorer.vqa_scores(image, statements, template)


@spaces.GPU(duration=120)
def gpu_rewrite(og, proxy, prompt, failed):
    return C.vlm.rewrite(og, proxy, prompt, failed)


@spaces.GPU(duration=60)
def gpu_render(prompt, seed, width, height):
    return C.renderer(prompt, seed, width, height)


@spaces.GPU(duration=30)
def gpu_embed(image):
    return C.dino.embed(image)


BACKENDS = Backends(extract_dsg=gpu_extract, vqa_scores=gpu_vqa, rewrite=gpu_rewrite,
                    render=gpu_render, embed=gpu_embed)

ITER_HEADERS = ["t", "seed", "S_VQA", "S_DINO (patch)", "DINO (CLS)", "joint", "failed", "edits applied"]
PRED_HEADERS = ["kind", "statement", "P(yes)", "effective", "status"]


def _iteration_rows(run):
    rows = []
    for it in run.iterations:
        applied = "" if it.patch is None else f"{len(it.patch.applied)} (+{len(it.patch.added)} added)"
        if it.rewrite_error:
            applied = "rewrite error"
        rows.append([it.t, it.seed, round(it.mean_vqa, 3), round(it.dino_patch, 3), round(it.dino_cls, 3),
                     round(it.joint, 3), len(it.failed), applied])
    return rows


def _predicate_rows(run, fail_threshold):
    best = run.best
    if best is None:
        return [[p.kind, p.statement, None, None, "pending"] for p in run.predicates]
    rows = []
    for sp in best.scored:
        status = ("blocked: " + ", ".join(sp.blocked_by)) if sp.blocked_by else (
            "pass" if sp.effective >= fail_threshold else "FAIL")
        rows.append([sp.predicate.kind, sp.predicate.statement, round(sp.raw, 3), round(sp.effective, 3), status])
    return rows


def _outputs(run, fail_threshold, report_path=None):
    gallery = [(it.proxy_image, f"t={it.t} · VQA {it.mean_vqa:.2f} · DINO {it.dino_patch:.2f}"
                + (" · best" if it is run.best else "")) for it in run.iterations]
    dropped = [[sp.predicate.kind, sp.predicate.statement, round(sp.raw, 3)] for sp in run.dropped]
    dropped += [["position → re-localized", f"{m['entity']}: {m['from']} → {m['to']}", m["p_yes"]]
                for m in run.relocalized]
    return (
        f"**Status:** {run.status}",
        run.best_prompt,
        gallery,
        _iteration_rows(run),
        _predicate_rows(run, fail_threshold),
        run.graph.model_dump() if run.graph else None,
        dropped,
        report_path,
    )


def invert(image, max_iters, vqa_gate, dino_gate, fail_threshold, w_vqa, base_seed, ground):
    if image is None:
        raise gr.Error("Upload an image.")
    cfg = EngineConfig(max_iters=int(max_iters), vqa_gate=vqa_gate, dino_gate=dino_gate,
                       fail_threshold=fail_threshold, w_vqa=w_vqa, w_dino=round(1 - w_vqa, 4),
                       base_seed=int(base_seed), ground=bool(ground))
    run = None
    for run in iter_inversion(image, BACKENDS, cfg):
        yield _outputs(run, fail_threshold)
    assert run is not None
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="inversion_report_", delete=False,
                                     encoding="utf-8") as f:
        json.dump(run.to_dict(), f, indent=2)
    yield _outputs(run, fail_threshold, f.name)


with gr.Blocks(title="Prompt Reverse Engineer") as demo:
    gr.Markdown("# Prompt Reverse Engineer")
    gr.Markdown("## Image → Verified Prompt")
    gr.Markdown(
        "Closed-loop inversion: **(1)** a constrained Davidsonian scene graph is extracted on a 3×3 grid, "
        "**(2)** compiled into dependency-ordered yes/no predicates and a canonical prompt, "
        "**(3)** rendered by a fast proxy text-to-image model, and **(4)** verified with VQAScore + DINOv2 patch "
        "similarity. Only failed predicates are rewritten, until S_VQA and S_DINO clear their gates."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Original image")
            max_iters = gr.Slider(1, 8, value=5, step=1, label="Max iterations (t)")
            vqa_gate = gr.Slider(0.5, 1.0, value=0.90, step=0.01, label="S_VQA gate")
            dino_gate = gr.Slider(0.3, 1.0, value=0.80, step=0.01, label="S_DINO gate (patch cosine)")
            with gr.Accordion("Advanced", open=False):
                fail_threshold = gr.Slider(0.3, 0.95, value=0.75, step=0.01,
                                           label="Predicate failure threshold (sent to rewriter)")
                w_vqa = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Joint-score weight on S_VQA (rest on DINO)")
                base_seed = gr.Number(value=42, precision=0, label="Base proxy seed (seed = base + t)")
                ground = gr.Checkbox(value=True, label="Ground predicates on the original (prune hallucinations)")
            run_button = gr.Button("Invert", variant="primary")

        with gr.Column(scale=2):
            status = gr.Markdown("**Status:** idle")
            best_prompt = gr.Textbox(label="Verified prompt P*", lines=10, interactive=False, buttons=["copy"])
            gallery = gr.Gallery(label="Proxy renders Î(t)", columns=3, height="auto")
            iterations = gr.Dataframe(headers=ITER_HEADERS, label="Iterations", interactive=False)

    with gr.Accordion("Predicates (best iteration)", open=False):
        predicates = gr.Dataframe(headers=PRED_HEADERS, interactive=False, wrap=True)
    with gr.Accordion("Scene graph G (after grounding)", open=False):
        graph_json = gr.JSON()
    with gr.Accordion("Grounding: predicates pruned as hallucinations", open=False):
        dropped = gr.Dataframe(headers=["kind", "statement", "P(yes) on original"], interactive=False, wrap=True)
    report = gr.File(label="Run report (JSON)")

    gr.Markdown("Author: Larrain, G")

    with gr.Accordion("References", open=False):
        gr.Markdown("""
            @misc{qwen3technicalreport,
                title={Qwen3 Technical Report},
                author={Qwen Team},
                year={2025},
                eprint={2505.09388},
                archivePrefix={arXiv},
                primaryClass={cs.CL},
                url={https://arxiv.org/abs/2505.09388},
            }

            @article{Qwen2.5-VL,
            title={Qwen2.5-VL Technical Report},
            author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
            journal={arXiv preprint arXiv:2502.13923},
            year={2025}
            }

            @article{Qwen2VL,
            title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
            author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
            journal={arXiv preprint arXiv:2409.12191},
            year={2024}
            }

            @article{Qwen-VL,
            title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
            author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
            journal={arXiv preprint arXiv:2308.12966},
            year={2023}
            }

            @inproceedings{cho2024dsg,
            title={Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation},
            author={Cho, Jaemin and Hu, Yushi and Garg, Roopal and Anderson, Peter and Krishna, Ranjay and Baldridge, Jason and Bansal, Mohit and Pont-Tuset, Jordi and Wang, Su},
            booktitle={International Conference on Learning Representations (ICLR)},
            year={2024}
            }

            @inproceedings{lin2024vqascore,
            title={Evaluating Text-to-Visual Generation with Image-to-Text Generation},
            author={Lin, Zhiqiu and Pathak, Deepak and Li, Baiqi and Li, Jiayao and Xia, Xide and Neubig, Graham and Zhang, Pengchuan and Ramanan, Deva},
            booktitle={European Conference on Computer Vision (ECCV)},
            year={2024}
            }

            @article{oquab2024dinov2,
            title={DINOv2: Learning Robust Visual Features without Supervision},
            author={Oquab, Maxime and Darcet, Timoth{\\'e}e and Moutakanni, Th{\\'e}o and others},
            journal={Transactions on Machine Learning Research (TMLR)},
            year={2024}
            }

            @article{simeoni2025dinov3,
            title={DINOv3},
            author={Sim{\\'e}oni, Oriane and others},
            journal={arXiv preprint arXiv:2508.10104},
            year={2025}
            }

            @misc{flux2024,
            title={FLUX},
            author={{Black Forest Labs}},
            year={2024},
            howpublished={https://github.com/black-forest-labs/flux}
            }
""")

    run_button.click(
        invert,
        [image, max_iters, vqa_gate, dino_gate, fail_threshold, w_vqa, base_seed, ground],
        [status, best_prompt, gallery, iterations, predicates, graph_json, dropped, report],
    )

if __name__ == "__main__":
    demo.launch()
