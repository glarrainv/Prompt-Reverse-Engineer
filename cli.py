"""Batch tooling for the stimulus bank.

  python cli.py invert  --input stimuli/ --out runs/          # lock P*_k for every S_k
  python cli.py medoid  --input renders/ --out medoids.json   # DINO medoid of the seeds per condition

Model backends are configured with the environment variables documented in
inversion/backends.py (local Qwen3-VL + FLUX by default; API pathway available).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def cmd_invert(args: argparse.Namespace) -> int:
    from inversion import EngineConfig, iter_inversion, save_run
    from inversion.backends import load_components

    images = _images(Path(args.input))
    if not images:
        print(f"no images found in {args.input}", file=sys.stderr)
        return 1
    cfg = EngineConfig(max_iters=args.max_iters, vqa_gate=args.vqa_gate, dino_gate=args.dino_gate,
                       fail_threshold=args.fail_threshold, w_vqa=args.w_vqa, w_dino=round(1 - args.w_vqa, 4),
                       base_seed=args.seed, ground=not args.no_ground)
    backends = load_components().backends()
    out_root = Path(args.out)
    summary = []
    for path in images:
        og = Image.open(path).convert("RGB")
        print(f"\n=== {path.name}")
        run = None
        for run in iter_inversion(og, backends, cfg):
            print(f"  {run.status}", flush=True)
        save_run(run, out_root / path.stem, og)
        best = run.best
        summary.append({"image": path.name, "stop_reason": run.stop_reason, "best_t": best.t,
                        "S_VQA": round(best.mean_vqa, 4), "S_DINO_patch": round(best.dino_patch, 4),
                        "joint": round(best.joint, 4), "n_predicates": len(run.predicates),
                        "n_grounding_dropped": len(run.dropped), "best_prompt": run.best_prompt})
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nwrote {out_root / 'summary.csv'}")
    return 0


def cmd_medoid(args: argparse.Namespace) -> int:
    import os

    from inversion.dino import DEFAULT_DINO_MODEL, DinoEmbedder, medoid_index

    root = Path(args.input)
    # Each subdirectory with images is one condition (e.g. renders/S1_nano_banana/seed*.png);
    # a flat directory of images is treated as a single condition.
    conditions = {d.name: _images(d) for d in sorted(root.iterdir()) if d.is_dir() and _images(d)}
    if not conditions and _images(root):
        conditions = {root.name: _images(root)}
    if not conditions:
        print(f"no images found under {root}", file=sys.stderr)
        return 1

    dino = DinoEmbedder(os.getenv("DINO_MODEL_ID") or DEFAULT_DINO_MODEL)
    result = {}
    for name, paths in conditions.items():
        embs = [dino.embed(Image.open(p)) for p in paths]
        idx, mean_sims = medoid_index(embs, key=args.key)
        result[name] = {"medoid": str(paths[idx]),
                        "mean_similarity": {str(p): round(s, 4) for p, s in zip(paths, mean_sims)}}
        print(f"{name}: medoid = {paths[idx].name} (mean {args.key} cosine {mean_sims[idx]:.3f})")
        if args.copy_to:
            dest = Path(args.copy_to)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paths[idx], dest / f"{name}{paths[idx].suffix}")
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("invert", help="run the closed-loop inversion on an image or a folder of images")
    inv.add_argument("--input", required=True)
    inv.add_argument("--out", default="runs")
    inv.add_argument("--max-iters", type=int, default=5)
    inv.add_argument("--vqa-gate", type=float, default=0.90)
    inv.add_argument("--dino-gate", type=float, default=0.80)
    inv.add_argument("--fail-threshold", type=float, default=0.75)
    inv.add_argument("--w-vqa", type=float, default=0.7)
    inv.add_argument("--seed", type=int, default=42)
    inv.add_argument("--no-ground", action="store_true", help="skip pruning predicates false on the original")
    inv.set_defaults(func=cmd_invert)

    med = sub.add_parser("medoid", help="select the DINO medoid render per condition folder")
    med.add_argument("--input", required=True)
    med.add_argument("--out", default="medoids.json")
    med.add_argument("--key", choices=["patch", "cls"], default="patch")
    med.add_argument("--copy-to", help="optional folder to copy each condition's medoid into")
    med.set_defaults(func=cmd_medoid)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
