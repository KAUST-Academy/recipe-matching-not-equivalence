#!/usr/bin/env python
"""WiSE-FT weight souping for the InvarEmbed phase-1 checkpoint.

Linearly interpolates every matching floating-point tensor between the
UNTRAINED base (Qwen/Qwen3-Embedding-0.6B) and the phase-1 fine-tuned
checkpoint (models/qwen3-0.6b-cas-cmnrl/final):

    theta_soup = alpha * theta_finetuned + (1 - alpha) * theta_base

(WiSE-FT, Wortsman et al. 2022: robust fine-tuning via weight-space
ensembling; alpha=1 is the fine-tuned model, alpha=0 the base.)

Motivation (phase-1 result, reproducibility.json id
"invarembed_phase1_full_sweep"): full FT doubled R@1 on every tier
(easy 8.32->15.92, medium 1.78->3.86, hard 0.00->0.16) but collapsed
recall depth (medium R@5 60.91->40.80). Interpolating back toward the
base is a zero-training attempt to keep the R@1 gain while recovering
the base's recall breadth.

Mechanics
---------
* Operates directly on the two model.safetensors files. Verified
  2026-07-29: both files carry the SAME 310 keys, all bf16 (base HF
  snapshot 97b0c614). Interpolation runs in float32, result is cast back
  to the fine-tuned tensor's dtype (bf16).
* Non-float tensors, shape mismatches, and fine-tuned-only keys are
  copied verbatim from the FINE-TUNED file (counted + reported);
  base-only keys are ignored with a warning. A run that interpolates
  <90% of tensors aborts (schema drift guard).
* Every other file of the fine-tuned model dir (config.json, tokenizer,
  modules.json, 1_Pooling/, 2_Normalize/, ...) is copied unchanged, so
  each output is a complete, directly loadable sentence-transformers
  model. A soup_config.json records provenance.
* Full non-finite (NaN/inf) check on every interpolated tensor.

Usage (login node, CPU, ~1 min/alpha, I/O bound):
    conda activate mathnet
    python scripts/soup_models.py                       # alphas 0.3,0.5,0.7
    python scripts/soup_models.py --alphas 0.5 --force  # rebuild one
Outputs: models/soup-a{alpha}/  (e.g. models/soup-a0.5/)

Evaluate with scripts/eval_soups.slurm (all three alphas x all tiers).
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT / ".hf_cache"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

WEIGHTS_NAME = "model.safetensors"


def resolve_weights(model_ref: str) -> Path:
    """Return the model.safetensors path for a local dir or a HF hub id."""
    p = Path(model_ref)
    if p.is_dir():
        f = p / WEIGHTS_NAME
        if not f.is_file():
            sys.exit(f"[fatal] {f} not found (sharded checkpoints unsupported)")
        return f
    from huggingface_hub import hf_hub_download
    try:  # prefer the local HF cache (login nodes may lack internet)
        return Path(hf_hub_download(model_ref, WEIGHTS_NAME, local_files_only=True))
    except Exception:
        print(f"[base] {model_ref} not in local HF cache; downloading...", flush=True)
        return Path(hf_hub_download(model_ref, WEIGHTS_NAME))


def load_tensors(path: Path):
    tensors, metadata = {}, None
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {"format": "pt"}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, metadata


def soup_state_dict(ft: dict, base: dict, alpha: float):
    """Interpolate matching float tensors; copy the rest from the fine-tuned."""
    out, n_interp, copied = {}, 0, []
    for k, t_ft in ft.items():
        t_b = base.get(k)
        if (t_b is not None and t_ft.is_floating_point()
                and t_b.is_floating_point() and t_ft.shape == t_b.shape):
            mixed = alpha * t_ft.float() + (1.0 - alpha) * t_b.float()
            if not torch.isfinite(mixed).all():
                sys.exit(f"[fatal] non-finite values in interpolated tensor {k}")
            out[k] = mixed.to(t_ft.dtype).contiguous()
            n_interp += 1
        else:
            out[k] = t_ft.clone().contiguous()
            copied.append(k)
    return out, n_interp, copied


def copy_model_dir(src: Path, dst: Path):
    """Copy everything of the fine-tuned model dir except the weights file."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == WEIGHTS_NAME:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Qwen/Qwen3-Embedding-0.6B",
                    help="HF hub id or local dir of the UNTRAINED base")
    ap.add_argument("--finetuned", default=str(PROJECT / "models/qwen3-0.6b-cas-cmnrl/final"),
                    help="local dir of the fine-tuned sentence-transformers model")
    ap.add_argument("--alphas", default="0.3,0.5,0.7",
                    help="comma-separated fine-tuned weights (alpha=1 -> pure FT)")
    ap.add_argument("--output-root", default=str(PROJECT / "models"),
                    help="soups are written to <output-root>/soup-a<alpha>/")
    ap.add_argument("--skip-existing", action="store_true",
                    help="silently skip alphas whose output dir already exists")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output dirs")
    args = ap.parse_args()

    alphas = []
    for a in args.alphas.split(","):
        a = a.strip()
        v = float(a)  # validates
        if not (0.0 < v < 1.0):
            sys.exit(f"[fatal] alpha {a} outside (0,1) — alpha=0/1 are the parents")
        alphas.append((a, v))

    ft_dir = Path(args.finetuned)
    ft_file = resolve_weights(str(ft_dir))
    base_file = resolve_weights(args.base)
    print(f"[load] finetuned: {ft_file}", flush=True)
    print(f"[load] base:      {base_file}", flush=True)
    ft, ft_meta = load_tensors(ft_file)
    base, _ = load_tensors(base_file)

    ft_only = sorted(set(ft) - set(base))
    base_only = sorted(set(base) - set(ft))
    if ft_only:
        print(f"[warn] {len(ft_only)} finetuned-only keys (copied verbatim): "
              f"{ft_only[:5]}...", flush=True)
    if base_only:
        print(f"[warn] {len(base_only)} base-only keys IGNORED: {base_only[:5]}...",
              flush=True)
    print(f"[keys] finetuned={len(ft)} base={len(base)} "
          f"common={len(set(ft) & set(base))}", flush=True)

    for a_str, a_val in alphas:
        out_dir = Path(args.output_root) / f"soup-a{a_str}"
        if out_dir.exists():
            if args.skip_existing:
                print(f"[skip] {out_dir} exists", flush=True)
                continue
            if not args.force:
                sys.exit(f"[fatal] {out_dir} exists (use --force or --skip-existing)")
            shutil.rmtree(out_dir)
        t0 = time.time()
        souped, n_interp, copied = soup_state_dict(ft, base, a_val)
        frac = n_interp / max(len(ft), 1)
        if frac < 0.9:
            sys.exit(f"[fatal] only {n_interp}/{len(ft)} tensors interpolated "
                     f"({frac:.0%}) — key schemas have drifted, refusing to save")
        copy_model_dir(ft_dir, out_dir)
        save_file(souped, str(out_dir / WEIGHTS_NAME), metadata=ft_meta)
        provenance = {
            "method": "WiSE-FT linear weight interpolation (Wortsman et al. 2022)",
            "formula": "alpha * finetuned + (1 - alpha) * base",
            "alpha_finetuned": a_val,
            "base": args.base,
            "base_weights_file": str(base_file),
            "finetuned": str(ft_dir),
            "tensors_interpolated": n_interp,
            "tensors_copied_from_finetuned": copied,
            "interpolation_dtype": "float32 (cast back to per-tensor finetuned dtype)",
            "built_by": "scripts/soup_models.py",
            "date": date.today().isoformat(),
        }
        (out_dir / "soup_config.json").write_text(json.dumps(provenance, indent=2))
        print(f"[soup] alpha={a_str}: {n_interp}/{len(ft)} tensors interpolated, "
              f"{len(copied)} copied -> {out_dir}  ({time.time() - t0:.1f}s)",
              flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
