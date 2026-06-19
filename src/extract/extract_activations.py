"""Step 1: one forward pass per problem -> final answer (argmax @ '=') + '=' activations per layer. CUDA only."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "common"))
import registry as reg  # noqa: E402


def load_rows(data_path, limit):
    with open(data_path, "r", encoding="utf-8") as f:
        rows = [(int(r["id"]), int(r["a"]), int(r["b"]), int(r["sum"])) for r in csv.DictReader(f)]
    return rows[:limit] if limit else rows


def make_hook(storage, layer_idx):
    def hook_fn(module, inp, output):
        hidden = output if not isinstance(output, tuple) else output[0]
        storage[layer_idx].append(hidden[:, -1, :].detach().float().cpu())
    return hook_fn


def validate(arr, n, d):
    norms = np.linalg.norm(arr, axis=1)
    return {"shape": list(arr.shape), "shape_ok": arr.shape == (n, d),
            "any_nan": bool(np.isnan(arr).any()), "any_inf": bool(np.isinf(arr).any()),
            "norm_mean": float(norms.mean()), "norm_min": float(norms.min()), "norm_max": float(norms.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "ground_truth" / "addition_2digit.csv"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "activations"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch

    registry = reg.load_registry()
    model_key = args.model or registry["primary"]
    cfg = reg.get_model_cfg(model_key, registry)
    template = cfg["prompt_template"]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else cfg["layers"]

    out_dir = Path(args.out_dir) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_paths = {L: out_dir / f"layer_{L:02d}.npy" for L in layers}
    answers_path = out_dir / "answers.csv"
    if not args.overwrite and answers_path.exists() and all(p.exists() for p in layer_paths.values()):
        print(f"outputs exist in {out_dir}; --overwrite to redo.")
        return

    rows = load_rows(Path(args.data), args.limit)
    n = len(rows)
    tokenizer, model, hidden_dim = reg.load_model_and_tokenizer(model_key, registry)
    blocks = reg.get_block_modules(model, cfg["block_path"])
    print(f"Extract | {model_key} layers={layers} n={n} prompt={template!r}")

    captured = {L: [] for L in layers}
    pred_ids, pred_strs = [], []
    t0 = time.time()
    n_batches = (n + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = rows[bi * args.batch_size:(bi + 1) * args.batch_size]
        prompts = [template.format(a=a, b=b) for (_, a, b, _) in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        assert bool(inputs["attention_mask"][:, -1].all()), "left-pad invariant broken"
        handles = [blocks[L].register_forward_hook(make_hook(captured, L)) for L in layers]
        try:
            with torch.inference_mode():
                out = model(**inputs, use_cache=False)
            batch_pred = out.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
        finally:
            for h in handles:
                h.remove()
        pred_ids.extend(batch_pred)
        pred_strs.extend(tokenizer.decode([tid]) for tid in batch_pred)
        if (bi + 1) % 25 == 0 or (bi + 1) == n_batches:
            print(f"  batch {bi+1}/{n_batches}")
    runtime = time.time() - t0

    report = {}
    for L in layers:
        arr = torch.cat(captured[L], dim=0).numpy().astype(np.float32)
        if arr.shape != (n, hidden_dim):
            raise RuntimeError(f"layer {L}: {arr.shape} != ({n},{hidden_dim})")
        np.save(layer_paths[L], arr)
        report[str(L)] = validate(arr, n, hidden_dim)

    exp_by_s = {s: tokenizer.encode(str(s), add_special_tokens=False)[0] for s in sorted({r[3] for r in rows})}
    n_correct = 0
    with open(answers_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "a", "b", "s", "predicted_id", "predicted_str", "correct"])
        for (pid, a, b, s), tid, pstr in zip(rows, pred_ids, pred_strs):
            correct = int(tid == exp_by_s[s])
            n_correct += correct
            w.writerow([pid, a, b, s, tid, pstr, correct])
    accuracy = n_correct / max(n, 1)

    manifest = {
        "schema_version": "geom_arith_extract_v1",
        "model_key": model_key, "hf_name": cfg["hf_name"], "prompt_template": template,
        "hidden_dim": hidden_dim, "layers": layers, "block_path": cfg["block_path"],
        "n_problems": n, "n_correct": n_correct, "accuracy": round(accuracy, 6),
        "decoding": "greedy argmax @ '=' (T=0)", "dtype": str(next(model.parameters()).dtype),
        "runtime_seconds": round(runtime, 2), "data_source": str(args.data), "validation": report,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    all_ok = all(report[str(L)]["shape_ok"] and not report[str(L)]["any_nan"] for L in layers)
    print(f"accuracy {accuracy*100:.2f}% ({n_correct}/{n}) | {runtime:.1f}s | -> {out_dir}")
    if not all_ok:
        print("FAIL: shape/NaN; see manifest.")
        sys.exit(1)
    print(f"PASS: {len(layers)} layers ({n},{hidden_dim}) + answers.csv")


if __name__ == "__main__":
    main()
