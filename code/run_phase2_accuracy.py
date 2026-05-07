"""
Phase 2: accuracy reproduction. Per-model job; takes --model as a CLI arg.
Per babel_execution_plan.md §5.

Reproduces KT 2025's accuracies:
  gpt-j-6b:     80.5%
  pythia-6.9b:  77.2%
  llama-3.1-8b: 98.0%

Decoding strategy:
  - GPT-J / Pythia: 1-step argmax of next-token logits at the '=' position.
    Single-token answer is sufficient because GPT-2 BPE tokenizes 0..198 as
    single tokens.
  - Llama 3.1: multi-token greedy decode via generate(max_new_tokens=4),
    decode the new tokens, and string-compare to str(s). This handles
    Llama's per-digit number tokenization (e.g., "75" -> ['7', '5']).

Pre-registered gate: |empirical - KT_reported| <= 5 percentage points.

Outputs:
  $BLACKBOX_DATA/correctness/{model}.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from accuracy_check import (
    MODEL_CONFIG, load_model_and_tokenizer, save_correctness, summary_report,
    passes_sanity_gate, load_intersection_pairs,
)


def run_single_token_accuracy(model, tokenizer, pairs, model_key, batch_size,
                                save_every, partial_path):
    """1-step argmax decode for single-token-answer models (GPT-J, Pythia)."""
    cfg = MODEL_CONFIG[model_key]
    template = cfg["prompt_template"]

    # Resume from partial
    completed = set()
    rows: List[Dict] = []
    if partial_path and os.path.exists(partial_path):
        cached = pd.read_parquet(partial_path)
        rows = cached.to_dict(orient="records")
        completed = set((int(r["a"]), int(r["b"])) for r in rows)
        print(f"Resuming from partial: {len(rows)} pairs already done")

    remaining = [p for p in pairs if (int(p["a"]), int(p["b"])) not in completed]
    if not remaining:
        return pd.DataFrame(rows)

    n_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(n_batches), desc=f"{model_key} (1-step)"):
        batch = remaining[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        prompts = [template.format(a=p["a"], b=p["b"]) for p in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        last_logits = out.logits[:, -1, :]
        predicted_ids = last_logits.argmax(dim=-1).cpu().tolist()

        for p, pred_id in zip(batch, predicted_ids):
            exp_id = int(p["first_answer_token_id"])
            try:
                pred_str = tokenizer.decode([pred_id])
            except Exception:
                pred_str = ""
            rows.append({
                "a":             int(p["a"]),
                "b":             int(p["b"]),
                "s":             int(p["s"]),
                "predicted_id":  int(pred_id),
                "predicted_str": pred_str,
                "expected_id":   exp_id,
                "expected_str":  p["first_answer_token_str"],
                "correct":       bool(pred_id == exp_id),
            })

        if partial_path and (batch_idx + 1) % save_every == 0:
            pd.DataFrame(rows).to_parquet(partial_path, index=False)

    return pd.DataFrame(rows)


def run_multitoken_accuracy(model, tokenizer, pairs, model_key, batch_size,
                              save_every, partial_path, max_new_tokens=4):
    """Multi-token greedy decode for Llama 3.1 (per-digit tokenization).

    Uses generate(do_sample=False, max_new_tokens=max_new_tokens), then
    decodes the new tokens (slicing off the prompt) and string-compares
    the trimmed decode to str(s).
    """
    cfg = MODEL_CONFIG[model_key]
    template = cfg["prompt_template"]

    completed = set()
    rows: List[Dict] = []
    if partial_path and os.path.exists(partial_path):
        cached = pd.read_parquet(partial_path)
        rows = cached.to_dict(orient="records")
        completed = set((int(r["a"]), int(r["b"])) for r in rows)
        print(f"Resuming from partial: {len(rows)} pairs already done")

    remaining = [p for p in pairs if (int(p["a"]), int(p["b"])) not in completed]
    if not remaining:
        return pd.DataFrame(rows)

    n_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(n_batches), desc=f"{model_key} (multi-token)"):
        batch = remaining[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        prompts = [template.format(a=p["a"], b=p["b"]) for p in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_lens = inputs["attention_mask"].sum(dim=1)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        # generated has shape (B, prompt_len_padded + max_new_tokens). Decode
        # only the newly-generated tokens per row.
        for i, p in enumerate(batch):
            seq = generated[i]
            # With left padding: prompt sits at positions [pad_count, total_input_len),
            # so new tokens are at positions [total_input_len, total_input_len + max_new_tokens).
            total_input_len = inputs["input_ids"].shape[1]
            new_token_ids = seq[total_input_len: total_input_len + max_new_tokens].cpu().tolist()
            new_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            # Strip leading whitespace; compare
            cleaned = new_text.strip()
            expected = str(p["s"])
            # Check if the cleaned decode starts with the expected number, and
            # the next char (if any) is non-digit (so we don't accept "752" for s=75).
            correct = (cleaned.startswith(expected)
                        and (len(cleaned) == len(expected) or not cleaned[len(expected)].isdigit()))
            rows.append({
                "a":             int(p["a"]),
                "b":             int(p["b"]),
                "s":             int(p["s"]),
                "generated_ids": new_token_ids,
                "generated_str": new_text,
                "expected_str":  expected,
                "correct":       bool(correct),
            })

        if partial_path and (batch_idx + 1) % save_every == 0:
            pd.DataFrame(rows).to_parquet(partial_path, index=False)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIG.keys()))
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("BLACKBOX_DATA", "/data/user_data/anshulk/blackbox"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    intersect_path = os.path.join(args.data_dir, "tokenizer_audit", "intersection.json")
    pairs = load_intersection_pairs(intersect_path)
    # Each pair already has first_answer_token_id from Phase 1's audit.
    print(f"Loaded {len(pairs)} pairs from {intersect_path}")

    out_dir = os.path.join(args.data_dir, "correctness")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model}.parquet")

    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, device="cuda")
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Pick decoding strategy: per-model audit's `all_answers_single_token` flag
    audit_path = os.path.join(args.data_dir, "tokenizer_audit", f"{args.model}.json")
    with open(audit_path) as f:
        audit = json.load(f)
    if audit["all_answers_single_token"]:
        print(f"  using 1-step argmax decode (all answers single-token)")
        df = run_single_token_accuracy(
            model, tokenizer, pairs, args.model,
            batch_size=args.batch_size, save_every=args.save_every,
            partial_path=out_path,
        )
    else:
        print(f"  using multi-token greedy decode (some answers multi-token)")
        df = run_multitoken_accuracy(
            model, tokenizer, pairs, args.model,
            batch_size=max(args.batch_size // 2, 8),
            save_every=args.save_every, partial_path=out_path,
        )

    save_correctness(df, out_path)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print()
    print(summary_report(df, args.model))
    print()
    if passes_sanity_gate(df, args.model, tolerance_pp=5.0):
        print(f"PASS: empirical accuracy within 5pp of KT.")
    else:
        print(f"FAIL: empirical accuracy more than 5pp from KT. Halt before Phase 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()
