"""
Phase 1: tokenizer audit. CPU-only; runs on Babel login node in ~5 min.

Per babel_execution_plan.md §4. For each (a, b) in {0,...,99}^2 and each
target model, determines whether the prompt tokenizes such that:
  - operand a appears as a single token
  - operand b appears as a single token
  - the FIRST token of the answer (s = a + b) is what the model would predict

For GPT-J 6B and Pythia 6.9B (GPT-2 BPE), all 0-198 are single-token, so
the audit retains all 10000 pairs. For Llama 3.1 8B (tiktoken-style BPE
that splits multi-digit numbers per-digit), the audit relaxes the answer
check to the FIRST DIGIT only, since that's what the model predicts in
one greedy step at the '=' position. Multi-token full answers are then
decoded via generate() in Phase 2 and string-compared.

Outputs:
  $BLACKBOX_DATA/tokenizer_audit/gpt-j-6b.json
  $BLACKBOX_DATA/tokenizer_audit/pythia-6.9b.json
  $BLACKBOX_DATA/tokenizer_audit/llama-3.1-8b.json
  $BLACKBOX_DATA/tokenizer_audit/intersection.json

Pre-registered gate: |intersection| >= 7000 pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from tokenizer_audit import (
    audit_tokenizer, intersect_audits, save_audit, save_intersection,
    summary_report, PROMPT_TEMPLATES,
)


MODELS = {
    "gpt-j-6b":     "EleutherAI/gpt-j-6B",
    "pythia-6.9b":  "EleutherAI/pythia-6.9b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("BLACKBOX_DATA", "/data/user_data/anshulk/blackbox"),
        help="Root data directory (defaults to $BLACKBOX_DATA env var).",
    )
    parser.add_argument(
        "--gate-threshold", type=int, default=7000,
        help="Pre-registered intersection-size gate (default 7000).",
    )
    args = parser.parse_args()

    audit_dir = os.path.join(args.data_dir, "tokenizer_audit")
    os.makedirs(audit_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizers: Dict[str, object] = {}
    for key, hf_name in MODELS.items():
        print(f"Loading tokenizer for {key} ({hf_name})...", flush=True)
        tokenizers[key] = AutoTokenizer.from_pretrained(hf_name)
        print(f"  vocab size: {tokenizers[key].vocab_size}", flush=True)

    audits: List[Dict] = []
    for key in MODELS:
        print(f"\nAuditing {key}...", flush=True)
        print(f"  prompt template: {PROMPT_TEMPLATES[key]!r}", flush=True)
        audit = audit_tokenizer(tokenizers[key], key)
        audits.append(audit)
        out_path = os.path.join(audit_dir, f"{key}.json")
        save_audit(audit, out_path)
        print(f"  retained {audit['n_retained']} / 10000, dropped {audit['n_dropped']}.", flush=True)
        print(f"  saved to {out_path}", flush=True)

    intersection = intersect_audits(audits)
    intersect_path = os.path.join(audit_dir, "intersection.json")
    save_intersection(intersection, audits, intersect_path)
    print(f"\nSaved {len(intersection)} retained pairs to {intersect_path}")

    print()
    print(summary_report(audits, intersection))

    n_int = len(intersection)
    print()
    if n_int < args.gate_threshold:
        print(f"FAIL: intersection has {n_int} pairs (target >= {args.gate_threshold}).")
        print("Investigate per-model dropped lists before continuing to Phase 2.")
        for au in audits:
            bad_ans = sorted(int(n) for n, ok in au["answer_ok"].items() if not ok)
            print(f"  {au['model']}: multi-token answers = {bad_ans if bad_ans else '(none)'}")
        sys.exit(1)
    else:
        print(f"PASS: intersection has {n_int} pairs (>= {args.gate_threshold}).")
        print("Proceed to Phase 2 (accuracy reproduction).")


if __name__ == "__main__":
    main()
