"""Phase 1, Step A: build the canonical 10K-pair addition dataset.

Per full_paper_plan.md §3.2:1587, the primary dataset is all ordered
pairs (a, b) ∈ {0,…,99}² with s = a + b ∈ {0,…,198}. Tokenizer-
independent; per-model single-token retention is recorded separately
in tokenizer_audit/{model}.json.

Single source of truth for ground-truth s across every downstream
phase. Phase 2 reads it for ordering; Phase 3 reads it for activation-
extraction indexing; Phase 5 reads it for the matched-permutation
features φ defined in full_paper_plan.md §3.3:1691-1702.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from _logging import setup_logger, default_log_path  # noqa: E402

OPERAND_RANGE = range(0, 100)


def build_dataset() -> Dict:
    pairs: List[Dict] = []
    pid = 0
    for a in OPERAND_RANGE:
        for b in OPERAND_RANGE:
            s = a + b
            if s < 50:
                sum_bin = 0
            elif s < 100:
                sum_bin = 1
            elif s < 150:
                sum_bin = 2
            else:
                sum_bin = 3
            pairs.append({
                "id": pid,
                "a": a,
                "b": b,
                "s": s,
                "carry_ones": int((a % 10) + (b % 10) >= 10),
                "carry_tens": int(s >= 100),
                "a_decile": a // 10,
                "b_decile": b // 10,
                "sum_bin": sum_bin,
            })
            pid += 1
    return {
        "n_pairs": len(pairs),
        "operand_range": [0, 99],
        "answer_range": [0, 198],
        "ordered": True,
        "spec": "full_paper_plan.md §3.2:1587 — 10,000 ordered pairs",
        "matched_permutation_features": (
            "carry_ones, carry_tens, a_decile, b_decile, sum_bin "
            "(full_paper_plan.md §3.3:1691-1702)"
        ),
        "pairs": pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("BLACKBOX_DATA", "/data/user_data/anshulk/blackbox"),
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    log_path = default_log_path(args.data_dir, "phase1_dataset")
    log = setup_logger(__name__, log_path=log_path)
    log.info("Starting dataset build; log -> %s", log_path)

    out_dir = os.path.join(args.data_dir, "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, "dataset_10k.json")

    data = build_dataset()
    assert data["n_pairs"] == 10000, data["n_pairs"]
    assert all(p["s"] == p["a"] + p["b"] for p in data["pairs"])
    assert all(0 <= p["s"] <= 198 for p in data["pairs"])
    assert all(0 <= p["a"] <= 99 for p in data["pairs"])
    assert all(0 <= p["b"] <= 99 for p in data["pairs"])
    assert data["pairs"][0] == {
        "id": 0, "a": 0, "b": 0, "s": 0,
        "carry_ones": 0, "carry_tens": 0,
        "a_decile": 0, "b_decile": 0, "sum_bin": 0,
    }
    assert data["pairs"][9999]["a"] == 99
    assert data["pairs"][9999]["b"] == 99
    assert data["pairs"][9999]["s"] == 198

    p_47_56 = data["pairs"][47 * 100 + 56]
    assert p_47_56["a"] == 47 and p_47_56["b"] == 56 and p_47_56["s"] == 103
    assert p_47_56["carry_ones"] == 1 and p_47_56["carry_tens"] == 1

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    n_carry_ones = sum(p["carry_ones"] for p in data["pairs"])
    n_carry_tens = sum(p["carry_tens"] for p in data["pairs"])

    log.info("Wrote %d pairs to %s", data["n_pairs"], out_path)
    log.info("  first: %s", data["pairs"][0])
    log.info("  last:  %s", data["pairs"][-1])
    log.info(
        "  carry distribution: ones=%d  tens=%d  (out of %d)",
        n_carry_ones, n_carry_tens, data["n_pairs"],
    )
    log.info(
        "  sum_bin distribution: %s",
        {
            b: sum(1 for p in data["pairs"] if p["sum_bin"] == b)
            for b in range(4)
        },
    )


if __name__ == "__main__":
    main()
