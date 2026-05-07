"""Phase 1, Step B: comprehensive per-integer tokenization test.

Loads each of the three real tokenizers and enumerates every integer
n ∈ [0, 198] (covers the operand range [0, 99] and the answer range
[0, 198] in one pass). For each (model, n) records:
  - bare-string encode ids + decoded list (canonical, drives retention)
  - leading-space encode ids + decoded (informational)
  - single_token bool, round_trip bool

Also tokenizes 6 canonical sample (a, b) prompts per model and locates
the '=' token position — Phase 3 extracts activations there.

Outputs:
  $BLACKBOX_DATA/tokenizer_audit/tokenization_test_{model}.json
  $BLACKBOX_DATA/tokenizer_audit/tokenization_test_cross_model.json

Hard-fail conditions (halt before audit):
  1. Tokenizer load fails.
  2. Any n ∈ [0, 9] is multi-token in any model.
  3. Any n ∈ [0, 198] fails round-trip in any model.
  4. The '=' token is missing or non-unique in any sample prompt.

Soft-warn conditions:
  - KT's claimed single-token upper bound (KT_paper.md:333-335) does
    not hold for a model. Informational; gate is on n_int.
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
from tokenizer_audit import PROMPT_TEMPLATES  # noqa: E402

MODELS = {
    "gpt-j-6b": "EleutherAI/gpt-j-6B",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}

INTEGER_RANGE = range(0, 199)
OPERAND_RANGE = range(0, 100)
ANSWER_RANGE = range(0, 199)

KT_SINGLE_TOKEN_UPPER_BOUND = {
    "gpt-j-6b": 361,
    "pythia-6.9b": 557,
    "llama-3.1-8b": 999,
}

SAMPLE_PAIRS = [
    (0, 0),
    (1, 1),
    (42, 17),
    (50, 50),
    (99, 99),
    (47, 56),
]


def encode_with_detail(tok, text: str) -> Dict:
    ids = tok.encode(text, add_special_tokens=False)
    return {
        "text": text,
        "ids": [int(i) for i in ids],
        "decoded": [tok.decode([int(i)]) for i in ids],
        "n_tokens": len(ids),
        "single_token": len(ids) == 1,
    }


def round_trip_ok(tok, n: int) -> bool:
    ids = tok.encode(str(n), add_special_tokens=False)
    if not ids:
        return False
    return tok.decode(ids).strip() == str(n)


def per_integer_record(tok, n: int) -> Dict:
    bare = encode_with_detail(tok, str(n))
    leading = encode_with_detail(tok, " " + str(n))
    return {
        "n": n,
        "bare_ids": bare["ids"],
        "bare_decoded": bare["decoded"],
        "bare_single_token": bare["single_token"],
        "leading_space_ids": leading["ids"],
        "leading_space_decoded": leading["decoded"],
        "leading_space_single_token": leading["single_token"],
        "round_trip": round_trip_ok(tok, n),
    }


def sample_prompt_record(tok, model_key: str, a: int, b: int) -> Dict:
    template = PROMPT_TEMPLATES[model_key]
    prompt = template.format(a=a, b=b)
    enc = encode_with_detail(tok, prompt)
    eq_positions = [
        i for i, t in enumerate(enc["decoded"]) if t.strip() == "="
    ]
    return {
        "a": a, "b": b, "prompt": prompt,
        "n_tokens": enc["n_tokens"],
        "ids": enc["ids"],
        "decoded": enc["decoded"],
        "eq_positions": eq_positions,
        "eq_unique": len(eq_positions) == 1,
    }


def test_one_tokenizer(tok, model_key: str, hf_repo: str, log) -> Dict:
    log.info("=" * 78)
    log.info("Tokenization test: %s", model_key)
    log.info(
        "  hf_repo=%s class=%s vocab_size=%d",
        hf_repo, tok.__class__.__name__, tok.vocab_size,
    )
    log.info("=" * 78)

    per_int: Dict[str, Dict] = {}
    for n in INTEGER_RANGE:
        per_int[str(n)] = per_integer_record(tok, n)

    operand_single = sum(
        per_int[str(n)]["bare_single_token"] for n in OPERAND_RANGE
    )
    answer_single = sum(
        per_int[str(n)]["bare_single_token"] for n in ANSWER_RANGE
    )
    rt_fails = [n for n in INTEGER_RANGE if not per_int[str(n)]["round_trip"]]
    operand_multi = sorted([
        n for n in OPERAND_RANGE if not per_int[str(n)]["bare_single_token"]
    ])
    answer_multi = sorted([
        n for n in ANSWER_RANGE if not per_int[str(n)]["bare_single_token"]
    ])

    log.info("Per-integer summary:")
    log.info("  operand range [0, 99]:  %d / 100 single-token", operand_single)
    log.info("  answer range  [0, 198]: %d / 199 single-token", answer_single)
    log.info("  operand multi-token: %s", operand_multi)
    head = answer_multi[:30]
    tail_marker = "..." if len(answer_multi) > 30 else ""
    log.info(
        "  answer multi-token (count=%d): %s%s",
        len(answer_multi), head, tail_marker,
    )
    log.info("  round-trip fails: %s", rt_fails or "(none)")

    upper = KT_SINGLE_TOKEN_UPPER_BOUND[model_key]
    upper_record = per_integer_record(tok, upper)
    log.info(
        "KT-claim verification: upper=%d bare_single=%s ids=%s",
        upper, upper_record["bare_single_token"], upper_record["bare_ids"],
    )

    samples = []
    log.info("Sample prompt tokenizations:")
    for (a, b) in SAMPLE_PAIRS:
        rec = sample_prompt_record(tok, model_key, a, b)
        samples.append(rec)
        log.info(
            "  (a=%2d, b=%2d) n_tokens=%d eq_positions=%s eq_unique=%s",
            a, b, rec["n_tokens"], rec["eq_positions"], rec["eq_unique"],
        )

    fails = []
    for n in range(0, 10):
        if not per_int[str(n)]["bare_single_token"]:
            fails.append(f"single_digit_{n}_multi_token")
    if rt_fails:
        fails.append(f"round_trip_fail_for: {rt_fails[:10]}")
    for s in samples:
        if not s["eq_unique"]:
            fails.append(f"eq_position_not_unique_for_({s['a']},{s['b']})")

    return {
        "model": model_key,
        "hf_repo": hf_repo,
        "tokenizer_class": tok.__class__.__name__,
        "vocab_size": int(tok.vocab_size),
        "prompt_template": PROMPT_TEMPLATES[model_key],
        "per_integer": per_int,
        "operand_single_token_count": operand_single,
        "answer_single_token_count": answer_single,
        "operand_multi_token_list": operand_multi,
        "answer_multi_token_list": answer_multi,
        "kt_claimed_upper": upper,
        "kt_claim_holds": bool(upper_record["bare_single_token"]),
        "round_trip_fails": rt_fails,
        "sample_prompts": samples,
        "hard_fails": fails,
    }


def cross_model_summary(results: List[Dict], log) -> Dict:
    operand_multi_any = set()
    answer_multi_any = set()
    for r in results:
        operand_multi_any.update(r["operand_multi_token_list"])
        answer_multi_any.update(r["answer_multi_token_list"])

    excluded_pairs = 0
    for a in OPERAND_RANGE:
        for b in OPERAND_RANGE:
            s = a + b
            if (
                a in operand_multi_any
                or b in operand_multi_any
                or s in answer_multi_any
            ):
                excluded_pairs += 1
    predicted = 10000 - excluded_pairs

    log.info("=" * 78)
    log.info("CROSS-MODEL PREVIEW (strict-primary intersection)")
    log.info("=" * 78)
    log.info(
        "  operand multi-token (any model): %s", sorted(operand_multi_any)
    )
    answer_any_sorted = sorted(answer_multi_any)
    head = answer_any_sorted[:30]
    tail_marker = "..." if len(answer_any_sorted) > 30 else ""
    log.info(
        "  answer multi-token (any model) count=%d: %s%s",
        len(answer_any_sorted), head, tail_marker,
    )
    log.info(
        "  predicted strict-primary n_int = %d / 10000 "
        "(gate >= 7000 -> %s)",
        predicted, "PASS" if predicted >= 7000 else "FAIL",
    )
    return {
        "operand_multi_token_any_model": sorted(operand_multi_any),
        "answer_multi_token_any_model": sorted(answer_multi_any),
        "predicted_strict_intersection": predicted,
        "predicted_gate_passes": predicted >= 7000,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("BLACKBOX_DATA", "/data/user_data/anshulk/blackbox"),
    )
    args = ap.parse_args()

    log_path = default_log_path(args.data_dir, "phase1_tokenization_test")
    log = setup_logger(__name__, log_path=log_path)
    log.info("Comprehensive tokenization test start; log -> %s", log_path)

    out_dir = os.path.join(args.data_dir, "tokenizer_audit")
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoTokenizer

    results: List[Dict] = []
    hard_fail_total = 0
    for key, repo in MODELS.items():
        log.info("Loading tokenizer: %s (%s)", key, repo)
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as exc:
            log.error("FAIL: tokenizer load error for %s: %s", key, exc)
            sys.exit(1)
        r = test_one_tokenizer(tok, key, repo, log)
        results.append(r)

        out_path = os.path.join(out_dir, f"tokenization_test_{key}.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)
        log.info("Saved per-integer table -> %s", out_path)

        if r["hard_fails"]:
            log.error("HARD FAIL for %s: %s", key, r["hard_fails"])
            hard_fail_total += 1
        if not r["kt_claim_holds"]:
            log.warning(
                "Soft-warn: KT's single-token upper bound (%d) does NOT "
                "hold for %s",
                r["kt_claimed_upper"], key,
            )

    cross = cross_model_summary(results, log)
    cross_path = os.path.join(out_dir, "tokenization_test_cross_model.json")
    with open(cross_path, "w") as f:
        json.dump(cross, f, indent=2)
    log.info("Saved cross-model preview -> %s", cross_path)

    if hard_fail_total:
        log.error(
            "Tokenization test FAIL for %d model(s). Halt before audit.",
            hard_fail_total,
        )
        sys.exit(1)
    log.info("Tokenization test PASS for all 3 models. Proceed to audit.")


if __name__ == "__main__":
    main()
