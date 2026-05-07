"""Phase 1 driver per babel_execution_plan.md §4. CPU-only; ~5 min.

Strict-primary protocol from full_paper_plan.md §3.1/§3.2; Llama-only
relaxed fallback from babel_execution_plan.md §4:340-348 if and only if
the strict gate (n_∩ >= 7000) fails with Llama as the bottleneck. Per
§4:370-373, halt if the gate fails for non-Llama reasons.

Outputs (all under $BLACKBOX_DATA/tokenizer_audit/):
  - {gpt-j-6b,pythia-6.9b,llama-3.1-8b}.json — per-model audit
  - intersection.json — three-way intersection + gate decision
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from _logging import setup_logger, default_log_path  # noqa: E402
from tokenizer_audit import (  # noqa: E402
    audit_tokenizer_strict,
    audit_tokenizer_relaxed_llama,
    intersect_audits,
    save_audit,
    save_intersection,
    summary_report,
)

MODELS = {
    "gpt-j-6b": "EleutherAI/gpt-j-6B",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("BLACKBOX_DATA", "/data/user_data/anshulk/blackbox"),
    )
    ap.add_argument(
        "--gate-threshold", type=int, default=7000,
        help="Pre-registered gate from babel_execution_plan.md §16:1414.",
    )
    ap.add_argument(
        "--llama-relaxed-trigger", type=int, default=5000,
        help="If strict |R_llama| < this AND gate fails, activate relaxed Llama "
             "(babel_execution_plan.md §4:340).",
    )
    args = ap.parse_args()

    log_path = default_log_path(args.data_dir, "phase1_audit")
    log = setup_logger(__name__, log_path=log_path)
    log.info("Phase 1 audit start; log -> %s", log_path)
    log.info("data_dir=%s gate=%d", args.data_dir, args.gate_threshold)

    audit_dir = os.path.join(args.data_dir, "tokenizer_audit")
    os.makedirs(audit_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizers: Dict[str, object] = {}
    for key, repo in MODELS.items():
        log.info("Loading tokenizer: %s (%s)", key, repo)
        t0 = time.time()
        tokenizers[key] = AutoTokenizer.from_pretrained(repo)
        log.info(
            "  class=%s vocab_size=%d  loaded in %.1fs",
            tokenizers[key].__class__.__name__,
            tokenizers[key].vocab_size,
            time.time() - t0,
        )

    audits = []
    for key, repo in MODELS.items():
        log.info("Strict audit: %s", key)
        au = audit_tokenizer_strict(tokenizers[key], key, repo)
        audits.append(au)
        out = os.path.join(audit_dir, f"{key}.json")
        save_audit(au, out)
        log.info(
            "  retained=%d dropped=%d all_ans_single=%s",
            au["n_retained"], au["n_dropped"], au["all_answers_single_token"],
        )
        log.info("  operand_multi=%s", au["operand_multi_token_list"])
        ans_multi = au["answer_multi_token_list"]
        log.info(
            "  answer_multi (count=%d): %s%s",
            len(ans_multi),
            ans_multi[:30],
            "..." if len(ans_multi) > 30 else "",
        )
        log.info("  saved -> %s", out)

    intersection = intersect_audits(audits)
    n_int = len(intersection)
    n_per = {au["model"]: au["n_retained"] for au in audits}
    log.info("Strict three-way intersection: %d", n_int)
    for k, v in n_per.items():
        log.info("  %s: %d", k, v)

    relaxed_used = False

    if n_int >= args.gate_threshold:
        log.info("Gate PASS (strict): n_int=%d >= %d", n_int, args.gate_threshold)
    else:
        llama_n = n_per["llama-3.1-8b"]
        gj_n = n_per["gpt-j-6b"]
        py_n = n_per["pythia-6.9b"]
        llama_bottleneck = (llama_n < gj_n) and (llama_n < py_n)

        if llama_bottleneck and llama_n < args.llama_relaxed_trigger:
            log.warning(
                "Gate FAIL with Llama bottleneck (llama=%d < %d). "
                "Activating relaxed Llama protocol (babel §4:340-348).",
                llama_n, args.llama_relaxed_trigger,
            )
            au_relaxed = audit_tokenizer_relaxed_llama(
                tokenizers["llama-3.1-8b"],
                "llama-3.1-8b",
                MODELS["llama-3.1-8b"],
            )
            audits[2] = au_relaxed
            save_audit(au_relaxed, os.path.join(audit_dir, "llama-3.1-8b.json"))
            log.info(
                "  relaxed_llama retained=%d (was strict=%d)",
                au_relaxed["n_retained"], llama_n,
            )
            relaxed_used = True
            intersection = intersect_audits(audits)
            n_int = len(intersection)
            n_per = {au["model"]: au["n_retained"] for au in audits}
            log.info("  new intersection: %d", n_int)
        else:
            log.error(
                "Gate FAIL but Llama is NOT the bottleneck "
                "(gpt-j=%d, pythia=%d, llama=%d). "
                "Per babel_execution_plan.md §4:370-373, halt and inspect.",
                gj_n, py_n, llama_n,
            )

    intersect_path = os.path.join(audit_dir, "intersection.json")
    save_intersection(
        intersection, audits, intersect_path,
        args.gate_threshold, relaxed_used,
    )
    log.info("Saved intersection -> %s", intersect_path)
    log.info("\n%s", summary_report(audits, n_int, args.gate_threshold))

    if n_int < args.gate_threshold:
        log.error("FAIL: n_int=%d < %d. Do not proceed to Phase 2.",
                  n_int, args.gate_threshold)
        for au in audits:
            log.error(
                "  %s answer_multi_token=%s",
                au["model"], au["answer_multi_token_list"][:30],
            )
        sys.exit(1)
    log.info("PASS: n_int=%d >= %d. Phase 2 unblocked.",
             n_int, args.gate_threshold)


if __name__ == "__main__":
    main()
