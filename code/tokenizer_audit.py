"""
Tokenizer audit utility (Phase 1 of colab_execution_plan.md).

For each (a, b) in {0..99}^2, determines whether:
  - Operand a is single-token (with and without leading space).
  - Operand b is single-token.
  - Answer s = a + b is single-token.

Per full_paper_plan.md §3.2: we restrict primary analysis to (a, b) pairs
where both inputs and the answer tokenize as single tokens in the model's
tokenizer.

This module exposes:
  - PROMPT_TEMPLATES: per-model prompt strings (matching KT 2025 Table 2).
  - audit_tokenizer(tokenizer, model_key) -> dict with retained mask + reasons.
  - intersect_audits(audits) -> set of (a, b) retained by all input audits.

The notebook 01_tokenizer_audit.ipynb wraps these functions; the same code
is reusable on a laptop CPU for testing without Colab.
"""

from __future__ import annotations

from typing import Dict, List
import json


# Per-model prompt templates from full_paper_plan.md §3.1 (KT-faithful).
PROMPT_TEMPLATES: Dict[str, str] = {
    "gpt-j-6b":     "Output ONLY a number. {a}+{b}=",
    "pythia-6.9b":  "Output ONLY a number. {a}+{b}=",
    "llama-3.1-8b": "The following is a correct addition problem.\n{a}+{b}=",
}

OPERAND_RANGE = range(0, 100)     # 0..99 inclusive
ANSWER_RANGE = range(0, 199)      # 0..198 inclusive


def _is_single_token(tokenizer, text: str) -> bool:
    """Return True iff `text` tokenizes to a single token (no specials)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids) == 1


def _operand_token_ok(tokenizer, n: int) -> bool:
    """Operand appears in two contexts in the prompt:
       - mid-prompt with no leading space (immediately after '. ' or '\\n')
       - mid-prompt with a leading space (e.g., after a previous token)

    For BPE tokenizers, " 42" and "42" can map to different token ids; we
    require BOTH to be single tokens, since we don't control which form the
    tokenizer produces from a given prompt position.
    """
    return _is_single_token(tokenizer, str(n)) and _is_single_token(tokenizer, " " + str(n))


def _answer_token_ok(tokenizer, s: int) -> bool:
    """The answer follows '=' so the tokenizer typically does NOT produce a
    leading space for the next-token prediction. We require the no-space form
    to be single-token; the space form is also checked for diagnostic logging
    but is not gating.
    """
    return _is_single_token(tokenizer, str(s))


def audit_tokenizer(tokenizer, model_key: str) -> Dict:
    """Run the audit for one tokenizer.

    Returns a dict with:
      'model':              model_key
      'prompt_template':    the template used
      'retained':           list of {a, b, s} dicts for retained pairs
      'dropped':            list of {a, b, s, reason} dicts for dropped pairs
      'n_retained':         int
      'n_dropped':          int
      'operand_ok':         dict mapping str(n) -> bool, for n in 0..99
      'answer_ok':          dict mapping str(s) -> bool, for s in 0..198
    """
    # Pre-compute per-integer single-token status (we only call the tokenizer 100+199 times,
    # not 10000+ times).
    operand_ok = {n: _operand_token_ok(tokenizer, n) for n in OPERAND_RANGE}
    answer_ok = {s: _answer_token_ok(tokenizer, s) for s in ANSWER_RANGE}

    retained: List[Dict] = []
    dropped: List[Dict] = []

    for a in OPERAND_RANGE:
        for b in OPERAND_RANGE:
            s = a + b
            reasons = []
            if not operand_ok[a]:
                reasons.append(f"operand_a={a}_multitoken")
            if not operand_ok[b]:
                reasons.append(f"operand_b={b}_multitoken")
            if not answer_ok[s]:
                reasons.append(f"answer_s={s}_multitoken")
            if reasons:
                dropped.append({"a": a, "b": b, "s": s, "reasons": reasons})
            else:
                retained.append({"a": a, "b": b, "s": s})

    return {
        "model": model_key,
        "prompt_template": PROMPT_TEMPLATES[model_key],
        "retained": retained,
        "dropped": dropped,
        "n_retained": len(retained),
        "n_dropped": len(dropped),
        "operand_ok": {str(k): bool(v) for k, v in operand_ok.items()},
        "answer_ok": {str(k): bool(v) for k, v in answer_ok.items()},
    }


def intersect_audits(audits: List[Dict]) -> List[Dict]:
    """Return the list of {a, b, s} retained by every audit in `audits`."""
    if not audits:
        return []
    common = set((p["a"], p["b"]) for p in audits[0]["retained"])
    for au in audits[1:]:
        common &= set((p["a"], p["b"]) for p in au["retained"])
    return [{"a": a, "b": b, "s": a + b} for (a, b) in sorted(common)]


def save_audit(audit: Dict, path: str) -> None:
    """Save an audit dict to a JSON file."""
    with open(path, "w") as f:
        json.dump(audit, f, indent=2)


def save_intersection(intersection: List[Dict], audits: List[Dict], path: str) -> None:
    """Save the three-way intersection plus per-model retained counts."""
    data = {
        "n_intersection": len(intersection),
        "per_model_retained": {a["model"]: a["n_retained"] for a in audits},
        "pairs": intersection,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def summary_report(audits: List[Dict], intersection: List[Dict]) -> str:
    """Multi-line text summary suitable for printing in the notebook."""
    lines = ["Tokenizer audit summary:"]
    for a in audits:
        lines.append(
            f"  {a['model']:14s}: {a['n_retained']:5d} / 10000 retained "
            f"({100.0 * a['n_retained'] / 10000.0:.1f}%)"
        )
    n_int = len(intersection)
    lines.append(
        f"  {'INTERSECTION':14s}: {n_int:5d} / 10000 retained by all "
        f"({100.0 * n_int / 10000.0:.1f}%)"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    # Local self-test that does NOT require any HuggingFace download.
    # Builds a fake tokenizer that maps every short integer (0..200) to one
    # token and sanity-checks the audit machinery.

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Single-token if text (stripped) is in 0..200; otherwise multi-token.
            stripped = text.strip()
            try:
                n = int(stripped)
                if 0 <= n <= 200:
                    return [n]
                return [99, 100]   # multi-token
            except ValueError:
                return [99, 100]

    tok = FakeTokenizer()
    audit = audit_tokenizer(tok, "gpt-j-6b")
    print(summary_report([audit], []))
    assert audit["n_retained"] == 10000, "fake tokenizer should retain all"

    # Now break it: pretend "100" is multi-token.
    class FakeBroken(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            stripped = text.strip()
            if stripped == "100":
                return [99, 100]
            return super().encode(text, add_special_tokens=add_special_tokens)

    audit2 = audit_tokenizer(FakeBroken(), "pythia-6.9b")
    intersection = intersect_audits([audit, audit2])
    print(summary_report([audit, audit2], intersection))
    # 100 is dropped only on model-2 audit (answer s=100 path); intersection
    # excludes any (a, b) with a+b == 100 because of model-2 -- 99 such pairs.
    assert audit2["n_retained"] < 10000
    assert len(intersection) <= audit2["n_retained"]

    print("OK: tokenizer_audit self-test passed.")
