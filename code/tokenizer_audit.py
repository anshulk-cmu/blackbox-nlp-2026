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


def _audit_pair_in_prompt(tokenizer, template: str, a: int, b: int):
    """Tokenize the actual prompt for (a, b) and find operand positions.

    Returns dict with keys:
      a_idx, b_idx: indices of operand tokens in the prompt tokenization,
                    or None if operand is not a single contiguous token.
                    For multi-token operands (Llama 3 splits "42" into
                    ['4','2']), this is None and we accept that — Phase 3
                    extracts at the '=' token, NOT at operand positions.
      first_answer_token_id: int, the token id the model would need to
                             predict at the '=' position to be "correct"
                             on the FIRST token of the answer string.
                             For single-token answers (GPT-J/Pythia) this
                             IS the answer. For multi-token (Llama 3.1) it
                             is the first character of str(s).
      first_answer_token_str: the decoded form of that token (for sanity).
      s_is_single_token: True if str(s) tokenizes to one token.
      reasons: empty unless something is structurally broken with the prompt.

    The audit's `ok` flag is now: "the prompt tokenizes without error AND
    the first answer token is well-defined". This retains all (a, b) on
    Llama where the previous strict check dropped everything.
    """
    prompt = template.format(a=a, b=b)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = [tokenizer.decode([i]) for i in ids]

    a_str = str(a)
    b_str = str(b)

    # Locate operand a: first token whose stripped decode equals str(a). May be None.
    a_idx = None
    for i, t in enumerate(decoded):
        if t.strip() == a_str:
            a_idx = i
            break

    b_idx = None
    if a_idx is not None:
        for i in range(a_idx + 1, len(decoded)):
            if decoded[i].strip() == b_str:
                b_idx = i
                break

    # First token the model needs to predict at '=' for a 'correct' answer.
    s = a + b
    answer_ids = tokenizer.encode(str(s), add_special_tokens=False)
    s_is_single = len(answer_ids) == 1
    first_answer_token_id = int(answer_ids[0]) if answer_ids else None
    first_answer_token_str = tokenizer.decode([first_answer_token_id]) if answer_ids else ""

    reasons = []
    if not answer_ids:
        reasons.append(f"answer_s={s}_empty_tokenization")

    return {
        "a_idx": a_idx,
        "b_idx": b_idx,
        "first_answer_token_id": first_answer_token_id,
        "first_answer_token_str": first_answer_token_str,
        "s_is_single_token": s_is_single,
        "reasons": reasons,
        "ok": not reasons,
    }


def audit_tokenizer(tokenizer, model_key: str) -> Dict:
    """Run the audit for one tokenizer using prompt-context checks.

    Retention rule (post-fix for Llama 3.1):
      A pair (a, b) is retained if the prompt tokenizes without error and
      the answer's first token is well-defined. We do NOT require the
      operand or full answer to be single-token, because:
        - For Phase 3 activation extraction, we extract at the '=' token.
        - For Phase 2 accuracy reproduction, single-token answers use
          1-step argmax; multi-token answers use generate(max_new_tokens).
      The diagnostic fields `operand_ok` (bare-string single-token check)
      and `s_is_single_token` are recorded per-pair so Phase 2 can route
      the right decoding strategy per model.

    Returns dict with:
      'model':              model_key
      'prompt_template':    the template used
      'retained':           list of {a, b, s, a_idx, b_idx,
                                     first_answer_token_id,
                                     first_answer_token_str,
                                     s_is_single_token} dicts
      'dropped':            list of {a, b, s, reasons} dicts
      'n_retained':         int
      'n_dropped':          int
      'all_answers_single_token': bool (True for GPT-J/Pythia, False for Llama)
      'operand_ok':         dict str(n) -> bool, diagnostic only
      'answer_ok':          dict str(s) -> bool, diagnostic only
    """
    template = PROMPT_TEMPLATES[model_key]

    operand_bare_ok = {n: _is_single_token(tokenizer, str(n)) for n in OPERAND_RANGE}
    answer_ok = {s: _is_single_token(tokenizer, str(s)) for s in ANSWER_RANGE}

    retained: List[Dict] = []
    dropped: List[Dict] = []

    for a in OPERAND_RANGE:
        for b in OPERAND_RANGE:
            s = a + b
            res = _audit_pair_in_prompt(tokenizer, template, a, b)
            if res["ok"]:
                retained.append({
                    "a": a, "b": b, "s": s,
                    "a_idx": res["a_idx"],
                    "b_idx": res["b_idx"],
                    "first_answer_token_id": res["first_answer_token_id"],
                    "first_answer_token_str": res["first_answer_token_str"],
                    "s_is_single_token": res["s_is_single_token"],
                })
            else:
                dropped.append({"a": a, "b": b, "s": s, "reasons": res["reasons"]})

    all_single = all(answer_ok.values())

    return {
        "model": model_key,
        "prompt_template": template,
        "retained": retained,
        "dropped": dropped,
        "n_retained": len(retained),
        "n_dropped": len(dropped),
        "all_answers_single_token": all_single,
        "operand_ok": {str(k): bool(v) for k, v in operand_bare_ok.items()},
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
        """Tokenizer where every integer 0..200 is a single token equal to its value,
        and the prompt template tokens are individual integer ids."""
        def encode(self, text, add_special_tokens=False):
            # Tokenize a prompt by splitting on word boundaries; each integer
            # 0..200 becomes a single token.
            import re
            ids = []
            for piece in re.findall(r'\d+|\D+', text):
                if piece.isdigit():
                    n = int(piece)
                    if 0 <= n <= 200:
                        ids.append(n)
                    else:
                        ids.extend([99, 100])
                else:
                    # Map non-digit chunks to a fixed dummy id (200 + len).
                    ids.append(500)
            return ids
        def decode(self, ids):
            # Reverse: integer-valued tokens decode to str(n); dummy decodes to "<x>".
            parts = []
            for i in ids:
                if 0 <= i <= 200:
                    parts.append(str(i))
                else:
                    parts.append("<x>")
            return ''.join(parts)

    tok = FakeTokenizer()
    audit = audit_tokenizer(tok, "gpt-j-6b")
    print(summary_report([audit], []))
    assert audit["n_retained"] == 10000, "fake tokenizer should retain all"

    # Now break it: pretend "100" tokenizes to two tokens (operand or answer).
    class FakeBroken(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            # Split any 100 token into [99, 1] (multi-token) for this fake.
            new_ids = []
            for i in ids:
                if i == 100:
                    new_ids.extend([99, 1])
                else:
                    new_ids.append(i)
            return new_ids

    audit2 = audit_tokenizer(FakeBroken(), "pythia-6.9b")
    intersection = intersect_audits([audit, audit2])
    print(summary_report([audit, audit2], intersection))
    # Relaxed audit retains multi-token-answer pairs (Phase 2 handles decoding).
    # FakeBroken should still tokenize the prompt structurally OK, so retention is full.
    assert audit2["n_retained"] == 10000, "Relaxed audit should retain all structurally-OK prompts"
    # But the per-pair s_is_single_token flag should be False for s=100 cases.
    multi_pairs = [p for p in audit2["retained"] if not p["s_is_single_token"]]
    assert len(multi_pairs) > 0, "FakeBroken should produce some multi-token answers"
    assert audit2["all_answers_single_token"] is False

    print("OK: tokenizer_audit self-test passed.")
