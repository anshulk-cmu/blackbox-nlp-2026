"""Phase 1 utility: strict tokenizer audit per the project spec.

Strict primary protocol from full_paper_plan.md §3.1:1551-1554 +
§3.2:1587-1605: a pair (a, b) ∈ {0,…,99}² is retained for a model
iff str(a), str(b), and str(a+b) each tokenize to a single token.
The relaxed Llama-only fallback (operands single-token, allow multi-
token answer, predict on the first answer token) is documented at
babel_execution_plan.md §4:340-348 and is invoked from
run_phase1_audit.py only when the strict gate fails with Llama as
the bottleneck.

Public API:
  - PROMPT_TEMPLATES (full_paper_plan.md §3.1:1542-1548, KT-faithful)
  - audit_tokenizer_strict(tokenizer, model_key, hf_repo) -> dict
  - audit_tokenizer_relaxed_llama(tokenizer, model_key, hf_repo) -> dict
  - intersect_audits(audits) -> List[Dict]   (pairs with per-model token ids)
  - save_audit, save_intersection, summary_report

The module's __main__ runs a FakeTokenizer self-test that exercises
the audit machinery without any HF download.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import json

# Per full_paper_plan.md §3.1:1542-1548 (KT-faithful, KT Table 2).
PROMPT_TEMPLATES: Dict[str, str] = {
    "gpt-j-6b": "Output ONLY a number. {a}+{b}=",
    "pythia-6.9b": "Output ONLY a number. {a}+{b}=",
    "llama-3.1-8b": "The following is a correct addition problem.\n{a}+{b}=",
}

OPERAND_RANGE = range(0, 100)   # 0..99 inclusive (full_paper_plan.md §3.2)
ANSWER_RANGE = range(0, 199)    # 0..198 inclusive (paper_math.md §1.1)


def _is_single_token(tokenizer, text: str) -> bool:
    return len(tokenizer.encode(text, add_special_tokens=False)) == 1


def _first_token(tokenizer, text: str) -> Tuple[Optional[int], str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return (None, "")
    first_id = int(ids[0])
    return (first_id, tokenizer.decode([first_id]))


def _per_integer_single_token(tokenizer) -> Tuple[Dict[int, bool], Dict[int, bool]]:
    operand_ok = {n: _is_single_token(tokenizer, str(n)) for n in OPERAND_RANGE}
    answer_ok = {s: _is_single_token(tokenizer, str(s)) for s in ANSWER_RANGE}
    return operand_ok, answer_ok


def audit_tokenizer_strict(tokenizer, model_key: str, hf_repo: str) -> Dict:
    """Strict primary protocol per full_paper_plan.md §3.1/§3.2.

    Retain (a, b) iff str(a), str(b), str(a+b) are each length-1 tokens.
    """
    operand_ok, answer_ok = _per_integer_single_token(tokenizer)

    retained: List[Dict] = []
    dropped: List[Dict] = []

    for a in OPERAND_RANGE:
        for b in OPERAND_RANGE:
            s = a + b
            ok_a, ok_b, ok_s = operand_ok[a], operand_ok[b], answer_ok[s]
            if ok_a and ok_b and ok_s:
                first_id, first_str = _first_token(tokenizer, str(s))
                retained.append({
                    "a": a, "b": b, "s": s,
                    "ok_a": True, "ok_b": True, "ok_s": True,
                    "first_answer_token_id": first_id,
                    "first_answer_token_str": first_str,
                })
            else:
                reason_parts = []
                if not ok_a:
                    reason_parts.append(f"operand_a={a}_multi")
                if not ok_b:
                    reason_parts.append(f"operand_b={b}_multi")
                if not ok_s:
                    reason_parts.append(f"answer_s={s}_multi")
                dropped.append({
                    "a": a, "b": b, "s": s,
                    "ok_a": ok_a, "ok_b": ok_b, "ok_s": ok_s,
                    "reason": ",".join(reason_parts),
                })

    return _build_audit_dict(
        tokenizer, model_key, hf_repo, retained, dropped,
        operand_ok, answer_ok, policy="strict",
    )


def audit_tokenizer_relaxed_llama(tokenizer, model_key: str, hf_repo: str) -> Dict:
    """Llama-only relaxed fallback per babel_execution_plan.md §4:340-348.

    Operand must be single-token; answer is allowed to be multi-token.
    Use the FIRST answer token for prediction (model emits one token in
    one greedy step at the '=' position).
    """
    operand_ok, answer_ok = _per_integer_single_token(tokenizer)

    retained: List[Dict] = []
    dropped: List[Dict] = []

    for a in OPERAND_RANGE:
        for b in OPERAND_RANGE:
            s = a + b
            ok_a, ok_b = operand_ok[a], operand_ok[b]
            if not (ok_a and ok_b):
                reason_parts = []
                if not ok_a:
                    reason_parts.append(f"operand_a={a}_multi")
                if not ok_b:
                    reason_parts.append(f"operand_b={b}_multi")
                dropped.append({
                    "a": a, "b": b, "s": s,
                    "ok_a": ok_a, "ok_b": ok_b, "ok_s": answer_ok[s],
                    "reason": ",".join(reason_parts),
                })
                continue

            first_id, first_str = _first_token(tokenizer, str(s))
            if first_id is None:
                dropped.append({
                    "a": a, "b": b, "s": s,
                    "ok_a": ok_a, "ok_b": ok_b, "ok_s": False,
                    "reason": "answer_empty_tokenization",
                })
                continue

            retained.append({
                "a": a, "b": b, "s": s,
                "ok_a": True, "ok_b": True, "ok_s": answer_ok[s],
                "first_answer_token_id": first_id,
                "first_answer_token_str": first_str,
            })

    return _build_audit_dict(
        tokenizer, model_key, hf_repo, retained, dropped,
        operand_ok, answer_ok, policy="relaxed_llama",
    )


def _build_audit_dict(
    tokenizer, model_key: str, hf_repo: str,
    retained: List[Dict], dropped: List[Dict],
    operand_ok: Dict[int, bool], answer_ok: Dict[int, bool],
    policy: str,
) -> Dict:
    return {
        "model": model_key,
        "hf_repo": hf_repo,
        "prompt_template": PROMPT_TEMPLATES[model_key],
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "n_retained": len(retained),
        "n_dropped": len(dropped),
        "all_answers_single_token": all(answer_ok.values()),
        "operand_single_token": {str(n): bool(v) for n, v in operand_ok.items()},
        "answer_single_token": {str(s): bool(v) for s, v in answer_ok.items()},
        "operand_multi_token_list": sorted([n for n, v in operand_ok.items() if not v]),
        "answer_multi_token_list": sorted([s for s, v in answer_ok.items() if not v]),
        "retained": retained,
        "dropped": dropped,
        "policy": policy,
    }


def intersect_audits(audits: List[Dict]) -> List[Dict]:
    """Three-way intersection joined by (a, b).

    Each output pair carries `first_answer_token_id_per_model` so Phase 2
    needs no cross-file join.
    """
    if not audits:
        return []

    sets = [set((p["a"], p["b"]) for p in au["retained"]) for au in audits]
    common = sets[0].intersection(*sets[1:])

    by_model: Dict[str, Dict[Tuple[int, int], Dict]] = {
        au["model"]: {(p["a"], p["b"]): p for p in au["retained"]}
        for au in audits
    }
    by_model_str: Dict[str, Dict[Tuple[int, int], Dict]] = {
        au["model"]: {(p["a"], p["b"]): p for p in au["retained"]}
        for au in audits
    }

    out: List[Dict] = []
    for (a, b) in sorted(common):
        s = a + b
        first_ids = {
            m: by_model[m][(a, b)]["first_answer_token_id"] for m in by_model
        }
        first_strs = {
            m: by_model_str[m][(a, b)]["first_answer_token_str"]
            for m in by_model_str
        }
        out.append({
            "a": a, "b": b, "s": s,
            "first_answer_token_id_per_model": first_ids,
            "first_answer_token_str_per_model": first_strs,
        })
    return out


def save_audit(audit: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(audit, f, indent=2)


def save_intersection(
    pairs: List[Dict],
    audits: List[Dict],
    path: str,
    gate_threshold: int,
    llama_relaxed_used: bool,
) -> None:
    n = len(pairs)
    data = {
        "n_intersection": n,
        "per_model_retained": {a["model"]: a["n_retained"] for a in audits},
        "per_model_policy": {a["model"]: a["policy"] for a in audits},
        "gate_threshold": gate_threshold,
        "gate_passed": n >= gate_threshold,
        "llama_relaxed_fallback_used": llama_relaxed_used,
        "pairs": pairs,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def summary_report(
    audits: List[Dict], n_intersection: int, gate_threshold: int,
) -> str:
    lines = ["Tokenizer audit summary:"]
    for au in audits:
        pct = 100.0 * au["n_retained"] / 10000.0
        lines.append(
            f"  {au['model']:14s} ({au['policy']:14s}): "
            f"{au['n_retained']:5d} / 10000 ({pct:5.1f}%)  "
            f"all_ans_single={str(au['all_answers_single_token']):5s}  "
            f"|operand_multi|={len(au['operand_multi_token_list']):3d}  "
            f"|answer_multi|={len(au['answer_multi_token_list']):3d}"
        )
    pct_int = 100.0 * n_intersection / 10000.0
    gate_str = "PASS" if n_intersection >= gate_threshold else "FAIL"
    lines.append(
        f"  {'INTERSECTION':14s}: {n_intersection:5d} / 10000 "
        f"({pct_int:5.1f}%)  gate(>={gate_threshold})={gate_str}"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# FakeTokenizer self-test (no HF download).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import re

    class FakeTokenizer:
        """Every integer 0..200 maps to a single token equal to its value."""
        vocab_size = 1000

        def encode(self, text, add_special_tokens=False):
            ids: List[int] = []
            for piece in re.findall(r"\d+|\D+", text):
                if piece.isdigit():
                    n = int(piece)
                    if 0 <= n <= 200:
                        ids.append(n)
                    else:
                        ids.extend([99, 100])
                else:
                    ids.append(500)
            return ids

        def decode(self, ids):
            parts = []
            for i in ids:
                if 0 <= i <= 200:
                    parts.append(str(i))
                else:
                    parts.append("<x>")
            return "".join(parts)

    tok_ok = FakeTokenizer()
    audit_ok = audit_tokenizer_strict(tok_ok, "gpt-j-6b", "fake/ok")
    assert audit_ok["n_retained"] == 10000, audit_ok["n_retained"]
    assert audit_ok["n_dropped"] == 0
    assert audit_ok["all_answers_single_token"] is True
    assert audit_ok["operand_multi_token_list"] == []
    assert audit_ok["answer_multi_token_list"] == []
    assert audit_ok["policy"] == "strict"
    print("[fake/ok] strict: 10000 retained as expected.")

    class FakeBroken(FakeTokenizer):
        """Splits any 100-token into [99, 1] (mimics multi-token answers)."""
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            new_ids: List[int] = []
            for i in ids:
                if i == 100:
                    new_ids.extend([99, 1])
                else:
                    new_ids.append(i)
            return new_ids

    tok_broken = FakeBroken()
    audit_broken = audit_tokenizer_strict(tok_broken, "pythia-6.9b", "fake/broken")
    assert audit_broken["all_answers_single_token"] is False
    assert audit_broken["answer_multi_token_list"] == [100]
    assert audit_broken["operand_multi_token_list"] == []
    assert audit_broken["n_dropped"] > 0
    dropped_pairs_with_s_100 = sum(
        1 for d in audit_broken["dropped"] if d["s"] == 100
    )
    assert dropped_pairs_with_s_100 == audit_broken["n_dropped"]
    print(
        f"[fake/broken] strict: {audit_broken['n_retained']} retained, "
        f"{audit_broken['n_dropped']} dropped (all s=100 cases)."
    )

    inter = intersect_audits([audit_ok, audit_broken])
    assert len(inter) == audit_broken["n_retained"]
    sample = inter[0]
    assert "first_answer_token_id_per_model" in sample
    assert set(sample["first_answer_token_id_per_model"].keys()) == {
        "gpt-j-6b", "pythia-6.9b",
    }
    print(f"[intersect] {len(inter)} pairs in 2-way intersection.")

    audit_relaxed = audit_tokenizer_relaxed_llama(
        tok_broken, "llama-3.1-8b", "fake/broken",
    )
    assert audit_relaxed["policy"] == "relaxed_llama"
    assert audit_relaxed["n_retained"] == 10000
    assert audit_relaxed["n_dropped"] == 0
    multi_pairs = [
        p for p in audit_relaxed["retained"] if not p["ok_s"]
    ]
    assert len(multi_pairs) > 0
    print(
        f"[relaxed_llama] {audit_relaxed['n_retained']} retained "
        f"({len(multi_pairs)} with multi-token s)."
    )

    print(summary_report([audit_ok, audit_broken], len(inter), gate_threshold=7000))
    print("OK: tokenizer_audit self-test passed.")
