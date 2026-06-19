"""Step 0 gate: operands (0-99), answers (0-198), and the '=' last token must each be a single token."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "common"))
import registry as reg  # noqa: E402

_PAIR = re.compile(r"(\d+)\+(\d+)=")


def _overlaps(span, tok_span):
    s, e = span
    ts, te = tok_span
    return not (te <= s or ts >= e)


def _load_rows(p):
    with open(p, "r", encoding="utf-8") as f:
        return [(int(r["id"]), int(r["a"]), int(r["b"]), int(r["sum"])) for r in csv.DictReader(f)]


def check_answers(tok, max_sum):
    bad = [{"value": s, "n_tokens": len(tok.encode(str(s), add_special_tokens=False))}
           for s in range(max_sum + 1)
           if len(tok.encode(str(s), add_special_tokens=False)) != 1]
    return {"checked": max_sum + 1, "n_bad": len(bad), "bad": bad}


def check_prompts(tok, template, rows):
    bad_a, bad_b, bad_eq = {}, {}, []
    for (pid, a, b, s) in rows:
        prompt = template.format(a=a, b=b)
        m = _PAIR.search(prompt)
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offs, ids = enc["offset_mapping"], enc["input_ids"]
        if sum(_overlaps(m.span(1), o) for o in offs) != 1:
            bad_a.setdefault(a, prompt)
        if sum(_overlaps(m.span(2), o) for o in offs) != 1:
            bad_b.setdefault(b, prompt)
        if tok.decode([ids[-1]]).strip() != "=" and len(bad_eq) < 20:
            bad_eq.append({"id": pid, "prompt": prompt, "last_token": tok.decode([ids[-1]])})
    return {"n_prompts": len(rows), "bad_operand_a_values": sorted(bad_a),
            "bad_operand_b_values": sorted(bad_b), "n_bad_eq": len(bad_eq), "bad_eq_examples": bad_eq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "ground_truth" / "addition_2digit.csv"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "tokenizer_check"))
    args = ap.parse_args()

    registry = reg.load_registry()
    model_key = args.model or registry["primary"]
    cfg = reg.get_model_cfg(model_key, registry)
    template = cfg["prompt_template"]
    tokenizer = reg.load_tokenizer(model_key, registry)
    assert getattr(tokenizer, "is_fast", False), "fast tokenizer required (offset mapping)"
    rows = _load_rows(Path(args.data))

    ans = check_answers(tokenizer, max(r[3] for r in rows))
    pr = check_prompts(tokenizer, template, rows)
    gate_pass = (ans["n_bad"] == 0 and not pr["bad_operand_a_values"]
                 and not pr["bad_operand_b_values"] and pr["n_bad_eq"] == 0)

    report = {"model_key": model_key, "hf_name": cfg["hf_name"], "prompt_template": template,
              "answers": ans, "prompts": pr, "gate_pass": gate_pass}
    out_dir = Path(args.out_dir) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    offenders = ([{"kind": "answer", "value": b["value"], "detail": b["n_tokens"]} for b in ans["bad"]]
                 + [{"kind": "operand_a", "value": v, "detail": ">1"} for v in pr["bad_operand_a_values"]]
                 + [{"kind": "operand_b", "value": v, "detail": ">1"} for v in pr["bad_operand_b_values"]]
                 + [{"kind": "eq_not_last", "value": e["id"], "detail": e["last_token"]} for e in pr["bad_eq_examples"]])
    with open(out_dir / "offenders.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "value", "detail"])
        w.writeheader()
        w.writerows(offenders)

    print(f"{model_key}: answers_ok={ans['n_bad']==0} a_bad={pr['bad_operand_a_values']} "
          f"b_bad={pr['bad_operand_b_values']} eq_bad={pr['n_bad_eq']}")
    print("GATE: PASS" if gate_pass else "GATE: FAIL (offenders.csv) — do not extract")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
