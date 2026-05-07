"""
Accuracy reproduction utility (Phase 2 of babel_execution_plan.md).

For each (a, b) in the tokenizer-audit intersection, prompt the model with
the per-model template (KT 2025 Table 2), greedy-decode one next token, and
compare to the ground-truth answer token id. Save per-pair results to
parquet on Drive and report accuracy vs KT's published numbers.

This module exposes:
  - MODEL_CONFIG: per-model config (HF name, prompt, KT accuracy, dtype).
  - load_model(model_key, device): load the HF model in the configured dtype.
  - run_accuracy_check(model, tokenizer, pairs, model_key, ...) -> DataFrame.
  - save_correctness(df, path), load_correctness(path): parquet I/O.
  - summary_report(df, model_key): printable comparison vs KT.

The notebooks 02a/02b/02c wrap these functions; the same code is reusable
for re-runs and sensitivity analyses.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import json
import os

import torch
import pandas as pd


# Per-model config. Prompts match KT 2025 Table 2 (and full_paper_plan.md §3.1).
MODEL_CONFIG: Dict[str, Dict] = {
    "gpt-j-6b": {
        "hf_name":          "EleutherAI/gpt-j-6B",
        "prompt_template":  "Output ONLY a number. {a}+{b}=",
        "kt_accuracy_pct":  80.5,
        "dtype":            "bfloat16",   # A100 supports bf16
        "n_layers":         28,
    },
    "pythia-6.9b": {
        "hf_name":          "EleutherAI/pythia-6.9b",
        "prompt_template":  "Output ONLY a number. {a}+{b}=",
        "kt_accuracy_pct":  77.2,
        "dtype":            "bfloat16",
        "n_layers":         32,
    },
    "llama-3.1-8b": {
        "hf_name":          "meta-llama/Llama-3.1-8B",
        "prompt_template":  "The following is a correct addition problem.\n{a}+{b}=",
        "kt_accuracy_pct":  98.0,
        "dtype":            "bfloat16",
        "n_layers":         32,
    },
}


def load_model_and_tokenizer(model_key: str, device: str = "cuda"):
    """Load model + tokenizer per MODEL_CONFIG[model_key].

    Sets pad_token to eos_token and padding_side='left' so the next-token
    prediction is always at position -1 of each row in the batch (no need
    to mask-gather).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIG[model_key]

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = getattr(torch, cfg["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_name"], torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, tokenizer


def _build_prompts(pairs: List[Dict], template: str) -> List[str]:
    return [template.format(a=p["a"], b=p["b"]) for p in pairs]


def _expected_token_id(tokenizer, s: int) -> int:
    """The ground-truth answer token id (no leading space; follows '=').

    Asserts single-token; the tokenizer audit (Phase 1) already filtered.
    """
    ids = tokenizer.encode(str(s), add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Answer s={s} tokenizes to {len(ids)} tokens with this tokenizer. "
            f"Did you load the intersection.json from the same model's audit?"
        )
    return ids[0]


def run_accuracy_check(
    model,
    tokenizer,
    pairs: List[Dict],
    model_key: str,
    batch_size: int = 32,
    progress_every: int = 5,
    save_every: int = 200,
    partial_path: Optional[str] = None,
) -> pd.DataFrame:
    """Forward-pass each prompt once; greedy argmax of next-token logits.

    Saves a partial DataFrame to `partial_path` every `save_every` batches
    so a session drop loses at most that many pairs of work.

    Returns a DataFrame with columns:
      a, b, s, predicted_id, predicted_str, expected_id, correct.
    """
    cfg = MODEL_CONFIG[model_key]
    template = cfg["prompt_template"]

    # Resume from partial if it exists
    completed_keys = set()
    rows: List[Dict] = []
    if partial_path and os.path.exists(partial_path):
        cached = pd.read_parquet(partial_path)
        rows = cached.to_dict(orient="records")
        completed_keys = set((int(r["a"]), int(r["b"])) for r in rows)
        print(f"  resuming from partial: {len(rows)} pairs already done")

    # Filter remaining pairs
    remaining = [p for p in pairs if (int(p["a"]), int(p["b"])) not in completed_keys]
    if not remaining:
        print(f"  all {len(pairs)} pairs already in partial; nothing to do")
        return pd.DataFrame(rows)

    # Pre-compute the expected answer token id per unique s.
    unique_s = sorted({int(p["s"]) for p in remaining})
    expected_id_by_s = {s: _expected_token_id(tokenizer, s) for s in unique_s}

    n_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        batch = remaining[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        prompts = _build_prompts(batch, template)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        # left-padded -> next-token position is -1 of each row
        last_logits = out.logits[:, -1, :]                          # (B, V)
        predicted_ids = last_logits.argmax(dim=-1).cpu().tolist()    # length B

        for p, pred_id in zip(batch, predicted_ids):
            exp_id = expected_id_by_s[int(p["s"])]
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
                "expected_id":   int(exp_id),
                "correct":       bool(pred_id == exp_id),
            })

        if (batch_idx + 1) % progress_every == 0 or (batch_idx + 1) == n_batches:
            n_done = len(rows)
            n_correct = sum(1 for r in rows if r["correct"])
            print(f"    batch {batch_idx + 1}/{n_batches}: {n_done} pairs scored, "
                  f"running accuracy {100.0 * n_correct / max(n_done, 1):.2f}%")

        if partial_path and (batch_idx + 1) % save_every == 0:
            tmp_df = pd.DataFrame(rows)
            tmp_df.to_parquet(partial_path, index=False)

    df = pd.DataFrame(rows)
    if partial_path:
        df.to_parquet(partial_path, index=False)
    return df


def save_correctness(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def load_correctness(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def summary_report(df: pd.DataFrame, model_key: str) -> str:
    cfg = MODEL_CONFIG[model_key]
    n_total = len(df)
    n_correct = int(df["correct"].sum())
    pct = 100.0 * n_correct / max(n_total, 1)
    kt = cfg["kt_accuracy_pct"]
    delta_pp = pct - kt
    lines = [
        f"Model: {model_key} ({cfg['hf_name']})",
        f"  n_total:   {n_total}",
        f"  n_correct: {n_correct}",
        f"  accuracy:  {pct:.2f}%",
        f"  KT report: {kt:.1f}%",
        f"  delta:     {delta_pp:+.2f}pp",
    ]
    return "\n".join(lines)


def passes_sanity_gate(df: pd.DataFrame, model_key: str, tolerance_pp: float = 5.0) -> bool:
    """Pre-registered sanity gate from full_paper_plan.md §3.7 / babel_execution_plan.md §5.

    Pass iff |empirical accuracy - KT reported| <= tolerance_pp percentage points.
    """
    cfg = MODEL_CONFIG[model_key]
    pct = 100.0 * df["correct"].sum() / max(len(df), 1)
    return abs(pct - cfg["kt_accuracy_pct"]) <= tolerance_pp


def load_intersection_pairs(intersection_path: str) -> List[Dict]:
    """Load the Phase 1 intersection.json and return the list of pairs."""
    with open(intersection_path, "r") as f:
        data = json.load(f)
    return data["pairs"]


if __name__ == "__main__":
    # Local self-test that does NOT load any model (pure data-flow check).
    # Uses a fake tokenizer + fake model that always predicts the correct token.
    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "right"
        def encode(self, text, add_special_tokens=False):
            stripped = text.strip()
            try:
                return [int(stripped)]
            except ValueError:
                return [9999]
        def __call__(self, prompts, return_tensors=None, padding=False):
            # tokenize each prompt as a list of length 1 (we ignore content for the fake)
            B = len(prompts)
            ids = torch.zeros((B, 1), dtype=torch.long)
            attn = torch.ones((B, 1), dtype=torch.long)
            class Enc:
                pass
            e = Enc()
            e.input_ids = ids
            e.attention_mask = attn
            def to(self, device):
                return self
            e.to = lambda device: e
            # also mimic dict
            d = {"input_ids": ids, "attention_mask": attn}
            class D(dict):
                def to(self_, device):
                    return self_
            return D({"input_ids": ids, "attention_mask": attn})
        def decode(self, ids):
            return str(ids[0])

    class FakeModel:
        device = torch.device("cpu")
        def eval(self_):
            return self_
        def __call__(self_, input_ids=None, attention_mask=None, use_cache=False):
            # Predict s = a + b for each prompt using a side-channel: but our
            # FakeTokenizer doesn't carry that, so instead this fake "predicts"
            # token id 0 for every prompt. The accuracy check therefore should
            # produce a sensible (but wrong) DataFrame.
            B = input_ids.shape[0]
            V = 200
            logits = torch.zeros((B, 1, V))
            logits[:, 0, 0] = 100.0   # always predicts id 0
            class Out:
                pass
            o = Out()
            o.logits = logits
            return o

    pairs = [{"a": 3, "b": 4, "s": 7}, {"a": 1, "b": 2, "s": 3}]
    df = run_accuracy_check(FakeModel(), FakeTokenizer(), pairs, "gpt-j-6b",
                              batch_size=2, save_every=10000, progress_every=1)
    print(df)
    assert len(df) == 2
    # Fake predicts 0 for both; expected 7 and 3. So both wrong.
    assert df["correct"].sum() == 0
    print("OK: accuracy_check self-test passed (fake-model path).")
