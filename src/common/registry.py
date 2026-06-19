"""Model registry + loaders (reads configs/models.yaml). CUDA bf16/fp16 only."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"


def load_registry(path=CONFIG_PATH):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_cfg(model_key, registry=None):
    reg = registry or load_registry()
    if model_key not in reg["models"]:
        raise ValueError(f"Unknown model {model_key!r}. Known: {sorted(reg['models'])}")
    return reg["models"][model_key]


def primary_model(registry=None):
    return (registry or load_registry())["primary"]


def load_tokenizer(model_key, registry=None):
    from transformers import AutoTokenizer
    cfg = get_model_cfg(model_key, registry)
    tok = AutoTokenizer.from_pretrained(cfg["hf_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # position -1 = '='
    return tok


def load_model_and_tokenizer(model_key, registry=None):
    import torch
    from transformers import AutoModelForCausalLM
    assert torch.cuda.is_available(), "CUDA required (bf16/fp16 GPU only)"
    cfg = get_model_cfg(model_key, registry)
    tok = load_tokenizer(model_key, registry)
    dtype = getattr(torch, cfg["dtype"])  # bfloat16 | float16
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_name"], torch_dtype=dtype).to("cuda")
    model.eval()
    return tok, model, model.config.hidden_size


def get_block_modules(model, block_path):
    obj = model
    for attr in block_path.split("."):
        obj = getattr(obj, attr)
    return obj
