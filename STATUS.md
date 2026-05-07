# Project status — 2026-05-07

A snapshot of where the BlackBox-NLP 2026 project sits **right now** on this Babel account, after the Phase 1 close-out work of 2026-05-07. This file is updated by hand at the end of each session; the source of truth for *math* is [paper_math.md](paper_math.md), for *methodology* is [full_paper_plan.md](full_paper_plan.md), for *operations* is [babel_execution_plan.md](babel_execution_plan.md), and for *paths* is [SETUP.md](SETUP.md). This document records the **state**, not the plan.

---

## 1. One-paragraph state

The repo is cloned on Babel at `/home/anshulk/BlackBox-NLP/`. The pinned `blackbox` conda env exists at `/data/user_data/anshulk/envs/blackbox/` (7.4 GB), package cache at `/data/user_data/anshulk/conda_pkgs/` (4.2 GB), and the data root `$BLACKBOX_DATA = /data/user_data/anshulk/blackbox/` has the 9-subdir layout from `babel_execution_plan.md §2` plus `hf_cache/` (`HF_HOME`) and a new `dataset/` subdir for the canonical 10K-pair ground-truth artifact — 11 dirs in total. The toy passes **45/45 in 297.6 s** on `babel-s9-16` (RTX A6000). **Phase 0 and Phase 1 are both complete**: HF token from `.env` persisted at `$HF_HOME/token` (chmod 600), `huggingface-cli whoami` returns `anshul2048`, Llama 3.1 gate is open. **Phase 1 result: strict three-way intersection = 10000/10000** — KT's tokenization claims hold for all three models on `transformers 4.45.0` (GPT-J `GPT2TokenizerFast` 50257 vocab, Pythia `GPTNeoXTokenizerFast` 50254, Llama 3.1 `PreTrainedTokenizerFast` 128000); every integer in `[0, 198]` is single-token in every model's bare-string encoding; the `=` token is uniquely positioned in every sample prompt; the relaxed-Llama fallback was not triggered. The pre-registered gate `n_∩ ≥ 7000` ([babel_execution_plan.md §16:1414](babel_execution_plan.md)) PASS. **Phase 2 (accuracy reproduction) is unblocked**, with one known one-line schema edit needed at [code/run_phase2_accuracy.py:75](code/run_phase2_accuracy.py) to consume the new per-model `first_answer_token_id_per_model` field. **`$BLACKBOX_DATA` is exported per-session, not in `~/.bashrc`** (deliberate — other conda envs on this account would pick it up).

---

## 2. Filesystem state

All large artifacts live on the data drive; only the repo lives in `$HOME`.

| Path | Size | Purpose | Source of truth |
|---|---|---|---|
| `/home/anshulk/BlackBox-NLP/` | 2.6 MB | Git working tree (code + docs). | `git remote -v` → `git@github.com:anshulk-cmu/blackbox-nlp-2026.git`, branch `main`, clean. |
| `/home/anshulk/BlackBox-NLP/.env` | 1 KB, **gitignored** | Per-account secrets: `HF_TOKEN`, `HF_HOME`. Not committed. Template at [.env.example](.env.example). |
| `/data/user_data/anshulk/envs/blackbox/` | 7.4 GB | Conda env (Python 3.11.15, torch 2.4.1+CUDA 12.1, transformers 4.45.0, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, pandas 2.2.2, matplotlib 3.9.2). | [environment.yml](environment.yml) pins. |
| `/data/user_data/anshulk/conda_pkgs/` | 4.2 GB | Conda package cache, redirected via `CONDA_PKGS_DIRS` at create time. | Set explicitly so package downloads don't fill `$HOME`. |
| `/data/user_data/anshulk/blackbox/` (= `$BLACKBOX_DATA`) | ~10 MB | Run-output root. `dataset/`, `tokenizer_audit/`, and `logs/` populated by Phase 0 + Phase 1; remaining phase dirs empty. | [babel_execution_plan.md §2](babel_execution_plan.md). |
| ↳ `dataset/dataset_10k.json` | 1.8 MB | **NEW (Phase 1)** Canonical 10K-pair `(a, b, s)` ground truth + matched-permutation features (`carry_ones`, `carry_tens`, `a_decile`, `b_decile`, `sum_bin`). Single source of truth for downstream phases. | Built by `code/build_dataset.py`. Spec: `full_paper_plan.md §3.2:1587`. |
| ↳ `tokenizer_audit/{gpt-j-6b,pythia-6.9b,llama-3.1-8b}.json` | 1.9 MB each | Per-model strict audit: `n_retained=10000, n_dropped=0`, `policy=strict`, full per-pair records with `first_answer_token_id`. | Built by `code/run_phase1_audit.py`. |
| ↳ `tokenizer_audit/intersection.json` | 3.3 MB | Three-way intersection: `n_intersection=10000, gate_passed=True, llama_relaxed_fallback_used=False`; pairs carry `first_answer_token_id_per_model` for clean Phase 2 lookup. | Built by `code/run_phase1_audit.py`. |
| ↳ `tokenizer_audit/tokenization_test_{model}.json` | ~70 KB each | **NEW (Phase 1)** Per-integer tokenization table for `n ∈ [0, 198]`: bare/leading-space ids+decoded, single_token, round_trip, sample-prompt tokenizations with `=` position. Drives Appendix H. | Built by `code/test_tokenizers.py`. |
| ↳ `tokenizer_audit/tokenization_test_cross_model.json` | 154 B | Cross-model preview: `predicted_strict_intersection=10000`. | Built by `code/test_tokenizers.py`. |
| ↳ `correctness/`, `activations/`, `diagnostics/`, `manifolds/`, `stats/`, `aces/`, `paper_figs/` | empty | Awaiting Phase 2–7 outputs. | Created 2026-05-06 (+`diagnostics/`). |
| ↳ `logs/` | ~110 KB | `conda_env_create.log`, `toy_run_*.log`, plus three new Phase 1 logs (`phase1_dataset_<TS>.log`, `phase1_tokenization_test_<TS>.log`, `phase1_audit_<TS>.log`). | Python `logging` via `code/_logging.py`. |
| ↳ `hf_cache/` | 69 KB | `HF_HOME` redirect. Holds the read-scope token (`hf_cache/token`, 600 perms) and a Llama 3.1 `config.json` from the gate-check fetch. Future model-weight downloads land here. | Set in `.env`. |

The home quota (100 GB shared) is at ~52 GB used and untouched by this project; the data quota (30 TB) is at ~577 GB used with ~12 GB attributable to this project (env + cache).

---

## 3. Verification log — 2026-05-06

Everything below was performed on `babel-s9-16` (an interactive Babel node with one RTX A6000, capability 8.6).

### 3.1 Conda env build

```
Source : environment.yml (pinned)
Cmd    : CONDA_PKGS_DIRS=/data/user_data/anshulk/conda_pkgs \
            conda env create -f environment.yml \
            --prefix /data/user_data/anshulk/envs/blackbox
Result : exit 0
Trace  : $BLACKBOX_DATA/logs/conda_env_create.log
```

### 3.2 Env smoke test

After activation, the following imports succeeded:

```
numpy             1.26.4
scipy             1.13.1     (implicit via scikit-learn / sklearn)
scikit-learn      1.5.2
pandas            2.2.2
matplotlib        3.9.2
torch             2.4.1
torch.cuda.is_available()    → True
torch.cuda.get_device_name(0) → 'NVIDIA RTX A6000'
transformers      4.45.0
```

Env Python: `/data/user_data/anshulk/envs/blackbox/bin/python`. `which pip` reports `/usr/bin/pip` from a stale bash hash; `python -m pip` correctly resolves to `/data/user_data/anshulk/envs/blackbox/lib/python3.11/site-packages/pip`. Always use `python -m pip install …` rather than bare `pip install`. Documented in [SETUP.md §6](SETUP.md).

### 3.3 GPU smoke test (inside `run_toy.py`)

```
cuda_available     = True
device             = NVIDIA RTX A6000
torch_version      = 2.4.1
cuda_capability    = 8.6
matmul_ms          = ~436   (2048×2048 fp32 on cuda:0; ~310 ms on the 23:32 re-run)
matmul_ok          = True
```

The smoke test runs unconditionally at the start of every `run_toy.py` invocation and is persisted to `results.json["gpu"]`. The toy itself is numpy / CPU; the smoke test is a sanity check that the CUDA stack the real-model phases will need is alive.

### 3.4 Toy validation runs

| Config | Wall time | Result | Log |
|---|---|---|---|
| default (`d_m_main=256, d_m_small=64, d_m_mid=128`) | **297.6 s** (re-run 23:32) | **45 / 45 PASS** | `toy/outputs/toy_run_20260506_233246.log` |
| default (earlier) | 326 s | 45 / 45 PASS | `$BLACKBOX_DATA/logs/toy_run_20260506_214323.log` |
| `--big` (`d_m_main=512, d_m_small=128, d_m_mid=256`) | 564 s | **42 / 45 PASS** | `$BLACKBOX_DATA/logs/toy_run_20260506_213304.log` |

The 3 FAILs at `--big` (E1.null, E3.parametric@2000, E13.V_inf_null) are tolerance-scaling artifacts from the chi-squared null variance `2 k σ⁴` growing with `k = d_m - dim(M)` while tolerances stay fixed. See [toy/README.md §10.8](toy/README.md) for the full diagnosis and the two ways to make them PASS at `d_m=512` (grow `n_pairs`, or `√k`-scale tolerances). The default-config 45/45 PASS confirms the math infrastructure is intact; the --big result confirms the linalg + I/O pipeline scales.

The default-config run reproduces every documented baseline number from [toy/README.md §9](toy/README.md) exactly (E1.null=+0.0119, E1.mean_shift=+0.2831, E1.variance_only=+1.2237, E1.concentration slope=-0.6159, E11.r=1.var=0.9656, …).

### 3.5 HuggingFace authentication + Llama 3.1 gate

```
huggingface-cli whoami            → anshul2048
$HF_HOME                          = /data/user_data/anshulk/blackbox/hf_cache
$HF_HOME/token                    exists, perms = 600
HfApi().model_info(...Llama-3.1-8B) → OK, gated='manual', private=False
hf_hub_download(...,'config.json')  → OK
  config.architectures            = ['LlamaForCausalLM']
  config.hidden_size              = 4096
  config.num_hidden_layers        = 32
```

The token is loaded from `.env` (`set -a && source .env && set +a`) and was also persisted via `huggingface-cli login --token "$HF_TOKEN"` so future shells in this env work even without sourcing `.env`. `gated='manual'` is expected for Llama 3.1; the successful `config.json` fetch confirms this account has accepted the community license.

### 3.6 Numerical-output JSON

After each run, `toy/outputs/results.json` contains the 45-row pass/fail table plus `config`, `gpu`, `log_file`. Diffable across runs.

### 3.7 Phase 1 — tokenizer audit run (2026-05-07)

Comprehensive per-integer test + strict audit completed in well under the §4 5-min budget:

```
data path:       /data/user_data/anshulk/blackbox
dataset:         dataset/dataset_10k.json (10,000 rows, ordered pairs, ground truth s = a + b)
tokenizers:      transformers 4.45.0
  gpt-j-6b       GPT2TokenizerFast       vocab_size = 50,257
  pythia-6.9b    GPTNeoXTokenizerFast    vocab_size = 50,254
  llama-3.1-8b   PreTrainedTokenizerFast vocab_size = 128,000
per-integer test: every n ∈ [0, 198] is single-token in every model (operand 100/100; answer 199/199)
KT-claim check:  upper bounds hold — GPT-J 361, Pythia 557, Llama 999 all single-token
'=' position:    unique in every sample prompt (8 for GPT-J/Pythia, 11 for Llama)
strict audit:    all 3 models retain 10,000 / 10,000
intersection:    n_∩ = 10,000 (gate ≥ 7000 → PASS)
relaxed-Llama:   not triggered
cross-step:      dataset n_pairs == audit total; per-integer multi-token lists match (Step 7 ↔ Step 8)
elapsed wall:    ~3 sec (all CPU, on babel-s9-16)
```

The **three Appendix H numbers**: `n_gpt-j-6b = 10000`, `n_pythia-6.9b = 10000`, `n_llama-3.1-8b = 10000`, `n_∩ = 10000`. No multi-token cases excluded; the §1.2 "handful of multi-token cases we exclude" qualifier is empirically vacuous on the chosen `[0, 99]²` operand range.

---

## 4. Pipeline phase status

The 7 real-model phases from [babel_execution_plan.md](babel_execution_plan.md). "Coded" means a runner script exists in `code/`; "Wired" means a wrapper / SLURM submitter exists in `scripts/`; "Run" means executed on Babel against actual model weights.

| Phase | Goal | Coded | Wired | Run on Babel |
|---|---|---|---|---|
| 0 | Local + Babel one-time setup | n/a | n/a | ✅ **complete** (env, 9 data subdirs, GPU verified, HF token loaded, Llama 3.1 gate verified, VS Code Remote-SSH active, toy re-verified 45/45) |
| 1 | Tokenizer audit + 10K dataset + 3-way intersection | ✅ `code/build_dataset.py` + `code/test_tokenizers.py` + `code/tokenizer_audit.py` + `code/run_phase1_audit.py` (all use shared `code/_logging.py`) | ✅ direct `python ...` on the GPU node (no SLURM needed; `scripts/phase1_audit.sh` deleted as obsolete) | ✅ **complete** (n_∩ = 10000 / 10000; gate PASS; relaxed-Llama not triggered) — see §3.7 |
| 2 | Accuracy reproduction (KT 80.5 / 77.2 / 98) | ✅ `code/run_phase2_accuracy.py` + `code/accuracy_check.py` (needs 1-line edit at line 75 to consume `first_answer_token_id_per_model`) | ✅ `scripts/phase2_accuracy.sbatch` (one job per model, A6000 default with A100_80GB fallback noted in script) | ❌ not yet |
| 3 | Activation extraction at 4 layers per model | ❌ `run_phase3_extract.py` does not exist yet | ❌ `scripts/phase3_extract.sbatch` referenced in plan but not present | ❌ not yet |
| 4 | Manifold fits + layer pick (5 methods × 4 layers) | ❌ | ❌ | ❌ |
| 5 | T_n statistics (3 versions × V_1..V_5 × 3 modes) | ❌ | ❌ | ❌ |
| 6 | Causal interventions N / NV / S / R + Prop 9.3 | ❌ | ❌ `scripts/phase6_causal.sbatch` referenced | ❌ |
| 7 | Aggregation + 6 paper figures + master table | ❌ | ❌ | ❌ |

**Toy validation**: ✅ 45/45 PASS at default (re-verified this session), 42/45 at --big (per §3.4).

---

## 5. Pre-flight checklist (vs `babel_execution_plan.md §3.7`)

| Item | Status | Notes |
|---|---|---|
| VS Code Remote-SSH connected | ✅ done | This session is on `babel-s9-16` via Remote-SSH. |
| Repo cloned to `~/BlackBox-NLP/` (not the plan's default `~/blackbox-nlp-2026/`) | ✅ done | Path differs from the plan; SETUP.md is authoritative; phase scripts use absolute repo-root resolution. |
| Conda env `blackbox` exists | ✅ done | Created by `--prefix` at `/data/user_data/anshulk/envs/blackbox/`, *not* under `~/miniconda3/envs/`. Activate by absolute path. |
| `python toy/run_toy.py` produces 45 PASS / 0 FAIL | ✅ re-verified 2026-05-06 23:37 ET | At default `d_m`. 297.6 s wall. |
| `huggingface-cli whoami` succeeds | ✅ done | Returns `anshul2048`. Token from `.env`, also persisted at `$HF_HOME/token` (600 perms). |
| `$BLACKBOX_DATA` set + 9 plan-spec subdirs exist | ✅ done | **Per-session export**, not in `~/.bashrc` (other envs on this account). 9 plan subdirs incl. `diagnostics/`, plus `hf_cache/` and `dataset/` (11 total). |
| Llama 3.1 license accepted on HF account | ✅ verified | `model_info` returns `gated='manual'` and `config.json` download succeeds. |
| Quota check (`du -sh /data/user_data/$USER/`) | ✅ ~577 GB across all projects (~12 GB this project) | `blackbox/` itself ~10 MB after Phase 1; expected ~5 GB at full pipeline. |
| Phase 1 gate (`n_∩ ≥ 7000`, [babel §16:1414](babel_execution_plan.md)) | ✅ PASS | `n_∩ = 10000`; relaxed-Llama not triggered. See §3.7. |

**Phase 0 + Phase 1 are signed off.** Phase 2 (accuracy reproduction) is unblocked, modulo the 1-line edit at [code/run_phase2_accuracy.py:75](code/run_phase2_accuracy.py).

---

## 6. Documentation map (current)

| Doc | Length | What it covers |
|---|---|---|
| [README.md](README.md) | ~140 L | Top-level orientation, file index, repo structure, quick links. |
| [SETUP.md](SETUP.md) | ~80 L | Babel filesystem layout: env + data on `/data/user_data/`. Per-session activation. Gotchas (`pip` hash, `HF_HOME`, no symlinks across boundary). One-time setup commands. |
| [STATUS.md](STATUS.md) | this file | Where the project sits *right now* — env, data, what's verified, what's outstanding. |
| [.env.example](.env.example) | ~50 L | Template for `.env` (HF_TOKEN, HF_HOME, BLACKBOX_DATA, optional WANDB / Anthropic / OpenAI keys). Copy to `.env` and fill in. |
| [paper_math.md](paper_math.md) | 1159 L | Mathematical proposal. 7 theorems, 11 lemmas. Authoritative for theorem statements + proof outlines. |
| [full_paper_plan.md](full_paper_plan.md) | 3212 L | Methodology. Pre-registration, 4-way data split, 4-intervention causal pipeline, 22-entry reviewer-defense matrix. |
| [babel_execution_plan.md](babel_execution_plan.md) | ~1600 L | Operations. 7 phases on Babel via VS Code Remote-SSH. SLURM templates, resume logic. (Rewritten v2 in commit f5f8b89.) |
| [KT_paper.md](KT_paper.md) | 1818 L | Reading-script for the Kantamneni–Tegmark 2025 paper (talk prep). |
| [toy/README.md](toy/README.md) | ~1840 L | Synthetic toy walkthrough. Every E* experiment, what it tests, expected outputs, all 8 historical bug fixes (incl. §10.8 tolerance scaling). |
| [environment.yml](environment.yml) | 35 L | Pinned conda env spec. |

---

## 7. Recent git history

The last 8 commits on `main` (origin = `git@github.com:anshulk-cmu/blackbox-nlp-2026.git`):

```
f5f8b89 docs: align plan with paper_math.md, rewrite Babel plan v2, add .env scaffolding
791fbeb toy: add logging + GPU smoke test, document Babel d_m scaling
7af830f SETUP.md: document Babel layout (env + data on /data/user_data)
2d7c67e scripts: default Phase 2 to A6000 to match Babel availability
c6069ca Reset: pivot from Colab back to Babel + VS Code Remote-SSH
bcad47f Fix Phase 1 audit: use prompt-context tokenization (caught Llama bug)
6f2163e Tokenizer Audit run completed. 06 May 2026, 07:43PM ET
9a84951 Phase 2: accuracy reproduction notebooks (GPT-J, Pythia, Llama)
```

Working-tree changes since `f5f8b89` (this Phase 1 session, not yet committed):

- **Created:** `code/_logging.py`, `code/build_dataset.py`, `code/test_tokenizers.py`.
- **Rewritten from scratch:** `code/tokenizer_audit.py` (strict-primary + relaxed-Llama; full self-test), `code/run_phase1_audit.py` (auto-fallback driver with logging).
- **Deleted:** `scripts/phase1_audit.sh` (obsolete bash wrapper, broken on prefix-activated env).
- **Updated:** `STATUS.md` (this file).

---

## 8. Outstanding work — concrete next steps

Phase 0 + Phase 1 are complete; next is Phase 2 + implementation of phases 3–7:

1. **Phase 2 schema edit (1 line).** [code/run_phase2_accuracy.py:75](code/run_phase2_accuracy.py) reads `p["first_answer_token_id"]` flat; switch to `p["first_answer_token_id_per_model"][args.model]` to consume Phase 1's per-model intersection schema. Verify with the FakeTokenizer self-tests in `code/accuracy_check.py`.
2. **Submit Phase 2** as 3 separate SLURM jobs (one per model). Each ~1–2 h on A6000. Pass criterion per model: |empirical accuracy − KT-reported| ≤ 5pp ([babel §16:1417-1420](babel_execution_plan.md)).
3. **Code phases 3–7.** The `code/run_phase{3..7}_*.py` runners do not exist yet; `full_paper_plan.md` and `babel_execution_plan.md` specify what they should do but the actual Python is not written. This is the next big chunk of implementation work.
4. **Ongoing**: keep STATUS.md updated at the end of each session.

---

## 9. Reproducing this state from scratch

If a collaborator (or future-you) wants to rebuild this exact state on a fresh Babel account:

```bash
# Repo
cd ~ && git clone git@github.com:anshulk-cmu/blackbox-nlp-2026.git BlackBox-NLP
cd BlackBox-NLP

# Layout on data drive (10 plan-subdirs + dataset/ + hf_cache/)
mkdir -p /data/user_data/$USER/envs \
         /data/user_data/$USER/conda_pkgs \
         /data/user_data/$USER/blackbox/{tokenizer_audit,correctness,activations,diagnostics,manifolds,stats,aces,paper_figs,logs,hf_cache,dataset}

# Env
CONDA_PKGS_DIRS=/data/user_data/$USER/conda_pkgs \
    conda env create -f environment.yml \
    --prefix /data/user_data/$USER/envs/blackbox

# Per-session env (do NOT add to ~/.bashrc — other envs on this account)
export BLACKBOX_DATA=/data/user_data/$USER/blackbox
conda activate /data/user_data/$USER/envs/blackbox

# Secrets (.env gitignored)
cp .env.example .env
$EDITOR .env    # set HF_TOKEN, HF_HOME=/data/user_data/$USER/blackbox/hf_cache
set -a && source .env && set +a

# HF login (persists to $HF_HOME/token)
huggingface-cli login --token "$HF_TOKEN"
chmod 600 "$HF_HOME/token"

# Verify the Llama 3.1 gate
python -c "from huggingface_hub import hf_hub_download; \
           hf_hub_download('meta-llama/Meta-Llama-3.1-8B', 'config.json')"

# Toy validation
python toy/run_toy.py --log-dir "$BLACKBOX_DATA/logs"   # expect 45 / 45 PASS in ~5 min

# Phase 1 — run directly on a GPU node (no .sh, no SLURM)
python code/build_dataset.py --data-dir "$BLACKBOX_DATA"          # 10K-pair dataset
python code/test_tokenizers.py --data-dir "$BLACKBOX_DATA"        # comprehensive per-integer test (3 models)
python code/run_phase1_audit.py --data-dir "$BLACKBOX_DATA"       # strict audit + intersection (gate ≥ 7000)
```

That's the complete on-disk state captured by the table in §2 plus the verification numbers in §3.
