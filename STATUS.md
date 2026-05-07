# Project status — 2026-05-06

A snapshot of where the BlackBox-NLP 2026 project sits **right now** on this Babel account, after the env + data + verification setup work of 2026-05-06. This file is updated by hand at the end of each session; the source of truth for *math* is [paper_math.md](paper_math.md), for *methodology* is [full_paper_plan.md](full_paper_plan.md), for *operations* is [babel_execution_plan.md](babel_execution_plan.md), and for *paths* is [SETUP.md](SETUP.md). This document records the **state**, not the plan.

---

## 1. One-paragraph state

The repo is cloned onto Babel at `/home/anshulk/BlackBox-NLP/`. The pinned `blackbox` conda env exists at `/data/user_data/anshulk/envs/blackbox/` (7.4 GB), package cache at `/data/user_data/anshulk/conda_pkgs/` (4.2 GB), and the data root `$BLACKBOX_DATA = /data/user_data/anshulk/blackbox/` (124 KB so far — only contains build/run logs) has the 8-subdir layout from `babel_execution_plan.md §2`. The toy passes 45/45 in 326 s on the Babel CPU node at default `d_m`; CUDA is verified working on the attached A6000. **No real-model phase has been executed yet**: phases 1–7 are coded for phases 1–2 only, and even phases 1–2 have not been run on Babel against actual GPT-J / Pythia / Llama. Pre-flight items still outstanding: HF token install on this node, Llama 3.1 license confirmation on the HF account, VS Code Remote-SSH (optional). Everything else (env, data dirs, GPU verification, scripts on disk) is ready.

---

## 2. Filesystem state

All large artifacts live on the data drive; only the repo lives in `$HOME`.

| Path | Size | Purpose | Source of truth |
|---|---|---|---|
| `/home/anshulk/BlackBox-NLP/` | 2.6 MB | Git working tree (code + docs). | `git remote -v` → `git@github.com:anshulk-cmu/blackbox-nlp-2026.git`, branch `main`. |
| `/data/user_data/anshulk/envs/blackbox/` | 7.4 GB | Conda env (Python 3.11.15, torch 2.4.1+CUDA 12.1, transformers 4.45.0, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, pandas 2.2.2, matplotlib 3.9.2). | [environment.yml](environment.yml) pins. |
| `/data/user_data/anshulk/conda_pkgs/` | 4.2 GB | Conda package cache, redirected via `CONDA_PKGS_DIRS` at create time. | Set explicitly so package downloads don't fill `$HOME`. |
| `/data/user_data/anshulk/blackbox/` (= `$BLACKBOX_DATA`) | 124 KB | Run-output root. Subdirs created but mostly empty. | [babel_execution_plan.md §2](babel_execution_plan.md). |
| ↳ `tokenizer_audit/`, `correctness/`, `activations/`, `manifolds/`, `stats/`, `aces/`, `paper_figs/` | empty | Awaiting Phase 1–7 outputs. | Created 2026-05-06. |
| ↳ `logs/` | 6 files | `conda_env_create.log` (build trace) + `toy_run_<TS>.log` × 4 (default + --big runs) + the toy_run_<TS>.log from this session. | Logged via Python `logging` from `run_toy.py`. |

The home quota (100 GB shared) is at ~52 GB used and untouched by this project; the data quota (30 TB) is at ~566 GB used with ~12 GB attributable to this project (env + cache).

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
matmul_ms          = 436.4   (2048×2048 fp32 on cuda:0)
matmul_ok          = True
```

The smoke test runs unconditionally at the start of every `run_toy.py` invocation and is persisted to `results.json["gpu"]`. The toy itself is numpy / CPU; the smoke test is a sanity check that the CUDA stack the real-model phases will need is alive.

### 3.4 Toy validation runs

| Config | Wall time | Result | Log |
|---|---|---|---|
| default (`d_m_main=256, d_m_small=64, d_m_mid=128`) | 326 s | **45 / 45 PASS** | `$BLACKBOX_DATA/logs/toy_run_20260506_214323.log` |
| `--big` (`d_m_main=512, d_m_small=128, d_m_mid=256`) | 564 s | **42 / 45 PASS** | `$BLACKBOX_DATA/logs/toy_run_20260506_213304.log` |

The 3 FAILs at `--big` (E1.null, E3.parametric@2000, E13.V_inf_null) are tolerance-scaling artifacts from the chi-squared null variance `2 k σ⁴` growing with `k = d_m - dim(M)` while tolerances stay fixed. See [toy/README.md §10.8](toy/README.md) for the full diagnosis and the two ways to make them PASS at `d_m=512` (grow `n_pairs`, or `√k`-scale tolerances). The default-config 45/45 PASS confirms the math infrastructure is intact; the --big result confirms the linalg + I/O pipeline scales.

The default-config run reproduces every documented baseline number from [toy/README.md §9](toy/README.md) exactly (E1.null=+0.0119, E1.mean_shift=+0.2831, E1.variance_only=+1.2237, E1.concentration slope=-0.6159, E11.r=1.var=0.9656, …).

### 3.5 Numerical-output JSON

After each run, `toy/outputs/results.json` contains the 45-row pass/fail table plus `config`, `gpu`, `log_file`. Diffable across runs.

---

## 4. Pipeline phase status

The 7 real-model phases from [babel_execution_plan.md](babel_execution_plan.md). "Coded" means a runner script exists in `code/`; "Wired" means a wrapper / SLURM submitter exists in `scripts/`; "Run" means executed on Babel against actual model weights.

| Phase | Goal | Coded | Wired | Run on Babel |
|---|---|---|---|---|
| 0 | Local + Babel one-time setup | n/a | n/a | **partly done** — env, data dirs, GPU verified; HF login + VS Code Remote-SSH still outstanding (§5) |
| 1 | Tokenizer audit per model + 3-way intersection | ✅ `code/run_phase1_audit.py` + `code/tokenizer_audit.py` | ✅ `scripts/phase1_audit.sh` (CPU, login node) | ❌ not yet |
| 2 | Accuracy reproduction (KT 80.5 / 77.2 / 98) | ✅ `code/run_phase2_accuracy.py` + `code/accuracy_check.py` | ✅ `scripts/phase2_accuracy.sbatch` (one job per model, A6000 default with A100_80GB fallback noted in script) | ❌ not yet |
| 3 | Activation extraction at 4 layers per model | ❌ `run_phase3_extract.py` does not exist yet | ❌ `scripts/phase3_extract.sbatch` referenced in plan but not present | ❌ not yet |
| 4 | Manifold fits + layer pick (5 methods × 4 layers) | ❌ | ❌ | ❌ |
| 5 | T_n statistics (3 versions × V_1..V_5 × 3 modes) | ❌ | ❌ | ❌ |
| 6 | Causal interventions N / NV / S / R + Prop 9.3 | ❌ | ❌ `scripts/phase6_causal.sbatch` referenced | ❌ |
| 7 | Aggregation + 6 paper figures + master table | ❌ | ❌ | ❌ |

**Toy validation**: ✅ 45/45 PASS at default, 42/45 at --big (per §3.4).

---

## 5. Pre-flight checklist (vs `babel_execution_plan.md §3.6`)

| Item | Status | Notes |
|---|---|---|
| Repo cloned to `~/BlackBox-NLP/` (not the default `~/blackbox-nlp-2026/`) | ✅ done | Path differs from the plan; not load-bearing because all phase scripts use absolute repo-root resolution. |
| Conda env `blackbox` exists | ✅ done | Created by `--prefix` at `/data/user_data/anshulk/envs/blackbox/`, *not* under `~/miniconda3/envs/`. Activate by absolute path. |
| `python toy/run_toy.py` produces 45 PASS / 0 FAIL | ✅ verified 2026-05-06 | At default `d_m`. |
| `huggingface-cli whoami` succeeds | ❌ outstanding | No token installed on this account yet. Required for Phase 2 to download models. |
| `$BLACKBOX_DATA` set + dir exists | ✅ done | Set per session; permanent install via `~/.bashrc` is optional, see SETUP.md §5. |
| Llama 3.1 license accepted on HF account | 🔶 unknown | Per `babel_execution_plan.md §3.6`, "already accepted on the same HF account (per user)." Worth re-confirming before Phase 2. |
| VS Code Remote-SSH connected | ⚪ not required from a CLI session | `babel_execution_plan.md` recommends it for editing; not blocking for batch execution. |

**Two things to do before Phase 1 can even attempt to run on the real models:**

1. `huggingface-cli login` (paste a fresh read-scope HF token).
2. Re-confirm Llama 3.1 license acceptance on the HF account.

---

## 6. Documentation map (current)

| Doc | Length | What it covers |
|---|---|---|
| [README.md](README.md) | ~140 L | Top-level orientation, file index, repo structure, quick links. |
| [SETUP.md](SETUP.md) | ~80 L | Babel filesystem layout: env + data on `/data/user_data/`. Per-session activation. Gotchas (`pip` hash, `HF_HOME`, no symlinks across boundary). One-time setup commands (already executed). |
| [STATUS.md](STATUS.md) | this file | Where the project sits *right now* — env, data, what's verified, what's outstanding. |
| [paper_math.md](paper_math.md) | 1159 L | Mathematical proposal. 7 theorems, 11 lemmas. Authoritative for theorem statements + proof outlines. |
| [full_paper_plan.md](full_paper_plan.md) | 3212 L | Methodology. Pre-registration, 4-way data split, 4-intervention causal pipeline, 22-entry reviewer-defense matrix. |
| [babel_execution_plan.md](babel_execution_plan.md) | ~440 L | Operations. 7 phases on Babel via VS Code Remote-SSH. SLURM templates, resume logic. |
| [KT_paper.md](KT_paper.md) | 1818 L | Reading-script for the Kantamneni–Tegmark 2025 paper (talk prep). |
| [toy/README.md](toy/README.md) | ~1840 L | Synthetic toy walkthrough. Every E* experiment, what it tests, expected outputs, all 8 historical bug fixes (incl. §10.8 tolerance scaling). |
| [environment.yml](environment.yml) | 35 L | Pinned conda env spec. |

---

## 7. Recent git history

The last 9 commits on `main` (origin = `git@github.com:anshulk-cmu/blackbox-nlp-2026.git`):

```
7af830f SETUP.md: document Babel layout (env + data on /data/user_data)
2d7c67e scripts: default Phase 2 to A6000 to match Babel availability
c6069ca Reset: pivot from Colab back to Babel + VS Code Remote-SSH
bcad47f Fix Phase 1 audit: use prompt-context tokenization (caught Llama bug)
6f2163e Tokenizer Audit run completed. 06 May 2026, 07:43PM ET
9a84951 Phase 2: accuracy reproduction notebooks (GPT-J, Pythia, Llama)
b12b879 Phase 1: tokenizer audit notebook + utility + project README
9f440cc Pin Colab account + repo URL in execution plan
96ec82f Initial commit: math, plan, toy validation, Colab execution doc
```

Working-tree changes since `7af830f` (this session, not yet committed at the time of writing this file):

- `toy/run_toy.py` — added Python `logging`, CLI flags `--big / --d-m / --log-dir / --output-dir`, GPU smoke test on startup.
- `toy/README.md` — refreshed §3 (Babel run command + new flags), §10.7/10.8 (added tolerance-scaling note), §14.5 (CLI knobs replacing the obsolete monkey-patch instructions), "Last verified runs" snapshot.
- `SETUP.md` — added the `--big` example + cross-reference to §10.8.
- `code/tokenizer_audit.py`, `code/accuracy_check.py` — fixed stale `colab_execution_plan.md` references → `babel_execution_plan.md`.
- `STATUS.md` — this file (new).

---

## 8. Outstanding work — concrete next steps

In execution order:

1. **Install HF token on this Babel account** (`huggingface-cli login`). One-time. Required for Phase 1's tokenizer downloads and Phase 2's weight downloads.
2. **Re-confirm Llama 3.1 license** on the HF account. Required for Phase 2's Llama job.
3. **Optional but useful**: redirect the HF model cache off `$HOME` by exporting `HF_HOME=/data/user_data/$USER/hf_cache` (see [SETUP.md §6](SETUP.md)). Llama 3.1 8B weights alone are ~16 GB.
4. **Run Phase 1** end-to-end: `bash scripts/phase1_audit.sh`. Expected ~5 min on the login node. Pass criterion: 3-way intersection ≥ 7000 pairs.
5. **Submit Phase 2** as 3 separate SLURM jobs (one per model). Each ~1–2 h on A6000. Pass criterion per model: |empirical accuracy − KT-reported| ≤ 5pp.
6. **Code phases 3–7.** The `code/run_phase{3..7}_*.py` runners do not exist yet; full_paper_plan.md and babel_execution_plan.md specify what they should do but the actual Python is not written. This is the next big chunk of implementation work.

---

## 9. Reproducing this state from scratch

If a collaborator (or future-you) wants to rebuild this exact state on a fresh Babel account:

```bash
# Repo
cd ~ && git clone git@github.com:anshulk-cmu/blackbox-nlp-2026.git BlackBox-NLP
cd BlackBox-NLP

# Layout on data drive
mkdir -p /data/user_data/$USER/envs \
         /data/user_data/$USER/conda_pkgs \
         /data/user_data/$USER/blackbox/{tokenizer_audit,correctness,activations,manifolds,stats,aces,paper_figs,logs}

# Env
CONDA_PKGS_DIRS=/data/user_data/$USER/conda_pkgs \
    conda env create -f environment.yml \
    --prefix /data/user_data/$USER/envs/blackbox

# Verify
export BLACKBOX_DATA=/data/user_data/$USER/blackbox
conda activate /data/user_data/$USER/envs/blackbox
python toy/run_toy.py --log-dir "$BLACKBOX_DATA/logs"   # expect 45 / 45 PASS in ~5 min
```

That's the complete on-disk state captured by the table in §2 plus the verification numbers in §3.
