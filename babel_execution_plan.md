# Babel Execution Plan

**Status.** Operations document for the BlackboxNLP 2026 submission.

**Companions.**
- [paper_math.md](paper_math.md) — authoritative math (7 theorems, 11 lemmas, proof outlines).
- [full_paper_plan.md](full_paper_plan.md) — authoritative methodology (pre-registration, 4-way data split, 4 ACEs).
- [toy/README.md](toy/README.md) — synthetic-toy validation (45 / 45 PASS).
- **This document** — authoritative *execution plan* on CMU Babel cluster, integrated with VS Code Remote-SSH.

When this document and any other doc disagree on operational details (paths,
GPU choices, SLURM scripts), this document wins. For math and methodology,
the other documents win.

---

## 0. Why Babel, not Colab

We tried Colab Pro+ first and ran into:
- Browser-only UI; no integration with VS Code where the rest of development happens.
- Session timeouts and intermittent secrets-fetch failures.
- Stateful kernels with module caching that hides "did I actually pull the new code?" issues.
- Limited debug surface (no breakpoints, no real REPL, no proper file diffs).

Babel gives us:
- VS Code Remote-SSH connection — full IDE, breakpoints, file tree, terminal.
- Persistent storage — activation caches and model weights survive across sessions.
- A100 80 GB (vs. Colab's 40 GB) — comfortable for Llama 3.1 8B with batch sizes that don't matter on Colab.
- SLURM batch submission — fire-and-forget for long phases.
- Reproducibility — pinned conda env, lock files, deterministic runs.

The trade-off: SLURM queue waits (typically minutes; can be hours during high
demand). For the lengths of our phases this is acceptable.

---

## 1. The execution graph

```
[Phase 0: Local prep]            laptop                ~30 min
       │
       ▼
[Phase 1: Tokenizer audit]       login node, CPU       ~5 min
       │  -> /data/user_data/$USER/blackbox/tokenizer_audit/{model}.json
       ▼
[Phase 2: Accuracy reproduction] A100 ×3 (separate)    ~3 h total
       │  -> /data/user_data/$USER/blackbox/correctness/{model}.parquet
       ▼
[Phase 3: Activation extraction] A100 ×3 (separate)    ~9 h total
       │  -> /data/user_data/$USER/blackbox/activations/{model}/layer_{L}.npz
       ▼
[Phase 4: Manifold + layer pick] CPU                   ~1 h
       │  -> /data/user_data/$USER/blackbox/manifolds/{model}/...
       ▼
[Phase 5: T_n statistics]        CPU                   ~30 min
       │  -> /data/user_data/$USER/blackbox/stats/{model}/...
       ▼
[Phase 6: Causal interventions]  A100 ×3 (separate)    ~9 h total
       │  -> /data/user_data/$USER/blackbox/aces/{model}.json
       ▼
[Phase 7: Aggregation + figures] CPU                   ~30 min
          -> /data/user_data/$USER/blackbox/paper_figs/figure_{2..6}.pdf
```

**Total wall time: ~24 h.** With SLURM queue waits, plan ~3-5 days end-to-end.

---

## 2. Storage paths

All paths assume `$USER = anshulk`. The repo defines `BLACKBOX_DATA` env var
defaulting to `/data/user_data/$USER/blackbox/`; export it differently if your
allocation lives elsewhere.

```
/data/user_data/$USER/blackbox/         # $BLACKBOX_DATA
├── tokenizer_audit/
│   ├── gpt-j-6b.json
│   ├── pythia-6.9b.json
│   ├── llama-3.1-8b.json
│   └── intersection.json
├── correctness/
│   ├── gpt-j-6b.parquet              # cols: a, b, s, predicted, correct
│   ├── pythia-6.9b.parquet
│   └── llama-3.1-8b.parquet
├── activations/
│   ├── gpt-j-6b/layer_{07,14,21,27}.npz
│   ├── pythia-6.9b/layer_{08,16,24,31}.npz
│   └── llama-3.1-8b/layer_{08,16,24,31}.npz
├── manifolds/
│   └── {model}/M_hat_{param,dm}.npz, selection.json, principal_angle_matrix.npz
├── stats/
│   └── {model}/T_n_{naive,cross,matched}.json, T_n_V.json, failure_modes.json
├── aces/
│   └── {model}.json
├── paper_figs/
│   └── figure_{1..6}.pdf
└── logs/
    └── {phase}_{model}_{slurm_job_id}.log
```

**Quota check.** Activations dominate: 3 models × 4 layers × ~80 MB ≈ 1 GB.
Total project footprint: ~5 GB. Confirm with `du -sh /data/user_data/$USER/`
during Phase 0.

**Repo location.** Clone the GitHub repo to `~/blackbox-nlp-2026/`:

```bash
cd ~ && git clone https://github.com/anshulk-cmu/blackbox-nlp-2026.git
cd ~/blackbox-nlp-2026
```

This is *separate from* `$BLACKBOX_DATA`. The repo holds code and docs;
the data dir holds outputs.

---

## 3. Phase 0: Local + Babel one-time setup

**Owner: Anshul. ~30 min one time.**

### 3.1 Connect VS Code to Babel via Remote-SSH

1. Install the **Remote - SSH** extension in VS Code (publisher: Microsoft).
2. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and run
   **Remote-SSH: Connect to Host...** → **Add New SSH Host...**
3. Enter `ssh anshulk@babel.lti.cs.cmu.edu` (or the actual login host). Save
   to your `~/.ssh/config`.
4. Connect. VS Code installs a server-side helper, then opens a new window
   running on Babel.
5. **File → Open Folder** → `/home/anshulk/blackbox-nlp-2026` (after the clone
   in §3.2 below).

From this point on, you edit code and run terminals *inside* VS Code, but
everything executes on Babel.

### 3.2 Clone the repo on Babel

In the VS Code Remote-SSH terminal:

```bash
cd ~
git clone https://github.com/anshulk-cmu/blackbox-nlp-2026.git
cd blackbox-nlp-2026
```

### 3.3 Set up the conda environment

```bash
module load anaconda3                  # or whatever Babel's conda module is named
conda env create -f environment.yml -n blackbox
conda activate blackbox
python toy/run_toy.py                  # 45 / 45 PASS in ~4 min on CPU
```

The repo includes [environment.yml](environment.yml) pinning every dependency.

### 3.4 HuggingFace login (one-time, on Babel)

```bash
huggingface-cli login
# Paste a fresh read-scope HF token. Token is stored in ~/.cache/huggingface/token
# with 600 perms. NEVER paste a token into a script, commit, or chat.
```

Confirm:

```bash
huggingface-cli whoami
```

### 3.5 Create the data directory

```bash
mkdir -p /data/user_data/$USER/blackbox/{tokenizer_audit,correctness,activations,manifolds,stats,aces,paper_figs,logs}
echo "export BLACKBOX_DATA=/data/user_data/$USER/blackbox" >> ~/.bashrc
source ~/.bashrc
```

### 3.6 Llama 3.1 license

Already accepted on the same HF account (per user). No further action.

### Pre-flight checklist (must all be ✓ before Phase 1)

- [ ] VS Code connected to Babel via Remote-SSH; can edit files and run terminals.
- [ ] Repo cloned at `~/blackbox-nlp-2026/`.
- [ ] Conda env `blackbox` exists; `python toy/run_toy.py` produces `45 PASS / 0 FAIL`.
- [ ] `huggingface-cli whoami` shows your HF username.
- [ ] `$BLACKBOX_DATA` is set; the directory exists.
- [ ] Llama 3.1 license accepted on the HF account.

---

## 4. Phase 1: Tokenizer audit

**Goal.** Determine which `(a, b)` pairs in `{0,...,99}²` produce single-token
operands AND single-token answer in each tokenizer; compute three-way
intersection.

**Runtime.** Login node, CPU. ~5 min. No SLURM job needed.

```bash
cd ~/blackbox-nlp-2026
conda activate blackbox
python code/run_phase1_audit.py
```

**Outputs:**
- `$BLACKBOX_DATA/tokenizer_audit/{model}.json` (one per model)
- `$BLACKBOX_DATA/tokenizer_audit/intersection.json`

**Pre-registered gate:** intersection ≥ 7000 pairs. Note: Llama 3.1's
tiktoken-style tokenizer splits multi-digit numbers per-digit; for Llama,
the audit relaxes to a "first answer token is single-token" check. Details
in [code/run_phase1_audit.py](code/run_phase1_audit.py) docstring.

---

## 5. Phase 2: Accuracy reproduction

**Goal.** Reproduce KT 2025's accuracies (80.5% / 77.2% / 98%) on the
intersection set. Confirms the prompt + tokenizer + decode pipeline is set
up correctly before activation extraction.

**Runtime.** A100 per model. ~1 h each on bf16 forward-only.

**Per-model SLURM jobs:**

```bash
sbatch scripts/phase2_accuracy.sbatch gpt-j-6b
sbatch scripts/phase2_accuracy.sbatch pythia-6.9b
sbatch scripts/phase2_accuracy.sbatch llama-3.1-8b
```

The single sbatch script takes the model key as `$1`. SLURM template in
[scripts/phase2_accuracy.sbatch](scripts/phase2_accuracy.sbatch).

**Outputs:** `$BLACKBOX_DATA/correctness/{model}.parquet`.

**Pre-registered gate:** `|empirical_accuracy - KT_reported| ≤ 5pp` per model.

For Llama specifically (multi-digit tokenization), the script uses
multi-token greedy decode (`generate(max_new_tokens=4)`) and string-
compares the decoded output to `str(s)`.

---

## 6. Phase 3: Activation extraction

**Goal.** Cache residual-stream activations at the `=` token at each
candidate analysis layer per model:

- GPT-J (28 layers): `{7, 14, 21, 27}`
- Pythia (32 layers): `{8, 16, 24, 31}`
- Llama (32 layers): `{8, 16, 24, 31}`

**Runtime.** A100 per model. ~3 h each.

**Per-model SLURM jobs:**

```bash
sbatch scripts/phase3_extract.sbatch gpt-j-6b
sbatch scripts/phase3_extract.sbatch pythia-6.9b
sbatch scripts/phase3_extract.sbatch llama-3.1-8b
```

The script registers HuggingFace forward hooks on each candidate layer's
block, runs forward passes in batches, and saves per-layer npz files.
Resumable: per-batch partials are written so a job preemption never costs
more than ~50 prompts of work.

**Memory math (Llama 3.1 8B, A100 80 GB).**
- Model bf16: 16 GB.
- Per-token activations bf16: ~16 KB per sample per hook.
- Batch 32 with 4 hooks: ~2 MB extra per forward.
- Comfortable margin within 80 GB even with much larger batch sizes.

**Outputs:** `$BLACKBOX_DATA/activations/{model}/layer_{L}.npz`
with cols `[h: (n, 4096), a, b, s, correct]`.

---

## 7. Phase 4: Manifold fitting and layer selection

**Goal.** Per [full_paper_plan.md §3.6](full_paper_plan.md): fit M̂ at every
candidate layer, compute held-out R², select the best layer, and produce
the primary M̂ per model. Diffusion Maps as fallback if R² < 0.90.

**Runtime.** ~1 h, CPU.

```bash
python code/run_phase4_manifolds.py
```

**Outputs:** `$BLACKBOX_DATA/manifolds/{model}/{M_hat_param.npz, M_hat_dm.npz, principal_angle_matrix.npz, selection.json}`.

---

## 8. Phase 5: T_n statistics

**Goal.** For each model at the chosen layer: three versions of T_n
(naive / cross-fitted / matched + cross-fitted), localization across
V_1..V_5, three-mode failure breakdown.

**Runtime.** ~30 min, CPU.

```bash
python code/run_phase5_stats.py
```

**Outputs:** `$BLACKBOX_DATA/stats/{model}/...`

Pre-registered positive-result criteria per [full_paper_plan.md §6.4](full_paper_plan.md).

---

## 9. Phase 6: Causal interventions

**Goal.** Run N / NV / S / R interventions and Proposition 9.3 prediction.

**Runtime.** A100 per model. ~3 h each.

```bash
sbatch scripts/phase6_causal.sbatch gpt-j-6b
sbatch scripts/phase6_causal.sbatch pythia-6.9b
sbatch scripts/phase6_causal.sbatch llama-3.1-8b
```

**Outputs:** `$BLACKBOX_DATA/aces/{model}.json`.

Pre-registered ACE criteria per [full_paper_plan.md §3.9.4](full_paper_plan.md).

---

## 10. Phase 7: Aggregation and figures

**Goal.** Build the six paper figures + master results table.

**Runtime.** ~30 min, CPU.

```bash
python code/run_phase7_figures.py
```

**Outputs:** `$BLACKBOX_DATA/paper_figs/figure_{1..6}.pdf` and `master_results_table.csv`.

---

## 11. SLURM script template

All Phase-2/3/6 SLURM scripts share the same template. Concrete files in
[scripts/](scripts/). Skeleton:

```bash
#!/bin/bash
#SBATCH --job-name=blackbox_{phase}_{model}
#SBATCH --partition=general                    # or whichever partition is right
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user_data/%u/blackbox/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/blackbox/logs/%x_%j.err

set -e
module load anaconda3
source activate blackbox

cd ~/blackbox-nlp-2026
git pull --ff-only

MODEL_KEY=$1
python code/run_phaseN_xxx.py --model $MODEL_KEY
```

Per-phase variants override `--gres`, `--time`, and the python script name.

---

## 12. Failure recovery

Every phase that runs > 5 minutes:

- **Drive flush every 50 batches.** Partial outputs go to
  `*_partial_{batch_idx}.npz`; final concat reads all partials.
- **Resume by checking output first.** Each script starts with
  `if Path(out).exists(): print('cached, skipping'); return` — re-running
  is idempotent.
- **SLURM preemption.** If a job is preempted, resubmit; the script picks
  up from the last cached partial.

---

## 13. VS Code workflow

The recommended development cycle:

1. **Edit code locally** (or via VS Code Remote-SSH file editor on Babel).
2. **Commit and push** from the local clone.
3. **In the Babel terminal (inside VS Code Remote)**: `git pull` and run
   the phase script.
4. **Browse outputs** via the VS Code Remote file tree at
   `/data/user_data/$USER/blackbox/`.
5. **For interactive Python**: open a `.py` file in VS Code, click
   "Run Python File in Terminal" — it executes on Babel with the conda env.

Breakpoint debugging works via VS Code's Python debugger over Remote-SSH.

---

## 14. Documentation completeness

### Math (paper_math.md): complete (45 / 45 toy PASS validates).

### Methodology (full_paper_plan.md): complete (Babel paths restored).

### Toy (toy/README.md): complete; runs locally or on Babel CPU node in ~4 min.

### Operations (this document): complete.

### Outstanding before Phase 1:
- [ ] Pre-flight checklist (§3.6) signed off.
- [ ] Babel allocation confirmed (run `groups` on Babel; check storage quota).

---

## 15. References to update in other docs

This document is the new authoritative operations reference. The Colab
references previously added to other docs have been reverted to point at
this document.

---

*End of Babel execution plan.*
