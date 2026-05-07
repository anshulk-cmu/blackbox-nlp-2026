# Babel setup — paths and activation

This document records the **exact filesystem layout** for running the
BlackBox-NLP 2026 pipeline on a CMU Babel GPU node. The split is:

- **Code** — lives in `~/BlackBox-NLP/` (this repo). Small.
- **Conda env** — lives at `/data/user_data/$USER/envs/blackbox/` on the data drive. Roughly 5–10 GB; would be too large for `$HOME`.
- **Conda package cache** — `/data/user_data/$USER/conda_pkgs/`. Redirected so package downloads don't fill `$HOME` either.
- **All run outputs** — live under `/data/user_data/$USER/blackbox/` (referred to as `$BLACKBOX_DATA` in the rest of the docs).

`$HOME` is a 100 GB NFS share shared with several other projects; the data drive is a 30 TB allocation that is the only place anything large should ever go.

---

## 1. Paths at a glance

| Purpose | Path | Notes |
|---|---|---|
| Repo (code only) | `/home/anshulk/BlackBox-NLP/` | Git working tree. |
| Conda env | `/data/user_data/anshulk/envs/blackbox/` | Created with `--prefix`. Activated by absolute path, not by name. |
| Conda package cache | `/data/user_data/anshulk/conda_pkgs/` | Set via `CONDA_PKGS_DIRS` at create time; no global `.condarc` change. |
| Run outputs (`$BLACKBOX_DATA`) | `/data/user_data/anshulk/blackbox/` | Subdirs below. |
| ↳ Tokenizer audit | `$BLACKBOX_DATA/tokenizer_audit/` | Phase 1 output. |
| ↳ Accuracy parquets | `$BLACKBOX_DATA/correctness/` | Phase 2. |
| ↳ Activation cache | `$BLACKBOX_DATA/activations/{model}/layer_{L}.npz` | Phase 3. |
| ↳ Manifold fits | `$BLACKBOX_DATA/manifolds/{model}/...` | Phase 4. |
| ↳ Test statistics | `$BLACKBOX_DATA/stats/{model}/...` | Phase 5. |
| ↳ Causal ACEs | `$BLACKBOX_DATA/aces/{model}.json` | Phase 6. |
| ↳ Paper figures | `$BLACKBOX_DATA/paper_figs/figure_{1..6}.pdf` | Phase 7. |
| ↳ Logs | `$BLACKBOX_DATA/logs/` | One file per phase × model × SLURM job id. |

---

## 2. One-time setup (already done)

These commands were run once on Babel and **do not need to be re-run** on this account:

```bash
# Create directories on the data drive
mkdir -p /data/user_data/$USER/envs \
         /data/user_data/$USER/conda_pkgs \
         /data/user_data/$USER/blackbox/{tokenizer_audit,correctness,activations,manifolds,stats,aces,paper_figs,logs}

# Build the conda env from the pinned spec, with the package cache on the data drive
cd ~/BlackBox-NLP
CONDA_PKGS_DIRS=/data/user_data/$USER/conda_pkgs \
    conda env create -f environment.yml \
    --prefix /data/user_data/$USER/envs/blackbox
```

The build log is at `$BLACKBOX_DATA/logs/conda_env_create.log` for reference.

---

## 3. Per-session activation

Run these every time you open a new shell (or put them in `~/.bashrc` if you want them permanent — see §5).

```bash
export BLACKBOX_DATA=/data/user_data/$USER/blackbox
conda activate /data/user_data/$USER/envs/blackbox
```

**Note: activate by absolute path**, not by name. Because the env was created with `--prefix`, it does not appear in `conda env list` under a short name; `conda activate blackbox` would not find it. The prompt will show `(blackbox)` once activated.

Verify:

```bash
which python              # → /data/user_data/anshulk/envs/blackbox/bin/python
python -m pip --version   # → pip ... from /data/user_data/anshulk/envs/blackbox/lib/python3.11/site-packages/pip
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo $BLACKBOX_DATA       # → /data/user_data/anshulk/blackbox
```

Last verified on `babel-s9-16` (RTX A6000): Python 3.11.15, numpy 1.26.4, torch 2.4.1, transformers 4.45.0, `torch.cuda.is_available()` returns `True`. Env occupies 7.4 GB; package cache occupies 4.2 GB.

---

## 4. Running the pipeline

After activation, all `code/` and `scripts/` entry points work as documented in [babel_execution_plan.md](babel_execution_plan.md). Quick references:

```bash
# Sanity check (no GPU; ~4 min on CPU)
python toy/run_toy.py                       # expect: 45 PASS / 0 FAIL

# Phase 1 — tokenizer audit (login node, CPU, ~5 min)
bash scripts/phase1_audit.sh

# Phase 2 — accuracy reproduction (one SLURM job per model)
sbatch scripts/phase2_accuracy.sbatch gpt-j-6b
sbatch scripts/phase2_accuracy.sbatch pythia-6.9b
sbatch scripts/phase2_accuracy.sbatch llama-3.1-8b
```

The phase scripts read `$BLACKBOX_DATA` if it's set; otherwise they fall back to the literal default `/data/user_data/$USER/blackbox`. Either way the outputs land in the right place.

---

## 5. Optional: make activation persistent

If you want activation to happen automatically on login, add to `~/.bashrc`:

```bash
export BLACKBOX_DATA=/data/user_data/$USER/blackbox
alias blackbox='conda activate /data/user_data/$USER/envs/blackbox'
```

Then `blackbox` activates the env in any new shell. Adding the `conda activate` line itself to `.bashrc` is **not** recommended because this account has several conda envs for other projects and you should pick one explicitly per session.

---

## 6. Things to watch

- **Always `python -m pip install …`, not bare `pip install`.** When you `conda activate` an env in a shell that already ran a different `pip` (e.g. system `/usr/bin/pip`), bash's command-name hash can keep resolving `pip` to the old path until you `hash -r`. `python -m pip` skips the hash entirely and always invokes the active env's pip, writing to `/data/user_data/$USER/envs/blackbox/lib/...`. The danger of bare `pip install` in a stale shell is that it silently falls back to `~/.local/lib/python.../site-packages/`, which is on the home quota.
- **`huggingface-cli login` writes to `~/.cache/huggingface/token`.** That's a few KB so it's fine, but the *model cache* (`~/.cache/huggingface/hub/`) can balloon to tens of GB. Redirect it before any model download:
  ```bash
  export HF_HOME=/data/user_data/$USER/hf_cache
  ```
  Add this to `~/.bashrc` next to `BLACKBOX_DATA` if you'll be downloading models repeatedly.
- **Don't symlink across the home/data boundary inside the repo.** Git tracking gets confused; the SLURM scripts that `git pull` would silently break. Keep the boundary at the directory level: code in repo, data via `$BLACKBOX_DATA`.
- **Quota.** Run `du -sh /data/user_data/$USER/blackbox/ /data/user_data/$USER/envs/blackbox/` periodically. Expected steady-state: env ~10 GB, blackbox/ ~5 GB after all phases run.
