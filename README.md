# Off the Number Manifold: Geometric Signatures of Arithmetic Failures in Language Models

*A localized geometric diagnostic for mechanistic failure modes in LLM arithmetic — building on Kantamneni & Tegmark 2025 (the helix and the Clock algorithm) and asking the question they did not: when models get addition **wrong**, where does the geometry break?*

**Target venue.** BlackboxNLP 2026 archival track (co-located with EMNLP 2026, Budapest). Direct submission deadline **July 17, 2026 (AoE)**.

**Authors.** Anshul Kumar, Barnábás Póczos.

**Status.** Methodology locked, math validated (45/45 toy PASS), Babel execution plan finalized (v2, 2026-05-06). Ready for real-model run on **CMU Babel** via VS Code Remote-SSH.

**Working title note.** The earlier theorem-heavy title — *Off-Manifold Failure: A Geometric Test for Localized Computational Errors in Language Models* — led with the infrastructure rather than the finding. The new title leads with the empirical question (arithmetic failures) and the geometric framing (off the number manifold), per [full_paper_plan.md §12.7](full_paper_plan.md). The technical title is preserved internally for cross-references in the math file.

---

## What this repo contains

| File | Purpose |
|---|---|
| [paper_math.md](paper_math.md) | Mathematical proposal (7 theorems, 11 lemmas, 4 propositions/corollaries). Authoritative source for theorem statements and proof outlines. |
| [full_paper_plan.md](full_paper_plan.md) | End-to-end methodology: pre-registration, 4-way data split, four ACE interventions, layer selection, five-method comparison, reviewer-defense matrix. |
| [babel_execution_plan.md](babel_execution_plan.md) | Step-by-step Babel execution plan (v2, 1600+ lines): 8 phases (0 setup → 7 figures), per-phase math/plan/KT cross-references, REG diagnostic gate, 5-method manifold comparison, 3-version T_n, 4 ACE causal pipeline, pre-registered acceptance criteria, 9 known issues with fixes, full glossary. |
| [SETUP.md](SETUP.md) | Babel filesystem layout: where the conda env, package cache, and run outputs live on the data drive (so nothing large lands in `$HOME`); per-session activation snippet. |
| [STATUS.md](STATUS.md) | Project state snapshot — what's installed, verified, and outstanding on this Babel account right now. Updated by hand at the end of each session. |
| [environment.yml](environment.yml) | Pinned conda environment (`conda env create -f environment.yml`). |
| [toy/](toy/) | Synthetic-toy validation suite. **45 / 45 PASS in 245s on CPU.** Math verified before any GPU spend. |
| [code/](code/) | Importable utilities (`tokenizer_audit.py`, `accuracy_check.py`) + per-phase runners (`run_phase{1..7}_*.py`). |
| [scripts/](scripts/) | Shell wrappers and SLURM `.sbatch` files for each phase. |
| [KT_paper.md](KT_paper.md) | Reference: Kantamneni & Tegmark 2025, *Language models use trigonometry to do addition*. arXiv:2502.00873. |

---

## One-paragraph contribution

Kantamneni and Tegmark 2025 showed GPT-J 6B, Pythia 6.9B, and Llama 3.1 8B encode integers on a generalized helix (one linear axis plus four cosine/sine pairs at periods T ∈ {2, 5, 10, 100}) and use a Clock-style algorithm for two-digit addition — but they study only *correct* cases and explicitly leave the Llama 3.1 8B fit gap (their Figure 23) and the algorithm-composition step (their §5.5) as open questions. This paper introduces a localized geometric diagnostic for the **wrong** cases that closes both gaps. Given a candidate manifold recovered from correct activations and a hypothesized failure subspace V, the test classifies wrong-population activations into three structurally distinct failure modes — *on-curve* (position errors), *off-curve in-span* (drift along the span), or *off-span* (geometric structure broken). The diagnostic comes with a finite-sample concentration bound (Theorem 4.2) with anisotropic Hanson–Wright generalization (Theorem 4.10), an achievability rate `n ≳ rσ⁴/Δ⁴` for localization power (Theorem 5.2(a)) plus a proven two-point Le Cam lower bound at `σ²/Δ²` (the matching minimax rate is conjectural), a finite-sample manifold-recovery bound for the parametric estimator with operator-norm Wedin (Theorem 6.2) and explicit misspecification accounting (Theorem 6.5), a Neyman-orthogonal cross-fitting protocol with explicit Gateaux-derivative verification (Theorem 7.2), exact conditional null calibration via Lehmann–Romano matched permutation under the pooled-data exchangeability hypothesis `h ⊥ y | φ(a, b)` (Theorem 8.3), and a four-intervention causal pipeline with Hessian-bounded magnitude prediction under the Alignment Assumption ALN (Proposition 9.3). On Llama 3.1 8B — where KT report the weakest helix fit — the diagnostic is the headline empirical case: we predict an "off-span" failure mode driven by Llama's gated MLPs implementing computations outside the helix subspace.

---

## How to read this repo

**For a reviewer or collaborator (~2 hours):**

1. **Skim [paper_math.md](paper_math.md)** for the math (~30 min). Sections 1–3 set up the assumptions; sections 4–9 are the theorem statements and proof outlines.
2. **Skim [full_paper_plan.md](full_paper_plan.md)** for methodology motivation (~45 min). Section 2 maps to paper_math.md theorems; section 3 is the pipeline; section 8 is the reviewer-defense matrix.
3. **Read [toy/README.md](toy/README.md)** end-to-end (~30 min) to see how the math was validated on synthetic ground truth.
4. **Read [babel_execution_plan.md](babel_execution_plan.md)** to see how the real-model run is organized (~15 min).

**For the executor (Anshul, when starting Babel work):**

0. Open a shell on the Babel node and activate the env per [SETUP.md §3](SETUP.md):
   ```bash
   export BLACKBOX_DATA=/data/user_data/$USER/blackbox
   conda activate /data/user_data/$USER/envs/blackbox
   ```
   The env and `$BLACKBOX_DATA` directory tree are already built on this account; SETUP.md documents the paths and the one-time commands that produced them.
1. Verify the rest of the [babel_execution_plan.md §19 pre-flight checklist](babel_execution_plan.md) (VS Code Remote-SSH connected, HF login, Llama 3.1 license, all model weights pre-stageable).
2. Run `python toy/run_toy.py` once. Confirm `45 / 45 PASS`. Do not proceed without this.
3. Run **Phase 1** (tokenizer audit, ~5 min CPU): `bash scripts/phase1_audit.sh`. Gate: three-way intersection ≥ 7000 pairs.
4. Submit **Phase 2** (accuracy reproduction, ~3h GPU) for all three models in parallel: `./scripts/submit_all_models.sh 2`. Gate: per-model accuracy within 5pp of KT.
5. Submit **Phase 3** (activation extraction, ~9h GPU total): `./scripts/submit_all_models.sh 3`. Caches all 4 candidate layers per model.
6. Run **Phase 4a** (REG diagnostics, ~30 min CPU): `python code/run_phase4a_diagnostics.py`. Gate: σ̂_eff·κ̂_max ≤ 0.1 AND η̂ ≤ 0.1.
7. Run **Phase 4b** (manifold fit + layer selection, ~30 min CPU): `python code/run_phase4b_manifolds.py`. Gate: held-out R² ≥ 0.9 OR Diffusion Maps fallback.
8. Run **Phase 5a–5b** (test statistics + localization, ~1h CPU total): `python code/run_phase5a_stats.py && python code/run_phase5b_localization.py`.
9. Submit **Phase 6** (causal interventions, ~9h GPU total): `./scripts/submit_all_models.sh 6`. Gate: 4 ACE criteria pass on ≥ 2 of 3 models.
10. Run **Phase 7** (aggregation + figures, ~30 min CPU): `python code/run_phase7_figures.py`. Outputs the 6 paper figures + master results table.

Total elapsed: ~3–5 days including SLURM queue waits; ~24 GPU-hours + ~3 CPU-hours of actual compute. Every phase is resumable from per-batch partials.

---

## Quick links by question

- **"What's the math?"** → [paper_math.md](paper_math.md), Theorems 4.2 / 5.2 / 6.2 / 6.6 / 7.2 / 8.3 / 9.3.
- **"What's the methodology?"** → [full_paper_plan.md §2–§3](full_paper_plan.md).
- **"How was the math validated?"** → [toy/README.md](toy/README.md) (45 PASS / 0 FAIL).
- **"How will the real run be executed?"** → [babel_execution_plan.md](babel_execution_plan.md).
- **"What's pre-registered before any data is touched?"** → [full_paper_plan.md §6](full_paper_plan.md).
- **"What's the four-intervention causal pipeline?"** → [full_paper_plan.md §3.9](full_paper_plan.md), Proposition 9.3.
- **"What if the experiment doesn't fire?"** → [full_paper_plan.md §7 fallback structure](full_paper_plan.md). Every interpretation step has a publishable null version.
- **"Where does my conda env / HF cache / activations actually live on Babel?"** → [SETUP.md](SETUP.md). All large artifacts on `/data/user_data/$USER/`; only the repo lives in `$HOME`.

---

## Running the synthetic toy locally

The toy is the most useful sanity check; it runs on CPU and finishes in ~4 minutes:

```bash
# In a Python env with numpy, scipy, sklearn, matplotlib, torch (CPU is fine):
python toy/run_toy.py
```

Expected output: `45 PASS / 0 FAIL (45 total)`. If anything fails, do not proceed to Babel phases — investigate the toy first.

---

## Repository structure

```
.
├── README.md                        ← you are here
├── SETUP.md                         ← Babel filesystem layout (env + data on /data/user_data)
├── STATUS.md                        ← project-state snapshot (what's installed / verified / outstanding)
├── KT_paper.md                      ← reference paper
├── paper_math.md                    ← math
├── full_paper_plan.md               ← methodology
├── babel_execution_plan.md          ← operations (Babel + VS Code Remote-SSH)
├── environment.yml                  ← conda env pinning
├── .env.example                     ← secrets template (committed; copy to .env locally)
├── .env                             ← your local secrets (gitignored, never committed)
├── code/
│   ├── tokenizer_audit.py           ← Phase 1 utility
│   ├── accuracy_check.py            ← Phase 2 utility
│   ├── run_phase1_audit.py          ← Phase 1 runner
│   └── run_phase2_accuracy.py       ← Phase 2 runner (--model arg)
├── scripts/
│   ├── phase1_audit.sh              ← Phase 1 wrapper (CPU, login node)
│   └── phase2_accuracy.sbatch       ← Phase 2 SLURM job (A100; takes model key)
└── toy/
    ├── README.md                    ← toy walkthrough (45 / 45 PASS)
    ├── run_toy.py                   ← top-level runner (E1–E16)
    ├── synth_world.py               ← synthetic world construction
    ├── manifold_methods.py          ← five M̂ recovery methods
    ├── tests.py                     ← T_n / T_n^V / cross-fit / matched-permutation
    ├── causal.py                    ← four-intervention pipeline + smooth-max LD_τ
    ├── theorem_predictions.py       ← analytic predictions from paper_math.md
    └── outputs/                     ← regenerable; not committed
```

---

## Compute environment

- **Real-model runs.** **CMU Babel** cluster via VS Code Remote-SSH. A100 80 GB for activation extraction (Phase 3) and causal interventions (Phase 6); CPU nodes for tokenizer audit (Phase 1), REG diagnostics (Phase 4a), manifold fitting (Phase 4b), test statistics + localization (Phase 5a/5b), and figures (Phase 7). Phase 2 (accuracy reproduction) runs on A100 in parallel with Phase 3. Budget: ~24 GPU-hours total across the 8 phases (3 of them GPU-heavy, 5 of them CPU).
- **Toy runs.** Any Python 3.10+ environment with `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `torch`, `pandas`. CPU is sufficient. ~4 minutes wall clock.
- **Persistent storage.** Babel scratch at `/data/user_data/$USER/blackbox/` (set as `$BLACKBOX_DATA`). Conda env at `/data/user_data/$USER/envs/blackbox/`. See [SETUP.md](SETUP.md) for the full path table and the reasons (`$HOME` quota, no conda env in home).

---

## Security notes and secrets management

**Two ways to provide secrets to the pipeline:**

1. *Preferred for interactive Babel work:* `huggingface-cli login` once per
   account. The token lives in `~/.cache/huggingface/token` with `600` perms
   and is read automatically by `transformers`.
2. *Preferred for scripted runs and local dev:* a project-local `.env` file.
   The repo ships with [`.env.example`](.env.example) (committed template);
   copy it once and fill in your secrets:
   ```bash
   cp .env.example .env
   # then edit .env and paste your HF_TOKEN
   ```
   `.env` is gitignored and will never be committed. Code loads it via
   `from dotenv import load_dotenv; load_dotenv()` (the package
   `python-dotenv` is in [environment.yml](environment.yml)). For shell
   scripts: `set -a && source .env && set +a`.

**Standing rules:**

- Never paste secrets (HF tokens, API keys) in chat, commit messages, or
  scripts. Use `.env` or the `huggingface-cli` cache.
- The repo is public; this is intentional (faster review, easier
  collaboration). Nothing in the repo proper is sensitive — only `.env`
  ever holds anything secret, and it never leaves your local machine.
- [.gitignore](.gitignore) defensively blocks `*.token`, `hf_token*`, and
  `.env` patterns. Verify with `git check-ignore -v .env` before committing.

---

## Citation (placeholder, pre-publication)

```bibtex
@misc{kumar2026offnumbermanifold,
  title  = {Off the Number Manifold: Geometric Signatures of Arithmetic Failures in Language Models},
  author = {Kumar, Anshul and P{\'o}czos, Barn{\'a}b{\'a}s},
  year   = {2026},
  note   = {BlackboxNLP 2026, in submission}
}
```

---

## License

Code: MIT. Documents: CC BY 4.0. Pre-publication; finalize before submission.
