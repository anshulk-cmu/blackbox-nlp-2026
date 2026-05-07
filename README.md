# Off-Manifold Failure: A Geometric Test for Localized Computational Errors in Language Models

**Target venue.** BlackboxNLP 2026 archival track (co-located with EMNLP 2026, Budapest). Direct submission deadline **July 17, 2026 (AoE)**.

**Authors.** Anshul Kumar, Barnábás Póczos.

**Status.** Methodology locked, math validated (45/45 toy PASS), ready for real-model run on **CMU Babel** via VS Code Remote-SSH.

---

## What this repo contains

| File | Purpose |
|---|---|
| [paper_math.md](paper_math.md) | Mathematical proposal (7 theorems, 11 lemmas, 4 propositions/corollaries). Authoritative source for theorem statements and proof outlines. |
| [full_paper_plan.md](full_paper_plan.md) | End-to-end methodology: pre-registration, 4-way data split, four ACE interventions, layer selection, five-method comparison, reviewer-defense matrix. |
| [babel_execution_plan.md](babel_execution_plan.md) | Step-by-step Babel execution plan: SLURM scripts, conda env, VS Code Remote-SSH workflow, resume logic, 7 phases (~24 GPU-hours total). |
| [environment.yml](environment.yml) | Pinned conda environment (`conda env create -f environment.yml`). |
| [toy/](toy/) | Synthetic-toy validation suite. **45 / 45 PASS in 245s on CPU.** Math verified before any GPU spend. |
| [code/](code/) | Importable utilities (`tokenizer_audit.py`, `accuracy_check.py`) + per-phase runners (`run_phase{1..7}_*.py`). |
| [scripts/](scripts/) | Shell wrappers and SLURM `.sbatch` files for each phase. |
| [KT_paper.md](KT_paper.md) | Reference: Kantamneni & Tegmark 2025, *Language models use trigonometry to do addition*. arXiv:2502.00873. |

---

## One-paragraph contribution

Kantamneni and Tegmark 2025 showed GPT-J 6B, Pythia 6.9B, and Llama 3.1 8B encode integers on a generalized helix and use a Clock-style algorithm for two-digit addition — but they study only correct cases. This paper introduces a *localized geometric diagnostic* for the **wrong** cases: given a candidate manifold recovered from correct activations and a hypothesized failure subspace V, the test identifies whether wrong-population activations sit on M, off M but inside the helix span, or outside the span entirely. The diagnostic comes with a finite-sample concentration bound (Theorem 1), explicit minimax-optimal power against directional alternatives (Theorem 2), a finite-sample manifold-recovery bound for the parametric estimator (Theorem 3), a Neyman-orthogonal cross-fitting protocol (Theorem 7.2), exact conditional null calibration via Lehmann–Romano matched permutation (Theorem 8.3), and a four-intervention causal pipeline with Hessian-bounded magnitude prediction (Proposition 9.3). On Llama 3.1 8B (where KT report the helix fits least cleanly), the diagnostic is the headline empirical case.

---

## How to read this repo

**For a reviewer or collaborator (~2 hours):**

1. **Skim [paper_math.md](paper_math.md)** for the math (~30 min). Sections 1–3 set up the assumptions; sections 4–9 are the theorem statements and proof outlines.
2. **Skim [full_paper_plan.md](full_paper_plan.md)** for methodology motivation (~45 min). Section 2 maps to paper_math.md theorems; section 3 is the pipeline; section 8 is the reviewer-defense matrix.
3. **Read [toy/README.md](toy/README.md)** end-to-end (~30 min) to see how the math was validated on synthetic ground truth.
4. **Read [babel_execution_plan.md](babel_execution_plan.md)** to see how the real-model run is organized (~15 min).

**For the executor (Anshul, when starting Babel work):**

1. Verify the [babel_execution_plan.md §3.6 pre-flight checklist](babel_execution_plan.md) (VS Code Remote-SSH connected, conda env built, HF login, `$BLACKBOX_DATA` set).
2. Run `python toy/run_toy.py` once. Confirm `45 / 45 PASS`.
3. Run Phase 1 on the login node: `bash scripts/phase1_audit.sh`.
4. Submit Phase 2 SLURM jobs: `sbatch scripts/phase2_accuracy.sbatch <model_key>` (one per model).
5. Continue through phases 3–7 per [babel_execution_plan.md](babel_execution_plan.md).

---

## Quick links by question

- **"What's the math?"** → [paper_math.md](paper_math.md), Theorems 4.2 / 5.2 / 6.2 / 6.6 / 7.2 / 8.3 / 9.3.
- **"What's the methodology?"** → [full_paper_plan.md §2–§3](full_paper_plan.md).
- **"How was the math validated?"** → [toy/README.md](toy/README.md) (45 PASS / 0 FAIL).
- **"How will the real run be executed?"** → [babel_execution_plan.md](babel_execution_plan.md).
- **"What's pre-registered before any data is touched?"** → [full_paper_plan.md §6](full_paper_plan.md).
- **"What's the four-intervention causal pipeline?"** → [full_paper_plan.md §3.9](full_paper_plan.md), Proposition 9.3.
- **"What if the experiment doesn't fire?"** → [full_paper_plan.md §7 fallback structure](full_paper_plan.md). Every interpretation step has a publishable null version.

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
├── KT_paper.md                      ← reference paper
├── paper_math.md                    ← math
├── full_paper_plan.md               ← methodology
├── babel_execution_plan.md          ← operations (Babel + VS Code Remote-SSH)
├── environment.yml                  ← conda env pinning
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

- **Real-model runs.** **CMU Babel** cluster via VS Code Remote-SSH. A100 80 GB for activation extraction + causal interventions; CPU nodes for tokenizer audit + manifold fitting + statistics + figures. Budget: ~24 GPU-hours total across 7 phases.
- **Toy runs.** Any Python 3.10+ environment with `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `torch`, `pandas`. CPU is sufficient. ~4 minutes wall clock.
- **Persistent storage.** Babel scratch at `/data/user_data/$USER/blackbox/` (set as `$BLACKBOX_DATA`).

---

## Security notes

- HuggingFace tokens go **only** in `~/.cache/huggingface/token` (created by `huggingface-cli login`). Never paste in chat, files, commit messages, or scripts.
- The repo is public; this is intentional (faster review, easier collaboration). Nothing in the repo is sensitive.
- `.gitignore` defensively blocks `*.token`, `hf_token*`, and `.env` patterns.

---

## Citation (placeholder, pre-publication)

```bibtex
@misc{kumar2026offmanifold,
  title  = {Off-Manifold Failure: A Geometric Test for Localized Computational Errors in Language Models},
  author = {Kumar, Anshul and P{\'o}czos, Barn{\'a}b{\'a}s},
  year   = {2026},
  note   = {BlackboxNLP 2026, in submission}
}
```

---

## License

Code: MIT. Documents: CC BY 4.0. Pre-publication; finalize before submission.
