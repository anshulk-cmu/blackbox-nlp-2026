# Colab Pro Execution Plan

**Status.** Operations document for the BlackboxNLP 2026 submission.

**Companions.**
- [paper_math.md](paper_math.md) — authoritative math (7 theorems, 11 lemmas, proof outlines).
- [full_paper_plan.md](full_paper_plan.md) — authoritative methodology (pre-registration, 4-way data split, 4 ACEs).
- [toy/README.md](toy/README.md) — synthetic-toy validation (45 / 45 PASS).
- **This document** — authoritative *execution plan* on Google Colab Pro / Pro+.

When this document and any other doc disagree on operational details (paths,
GPU choices, runtime estimates, environment setup), this document wins.
For math and methodology, the other documents win.

---

## 0. The decision: Colab Pro, not Babel

The earlier plan referenced CMU Babel cluster. We are pivoting to **Google
Colab Pro / Pro+** because:

- Faster iteration cycles (no SLURM queue waits).
- A100 40 GB + L4 + V100 + T4 all available; pick per workload.
- Notebooks are easy to share and inspect during review.
- Drive integration gives persistent activation cache without scratch
  bookkeeping.
- The compute budget for the entire pipeline is ~30 GPU-hours, well within
  Colab Pro+ monthly allotment.

The trade-offs we accept:

- **A100 40 GB instead of 80 GB on Babel.** Llama 3.1 8B (bf16, ~16 GB
  weights + activations) fits comfortably with batch size 4 on a 40 GB A100.
  This was already the working assumption in the toy.
- **Session timeouts.** Pro: 12 h foreground / 24 h with payment. Pro+:
  24 h. We engineer all phases to be *resumable from Drive cache* so a
  dropped session never costs more than the work since the last cache flush
  (every ~50 prompts).
- **No tmux / persistent shell.** Each phase runs as a notebook; long-running
  loops use tqdm + Drive flush for crash safety.
- **HuggingFace gated models** (Llama 3.1 needs a HF token with Llama license
  agreement). Stored as a Colab Secret named `HF_TOKEN`.

---

## 1. The execution graph (phases at a glance)

```
[Phase 0: Local prep]            CPU,  laptop          ~30 min
       │
       ▼
[Phase 1: Tokenizer audit]       CPU,  Colab free      ~10 min
       │  -> Drive: /tokenizer_audit/{model}.json
       ▼
[Phase 2: Accuracy reproduction] T4 → A100, Colab Pro  ~6 h   (3 models)
       │  -> Drive: /correctness/{model}.parquet
       ▼
[Phase 3: Activation extraction] A100 40 GB, Colab Pro ~9 h   (3 models × 4 layers)
       │  -> Drive: /activations/{model}/layer_{L}.npz
       ▼
[Phase 4: Manifold + layer pick] CPU or T4             ~1 h
       │  -> Drive: /manifolds/{model}/M_hat_layer_{L_star}.npz
       ▼
[Phase 5: T_n statistics]        CPU                   ~30 min
       │  -> Drive: /stats/{model}/T_n_{version}.json
       ▼
[Phase 6: Causal interventions]  A100 40 GB            ~9 h   (3 models)
       │  -> Drive: /aces/{model}.json
       ▼
[Phase 7: Aggregation + figures] CPU                   ~30 min
          -> Drive: /paper_figs/figure_{2..6}.pdf
```

**Total wall time: ~26 hours.** Target: complete one phase per day across a
week, with a buffer week before the July 17 deadline for re-runs and
camera-ready prep.

---

## 2. Storage plan (Google Drive)

Mount Drive at `/content/drive/MyDrive/`. Project root at
`/content/drive/MyDrive/blackbox_nlp_2026/`.

```
blackbox_nlp_2026/
├── code/                       # cloned from this repo (git pull on session start)
├── tokenizer_audit/            # Phase 1 outputs
│   ├── gpt-j-6b.json
│   ├── pythia-6.9b.json
│   ├── llama-3.1-8b.json
│   └── intersection.json       # problems retained by all three
├── correctness/                # Phase 2 outputs (greedy decode results)
│   ├── gpt-j-6b.parquet        # cols: a, b, s, predicted, correct
│   ├── pythia-6.9b.parquet
│   └── llama-3.1-8b.parquet
├── activations/                # Phase 3 outputs (residual stream at "=" token)
│   ├── gpt-j-6b/
│   │   ├── layer_07.npz        # cols: h (n × d), a, b, s, correct
│   │   ├── layer_14.npz
│   │   ├── layer_21.npz
│   │   └── layer_27.npz
│   ├── pythia-6.9b/
│   │   ├── layer_08.npz
│   │   ├── layer_16.npz
│   │   ├── layer_24.npz
│   │   └── layer_31.npz
│   └── llama-3.1-8b/
│       ├── layer_08.npz
│       ├── layer_16.npz
│       ├── layer_24.npz
│       └── layer_31.npz
├── manifolds/                  # Phase 4 outputs
│   ├── gpt-j-6b/
│   │   ├── M_hat_param.npz     # parametric fit at chosen layer
│   │   ├── M_hat_dm.npz        # diffusion maps fit
│   │   └── selection.json      # chosen layer, R², which method primary
│   ├── pythia-6.9b/...
│   └── llama-3.1-8b/...
├── stats/                      # Phase 5 outputs
│   ├── gpt-j-6b/
│   │   ├── T_n_naive.json
│   │   ├── T_n_cross.json
│   │   ├── T_n_matched.json
│   │   ├── T_n_V.json          # V_1..V_5 results with BH-FDR
│   │   └── failure_modes.json  # T_curve vs T_span per mode
│   └── ...
├── aces/                       # Phase 6 outputs
│   ├── gpt-j-6b.json           # ACE_N, ACE_NV, ACE_S, ACE_R + Prop 9.3 prediction
│   └── ...
├── paper_figs/                 # Phase 7 outputs (paper-ready)
│   ├── figure_1_failure_modes_schematic.pdf
│   ├── figure_2_principal_angle_matrix.pdf
│   ├── figure_3_T_n_three_versions.pdf
│   ├── figure_4_localization.pdf
│   ├── figure_5_ACEs.pdf
│   └── figure_6_proposition_4_match.pdf
└── logs/                       # per-phase logs for reproducibility
    └── {phase}_{model}_{date}.log
```

**Drive quota.** Activations dominate. Per (model, layer):
`n × d × 4 bytes ≈ 5000 × 4096 × 4 = 80 MB`. Three models × 4 layers each =
`12 × 80 MB = 1 GB`. Total project quota: ~5 GB including correctness
parquet files and logs. Well within free Drive (15 GB).

---

## 3. Environment setup (every notebook starts with this)

Standard prelude block, paste at the top of every notebook:

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Pull latest code from this repo
import os
PROJECT = '/content/drive/MyDrive/blackbox_nlp_2026'
CODE_DIR = f'{PROJECT}/code'
REPO_URL = 'https://github.com/anshulk-cmu/blackbox-nlp-2026.git'
if not os.path.exists(CODE_DIR):
    !git clone {REPO_URL} {CODE_DIR}
%cd {CODE_DIR}
!git pull --ff-only

# 3. Pin dependency versions
!pip install -q \
  torch==2.4.1 \
  transformers==4.45.0 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  scikit-learn==1.5.2 \
  pandas==2.2.2 \
  pyarrow==17.0.0 \
  matplotlib==3.9.2 \
  tqdm==4.66.5

# 4. HuggingFace login (needed for Llama 3.1).
#    HF_TOKEN must be stored in Colab Secrets (🔑 sidebar in the notebook,
#    NOT in any committed file). The token must have READ scope and the
#    Llama 3.1 license must be accepted on the HF account that issued it.
from google.colab import userdata
import huggingface_hub
huggingface_hub.login(userdata.get('HF_TOKEN'))

# 5. Confirm GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
```

**Runtime selection per phase.**

| Phase | Runtime | Why |
|---|---|---|
| 0 (local prep) | n/a | Local laptop |
| 1 (tokenizer audit) | CPU | No model load, pure tokenizer ops |
| 2 (accuracy reproduction) | A100 40GB (Llama), T4 (GPT-J/Pythia) | Forward-only |
| 3 (activation extraction) | A100 40GB | bf16 forward + hooks |
| 4 (manifold fitting) | CPU or T4 | n × d × K matrix ops, fits in RAM |
| 5 (T_n statistics) | CPU | Pure NumPy on cached activations |
| 6 (causal interventions) | A100 40GB | Forward passes with patched residuals |
| 7 (figures + aggregation) | CPU | matplotlib, JSON aggregation |

In the Colab UI: **Runtime → Change runtime type → Hardware accelerator → A100 GPU**.

---

## 4. Phase 0: Local preparation (before opening Colab)

**Owner: Anshul, ~30 min on local laptop.**

**Account note.** Colab Pro+ is on `work.anshul95@gmail.com`. All Drive
paths in §2 of this document assume that account is the active one when the
notebook is opened. The Drive root therefore is
`/content/drive/MyDrive/blackbox_nlp_2026/` on `work.anshul95@gmail.com`.
GitHub repo is on `anshulk-cmu` (separate account, SSH-authed locally,
HTTPS clone in Colab via the public repo URL).

Steps:

1. **Push the current repo** (toy/, paper_math.md, full_paper_plan.md, this
   file) to a GitHub repo. **Done 2026-05-06 →
   https://github.com/anshulk-cmu/blackbox-nlp-2026.** Notebooks
   `git clone https://github.com/anshulk-cmu/blackbox-nlp-2026.git`.
2. **Add Llama 3.1 license agreement** at
   https://huggingface.co/meta-llama/Llama-3.1-8B (request access on the HF
   account that will issue the token used in Colab Secrets — typically the
   same account as `work.anshul95@gmail.com` for clean isolation).
3. **Create HuggingFace token** at https://huggingface.co/settings/tokens
   with **read** access. Store as a Colab Secret named `HF_TOKEN`
   (Colab notebook: 🔑 sidebar → Add new secret). **Never paste this token
   into chat, commit, or write to any file.**
4. **Create the Drive folder** `/MyDrive/blackbox_nlp_2026/` on
   `work.anshul95@gmail.com` (empty; notebooks will populate it).
5. **Confirm Colab Pro / Pro+ subscription is active on
   `work.anshul95@gmail.com`** (compute units > 100).
6. **Read this document end-to-end** before starting any phase.

Pre-flight checklist (must all be ✓ before Phase 1):

- [x] Repo pushed to GitHub (public, anshulk-cmu/blackbox-nlp-2026). Done 2026-05-06.
- [ ] Llama 3.1 license accepted on HF.
- [ ] HF_TOKEN stored in Colab Secrets (revoke any token that was ever
      pasted in chat or file; create a fresh read-scope one).
- [ ] `/MyDrive/blackbox_nlp_2026/` folder exists on work.anshul95@gmail.com.
- [ ] Colab Pro or Pro+ active on work.anshul95@gmail.com.
- [ ] `python toy/run_toy.py` passes 45/45 PASS locally (sanity check that the analysis code is healthy).

---

## 5. Phase 1: Tokenizer audit

**Goal.** Determine which `(a, b)` pairs in `{0,...,99}²` produce
single-token operands AND single-token answer in each of the three model
tokenizers, and compute the three-way intersection.

**Why first.** Defining the analysis dataset is a prerequisite for every
later phase. Pre-registered per [full_paper_plan.md §3.2](full_paper_plan.md).

**Runtime.** CPU on Colab free (no model load needed for tokenizer ops).

**Notebook: `01_tokenizer_audit.ipynb`** — outline:

1. Standard prelude (Section 3 of this doc).
2. Load each tokenizer:
   ```python
   from transformers import AutoTokenizer
   models = {
     'gpt-j-6b': 'EleutherAI/gpt-j-6B',
     'pythia-6.9b': 'EleutherAI/pythia-6.9b',
     'llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
   }
   tokenizers = {k: AutoTokenizer.from_pretrained(v) for k, v in models.items()}
   ```
3. For each model, for each `(a, b) ∈ {0,...,99}²`:
   - Tokenize the prompt template (per-model template per [full_paper_plan.md §3.1](full_paper_plan.md)).
   - Tokenize the operands a, b individually and confirm they are single tokens.
   - Tokenize the answer `s = a + b` (range `[0, 198]`) and confirm single-token.
   - Record retain/drop and the reason.
4. Save per-model JSON: `tokenizer_audit/{model}.json` with cols
   `[a, b, s, retained: bool, reason_dropped: str | null]`.
5. Compute three-way intersection: `(a, b)` retained by ALL three models.
   Save as `tokenizer_audit/intersection.json`.
6. Print summary:
   ```
   GPT-J 6B:    {n_retained_gpt} / 10000 retained
   Pythia 6.9B: {n_retained_pyt} / 10000 retained
   Llama 3.1 8B: {n_retained_llm} / 10000 retained
   Intersection (used for primary analysis): {n_intersection} / 10000
   ```
7. Sanity assertion: `n_intersection >= 7000`. If less, investigate
   tokenizer mismatches before continuing.

**Outputs to Drive:**
- `tokenizer_audit/{gpt-j-6b,pythia-6.9b,llama-3.1-8b}.json`
- `tokenizer_audit/intersection.json`

**Failure modes to watch for:**
- HF model name typo → 404 from `from_pretrained`.
- Llama license not accepted → 403 from HF (re-login with token).
- Multi-digit operands tokenize as multi-token in one model but not others
  (expected: this is exactly what the audit measures; no action needed).

---

## 6. Phase 2: Accuracy reproduction

**Goal.** Reproduce KT's reported per-model accuracies (80.5% / 77.2% / 98%)
on the intersection set. Establishes that the prompt + tokenizer + greedy
decoding pipeline is set up correctly.

**Why second.** If accuracies are wildly off from KT's, something is wrong
with the prompt or generation, and nothing downstream will work.

**Runtime.** Three notebooks, one per model, A100 for Llama, T4 acceptable
for GPT-J / Pythia. Per model: ~2 hours.

**Notebooks: `02a_accuracy_gptj.ipynb`, `02b_accuracy_pythia.ipynb`, `02c_accuracy_llama.ipynb`** — same structure:

1. Standard prelude. Set runtime: A100 (Llama) or T4 (GPT-J/Pythia).
2. Load model in bf16 (A100) or fp16 (T4):
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(
       hf_name, torch_dtype=torch.bfloat16, device_map='cuda'
   )
   model.eval()
   ```
3. Load `tokenizer_audit/intersection.json` from Drive.
4. For each `(a, b)` in the intersection, in batches of 32:
   - Build prompt per [full_paper_plan.md §3.1](full_paper_plan.md).
   - Forward pass; take argmax of next-token logits.
   - Compare to ground-truth single-token answer.
   - Cache to Drive every 50 batches (resume safety).
5. Save `correctness/{model}.parquet` with cols `[a, b, s, predicted, correct]`.
6. Print:
   ```
   {model}: accuracy = {n_correct}/{n_total} = {pct:.1f}%
   KT reports:  {kt_pct}%
   |delta|:     {abs_delta:.1f}pp
   ```
7. **Sanity gate:** if `|delta| > 5pp`, halt and re-check the prompt template
   before Phase 3. Five-percentage-point gap is large enough that Phase 3+
   results would be uninterpretable.

**Outputs to Drive:**
- `correctness/{model}.parquet`

**Per-model GPU notes:**
- **GPT-J 6B**: 12 GB in fp16; T4 (16 GB) sufficient. ~2 h.
- **Pythia 6.9B**: 14 GB in fp16; T4 tight, prefer V100 (16 GB) or A100. ~2 h.
- **Llama 3.1 8B**: 16 GB in bf16; A100 40 GB recommended. ~3 h.

---

## 7. Phase 3: Activation extraction

**Goal.** Cache residual-stream activations at the `=` token for each
candidate analysis layer per model. The four candidates per model are
pre-registered in [full_paper_plan.md §3.6](full_paper_plan.md):

- GPT-J (28 layers): {7, 14, 21, 27}
- Pythia (32 layers): {8, 16, 24, 31}
- Llama (32 layers): {8, 16, 24, 31}

**Runtime.** Three notebooks, one per model, A100 each. Per model: ~3 hours.

**Notebooks: `03a_extract_gptj.ipynb`, etc.** — same structure:

1. Standard prelude. **Runtime: A100 40GB.**
2. Load model in bf16, hooks on the four candidate blocks:
   ```python
   activations = {layer: [] for layer in CANDIDATE_LAYERS}
   def make_hook(layer_idx):
       def hook(module, inp, out):
           # `out` is the residual-stream output of block `layer_idx`.
           # We capture only the last-token (= sign) activation per sample.
           activations[layer_idx].append(out[:, -1, :].detach().to(torch.float32).cpu().numpy())
       return hook

   handles = []
   for L in CANDIDATE_LAYERS:
       layer_module = get_layer(model, L)  # transformer.h.L for GPT-J, etc.
       handles.append(layer_module.register_forward_hook(make_hook(L)))
   ```
3. Load `correctness/{model}.parquet`. Loop over all retained samples in
   batches of 4 (memory safety on A100 40 GB).
4. Forward pass per batch; hooks capture activations.
5. Every 50 batches, flush to Drive (`activations/{model}/layer_{L}_partial_{batch_idx}.npz`)
   to recover from session drops.
6. End: concatenate partials into the full per-layer files
   `activations/{model}/layer_{L}.npz` with cols
   `[h: (n, 4096), a: (n,), b: (n,), s: (n,), correct: (n,)]`.
7. Sanity assertion: `h.shape[0]` matches the retained-sample count from
   tokenizer audit.

**Memory math (Llama 3.1 8B, A100 40 GB).**
- Model weights bf16: ~16 GB.
- Per-token activations bf16: `4096 × 4 bytes ≈ 16 KB` per sample per layer.
- Batch size 4 with 4 hooks: `4 × 4 × 16 KB = 256 KB` extra per forward.
- Peak forward-pass memory: ~22 GB (with KV cache and intermediate buffers).
- Comfortable margin within 40 GB.

If A100 is unavailable some sessions, **fall back to V100 16 GB**: reduce
bf16 → fp16, drop batch size to 1, expect ~6 h per model.

**Outputs to Drive:**
- `activations/{model}/layer_{07,14,21,27}.npz` (GPT-J)
- `activations/{model}/layer_{08,16,24,31}.npz` (Pythia, Llama)

**Failure modes:**
- Out-of-memory mid-batch: drop batch size to 2, restart from last cached
  partial.
- Session drop at hour 2.5: resume from partials in Drive; the loop's
  outer `for` loop is keyed on batch index in the partial filename so it
  picks up where it left off.
- bf16 underflow on T4 fallback (no bf16 hardware): use fp16 instead and
  flag in logs.

---

## 8. Phase 4: Manifold fitting and layer selection

**Goal.** For each model, fit M̂ at every candidate layer using the
parametric estimator, compute held-out R², select the best layer per
[full_paper_plan.md §3.6](full_paper_plan.md), and produce the primary M̂
per model. Also fit Diffusion Maps as the fallback path.

**Runtime.** ~1 hour total, CPU sufficient.

**Notebook: `04_manifolds_and_layer_selection.ipynb`**:

1. Standard prelude. Runtime: CPU.
2. Import `toy/manifold_methods.py` (`fit_parametric`, `fit_diffusion_maps`,
   `sin_theta_max`).
3. For each model, for each candidate layer:
   - Load `activations/{model}/layer_{L}.npz`.
   - Restrict to correct samples.
   - 80/20 split for layer selection.
   - Fit parametric M̂ on 80%; compute held-out R² on 20%.
4. Per-model selection rule (pre-registered):
   - If max R² ≥ 0.90 across candidate layers, choose argmax.
   - Else: switch primary M̂ to Diffusion Maps; same layer pick rule on its
     R² metric (refit on the embedding-then-lift pipeline).
   - Save selection rationale.
5. Refit primary M̂ (parametric or DM) on full correct population at chosen
   layer.
6. Also fit the secondary M̂s (PCA-on-bins, Local PCA, Kernel PCA) for the
   five-method comparison figure.
7. Compute pairwise principal angles between all five M̂s. Save matrix.
8. Save:
   - `manifolds/{model}/M_hat_param.npz` — primary parametric U_M (orthonormal columns).
   - `manifolds/{model}/M_hat_dm.npz` — diffusion-maps fallback U_M.
   - `manifolds/{model}/principal_angle_matrix.npz` — 5×5 matrix of pairwise sin θ_max.
   - `manifolds/{model}/selection.json` — chosen layer, R², primary method.

**Outputs to Drive:**
- `manifolds/{model}/M_hat_*.npz`
- `manifolds/{model}/selection.json`
- `manifolds/{model}/principal_angle_matrix.npz`

**Sanity asserts:**
- All M̂s have orthonormal columns: `||U_M.T @ U_M - I||_F < 1e-6`.
- For at least 2/3 models, parametric R² ≥ 0.90 (matches KT for GPT-J,
  Pythia; Llama may fail and trigger DM fallback per Figure 23).

---

## 9. Phase 5: T_n statistics

**Goal.** For each model at the chosen layer, compute the three versions of
T_n (naive / cross-fitted / matched + cross-fitted), the localized T_n^V
across V_1..V_5, and the per-failure-mode breakdown (T_curve vs T_span).

**Runtime.** ~30 min total, CPU sufficient.

**Notebook: `05_test_statistics.ipynb`**:

1. Standard prelude. Runtime: CPU.
2. Import `toy/tests.py` — exact same code path the toy validated.
3. Load activations and primary M̂ per model.
4. For each model:
   - **Three versions of T_n** (paper_math.md Theorem 4.2):
     - Naive: single-fit M̂ on all correct.
     - Cross-fitted: K=5 folds (paper_math.md §7).
     - Matched + cross-fitted: stratified within-bin permutation
       (paper_math.md Theorem 8.3) using `make_phi_bins`.
   - **Bootstrap 95% CIs** (1000 resamples) on each version.
   - **Localization across V_1..V_5** (paper_math.md Theorem 5.2):
     - V_1 (carry direction): LDA on carry labels in correct samples.
     - V_2 (T=2 direction): cos(πa) basis column.
     - V_3 (higher Fourier): cos/sin pairs for T ∈ {3, 4, 6, 7, 8}.
     - V_4 (random baseline): 100 random unit directions ⊥ M̂.
     - V_5 (full perp): orthonormal basis for I − P_M̂.
     - Permutation p-values + BH-FDR at q=0.05.
   - **Three failure modes** (paper_math.md §2.3):
     - T_curve vs T_span on wrong samples.
     - Classify each model into the dominant mode.
5. Save:
   - `stats/{model}/T_n_naive.json`, `T_n_cross.json`, `T_n_matched.json`.
   - `stats/{model}/T_n_V.json`.
   - `stats/{model}/failure_modes.json`.

**Outputs to Drive:**
- `stats/{model}/T_n_*.json`
- `stats/{model}/T_n_V.json`
- `stats/{model}/failure_modes.json`

**Pre-registered positive-result criteria** (per [full_paper_plan.md §6.4](full_paper_plan.md)):
- T_n^matched > 0 with permutation p < 0.05 on at least 2/3 models.
- T_n^{V_k} > 0 with BH-corrected p < 0.05 for at least one V ∈ {V_1, V_2, V_3} on those models.

---

## 10. Phase 6: Causal interventions

**Goal.** Run the four-intervention causal pipeline (N, NV, S, R) on each
model at the chosen layer; compute Proposition 9.3 prediction (Hessian-bounded
ACE_S Taylor remainder).

**Runtime.** Three notebooks, A100 each. ~3 hours per model.

**Notebooks: `06a_causal_gptj.ipynb`, `06b_causal_pythia.ipynb`, `06c_causal_llama.ipynb`** — same structure:

1. Standard prelude. **Runtime: A100 40GB.**
2. Load model in bf16, register patching hook on the chosen analysis layer
   per [full_paper_plan.md §3.9.6](full_paper_plan.md):
   - GPT-J: `transformer.h.{ℓ}` forward_pre_hook on next block.
   - Pythia: `gpt_neox.layers.{ℓ}` block input.
   - Llama: `model.layers.{ℓ}` block output (post-residual).
3. Load activations (already cached) + chosen V (from Phase 5 output).
4. **Estimate μ̂_ξ** from cross-fitted residuals projected onto V.
5. **Run interventions**:
   - **ACE_N**: project all wrong activations onto M̂; replay through
     subsequent layers; measure ΔLD.
   - **ACE_NV**: zero out V-component of wrong activations; replay; ΔLD.
   - **ACE_S**: inject μ̂_ξ into correct activations; replay; ΔLD.
   - **ACE_R**: same as N-V but with 100 random V_random of matched dim.
6. **Bootstrap 95% CIs** (1000 resamples) per ACE.
7. **Proposition 9.3 prediction**: compute `ACE_S^linear` via autograd on
   `LD_τ` with τ=1.0, plus the Hessian operator-norm bound; report
   `|ACE_S - ACE_S^linear|` vs predicted bound.
8. Save `aces/{model}.json`:
   ```json
   {
     "ACE_N":  {"mean": ..., "ci_lo": ..., "ci_hi": ...},
     "ACE_NV": {"mean": ..., "ci_lo": ..., "ci_hi": ...},
     "ACE_S":  {"mean": ..., "ci_lo": ..., "ci_hi": ...},
     "ACE_R":  {"mean_over_replicates": ..., "std_over_replicates": ...},
     "Proposition_9_3": {
       "ACE_S_linear":      ...,
       "ACE_S_actual":      ...,
       "abs_diff":          ...,
       "Hessian_L":         ...,
       "remainder_bound":   ...,
       "bound_holds":       true
     }
   }
   ```

**Pre-registered positive-result criteria** (per [full_paper_plan.md §3.9.4](full_paper_plan.md)):
- ACE_N > 0 with bootstrap p < 0.05.
- ACE_NV ≥ 0.7 × ACE_N.
- ACE_S < 0 with bootstrap p < 0.05.
- ACE_NV − ACE_R > 2 × std(ACE_R).

**Memory math (causal patching, Llama 3.1 8B, A100 40 GB).**
- Model bf16: 16 GB.
- KV cache for forward at 32 layers, batch 1, prompt length ~20: ~50 MB.
- Patch operation adds negligible memory.
- Comfortable margin within 40 GB; can scale to batch 4.

---

## 11. Phase 7: Aggregation and figures

**Goal.** Build the six figures and result tables for the paper.

**Runtime.** ~30 min, CPU.

**Notebook: `07_figures_and_paper_table.ipynb`**:

1. Load all outputs from Drive (every JSON + every npz).
2. **Figure 1** (`paper_figs/figure_1_failure_modes_schematic.pdf`):
   schematic illustration of the three failure modes — manually styled.
3. **Figure 2** (`figure_2_principal_angle_matrix.pdf`):
   5×5 sin θ_max matrix per model (3 subplots), heatmap.
4. **Figure 3** (`figure_3_T_n_three_versions.pdf`):
   bar chart with bootstrap CIs for T_n^naive / T_n^cross / T_n^matched
   per model.
5. **Figure 4** (`figure_4_localization.pdf`):
   T_n^{V_k} per V per model with BH-FDR significance markers.
6. **Figure 5** (`figure_5_ACEs.pdf`):
   bar chart of ACE_N / ACE_NV / ACE_S / ACE_R per model with 95% CIs.
7. **Figure 6** (`figure_6_proposition_4_match.pdf`):
   scatter of predicted ACE_S^linear vs actual ACE_S, overlaid with the
   Hessian-bounded remainder envelope.
8. **Master table** of every numerical claim, exported as
   `paper_figs/master_results_table.csv`.

---

## 12. Failure recovery and resumability

Every notebook follows three rules:

1. **Idempotent cell ordering.** Re-running cells from top should not
   double-write or corrupt state.
2. **Drive flush every 50 iterations** for any loop running > 5 minutes.
   Partial outputs go to `partial_{batch_idx}.npz`; concatenation step at
   the end of the loop reads all partials and produces the final file.
3. **Resume by checking Drive first.** Every notebook starts with
   `if Path(output_path).exists(): print('cached, skipping'); return`.

If a session drops mid-phase:
- Re-open the notebook on the same Colab account.
- Re-run the prelude.
- Run the main loop; cached partials are picked up automatically.

If a session is *completely* lost (rare):
- The Drive cache is the source of truth. Re-fitting from cached activations
  is < 1 hour for any phase except 3 (extraction) and 6 (causal interventions).

---

## 13. Compute budget estimate

| Phase | Runtime type | Wall clock | Compute units (Pro+) |
|---|---|---|---|
| 1 | CPU | 10 min | 0 |
| 2 | A100 (Llama) + T4 (others) | 6 h | ~30 |
| 3 | A100 × 3 | 9 h | ~120 |
| 4 | CPU | 1 h | 0 |
| 5 | CPU | 30 min | 0 |
| 6 | A100 × 3 | 9 h | ~120 |
| 7 | CPU | 30 min | 0 |
| **Total** |  | **~26 h** | **~270 compute units** |

Colab Pro+ allotment: 500 compute units / month. We have ~85% margin for
re-runs, layer-sweep extensions, sensitivity analyses. **Comfortable.**

---

## 14. Documentation completeness checklist

Before any code is written, every item below must be ✓ for the doc set to
be considered complete.

### Math (paper_math.md)
- [x] All 7 theorems stated with explicit assumptions (GM, BD, REG).
- [x] All 11 lemmas with proof outlines.
- [x] Sharp Laurent–Massart form (Theorem 4.2(d)).
- [x] Minimax lower bound (Theorem 5.2(b)) with chaining flagged.
- [x] Exact null distribution of T_n^V (Theorem 5.2(c), corrected scaling).
- [x] Misspecification bias decomposition (Theorem 6.5).
- [x] Composed bound (Theorem 6.6).
- [x] Influence function variance V_inf (Corollary 7.3).
- [x] Lehmann–Romano matched permutation (Theorem 8.3).
- [x] Hessian-bounded Proposition 9.3.
- [x] Open questions for Barnábás listed (§11).
- [x] References (paper_math.md §References).

### Methodology (full_paper_plan.md)
- [x] Pre-registered V_1..V_5 (§3.8).
- [x] 4-way data split (§3.4 / §3.6).
- [x] Three-version T_n reporting (§3.7).
- [x] Four ACEs with positive-result criteria (§3.9.4).
- [x] Layer selection protocol (§3.6).
- [x] Tokenizer audit protocol (§3.2).
- [x] Five-method comparison (§2.6, §4).
- [x] Limitations stated (§5).
- [x] Reviewer-defense matrix (§8, 22 entries).
- [x] Pre-registration locked (§6).
- [x] Crosswalk to paper_math.md (§14).
- [x] Babel references replaced with Colab Pro (done 2026-05-06).

### Synthetic toy (toy/README.md)
- [x] All 16 experiments documented (E1–E16).
- [x] 45/45 PASS recorded.
- [x] Bug log of issues caught (§10).
- [x] theorem_predictions.py crosswalk to paper_math.md.
- [x] Update note re: Colab compatibility (done 2026-05-06).

### Operations (this document)
- [x] Compute environment (Colab Pro/Pro+).
- [x] Storage plan (Drive paths).
- [x] Phase-by-phase execution plan.
- [x] Per-notebook structure.
- [x] Pre-flight checklist.
- [x] Failure recovery.
- [x] Compute budget estimate.

### Outstanding (must complete before code):
- [x] Update [full_paper_plan.md](full_paper_plan.md) §0a / §3.6 / §3.9.6 /
      §0a-implication paragraph: replace "Babel" with "Colab Pro" and
      replace `/data/user_data/anshulk/...` paths with the Drive paths
      defined in §2 of this document. **Done 2026-05-06.**
- [x] Update [toy/README.md](toy/README.md): note that the toy runs on
      laptop CPU OR Colab CPU (no GPU needed for toy validation).
      **Done 2026-05-06.**
- [ ] Confirm the pre-flight checklist (§4 of this document) is signed off
      by the person who'll execute Phase 1. *(Owner: Anshul; pending.)*

---

## 15. References to update in other docs

After this document is committed, update the cross-doc references so all
documents agree on Colab as the execution environment.

**full_paper_plan.md:**
- §0a "Before running on GPT-J / Pythia / Llama on Babel" → "on Colab Pro"
- §0a "Bugs the toy caught (would have surfaced only on Babel otherwise)" → "on Colab"
- §0a "on Babel scratch" → "on Drive"
- §0a "The Babel run is now" → "The Colab run is now"
- §3.6 "Cache to scratch storage on Babel as `/data/user_data/anshulk/...`" →
  "Cache to Drive at `/content/drive/MyDrive/blackbox_nlp_2026/activations/{model}/layer_{ℓ}.npz`"
- §3.9.6 "we use bf16 weights on Babel A100 80 GB" → "we use bf16 weights
  on Colab A100 40 GB"

**toy/README.md:**
- "before we book GPU time on Babel" → "before we run the real-model
  notebooks on Colab Pro"
- "would otherwise have surfaced only on Babel" → "would otherwise have
  surfaced only when the real-model notebooks ran on Colab"
- "real Babel runs need that cache plumbing to be resumable" → "real-model
  Colab runs need that cache plumbing to be resumable"
- "Babel run says we can get" → "Colab run says we can get"
- "Re-running on Babel (Linux + MKL) may produce values" → "Re-running on
  Colab (Linux + MKL) may produce values"

---

## 16. Hand-off ready

When all checklist items in §14 are ✓ and the references in §15 are
applied, the documentation set is hand-off-ready. The person executing
Phase 1 should be able to:

1. Read this document end-to-end (~30 min).
2. Skim [paper_math.md](paper_math.md) for context (~30 min).
3. Skim [full_paper_plan.md](full_paper_plan.md) for the methodology motivation (~45 min).
4. Run the toy locally (`python toy/run_toy.py`) and confirm 45/45 PASS (~5 min).
5. Open Phase 1 notebook in Colab and execute (~10 min).
6. Continue through phases 2–7 over the following ~2 weeks.

Total handoff prep time: ~2 hours of reading + ~10 min of toy validation
before the first Colab cell is executed.

---

*End of Colab execution plan.*
