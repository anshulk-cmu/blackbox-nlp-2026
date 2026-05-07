# Babel Execution Plan

**Status.** Authoritative *operations* document for the BlackboxNLP 2026
submission. Version 2 (rewritten from scratch on 2026-05-06 to integrate
every theorem, lemma, and methodological choice from the math and plan
files).

**Companions and authority order.**
- [paper_math.md](paper_math.md) — authoritative *math* (7 theorems,
  numerous lemmas, proof outlines, all assumptions GM / BD / REG).
- [full_paper_plan.md](full_paper_plan.md) — authoritative *methodology*
  (pre-registration, 4-way data split, 4 ACEs, layer selection rules).
- [KT_paper.md](KT_paper.md) — authoritative *prior work* reference
  (helix definition, Clock algorithm, prompt templates, KT figures we
  need to replicate or extend).
- [toy/README.md](toy/README.md) — synthetic-toy validation
  (45 / 45 PASS as of 2026-05-06).
- **This document** — authoritative *execution plan* on CMU Babel
  cluster. Operational details (paths, GPU choices, SLURM scripts,
  timing, fallback recipes) live here.

When this document and any other doc disagree on operational details
this document wins. For math the math file wins. For methodology the
plan file wins. For prior-art references the KT file wins.

**Audience for this document.** A reader who knows the methodology but
needs to actually push the buttons on Babel without re-reading the
theory papers every time. Every step is spelled out in plain language;
math citations point back rather than re-derive. If something feels
under-explained the right move is to open the relevant file at the
section we cite.

---

## 0. Bird's-eye view in one minute

We have three pre-trained language models (GPT-J 6B, Pythia 6.9B, Llama
3.1 8B) and one task (two-digit addition `a + b` for `a, b ∈ [0, 99]`).
For each model we extract residual-stream activations at the equals-sign
token over all 10,000 ordered problems, split into correct and wrong
populations by greedy decoding, fit a candidate manifold `M̂` (the helix
basis of KT 2025) on the correct population, and run a localized
geometric test (`T_n` and `T_n^V`) plus a four-intervention causal
pipeline (necessity, localized necessity, sufficiency, specificity).
The pipeline is theorized in `paper_math.md` and pre-registered in
`full_paper_plan.md`. This document tells you exactly which buttons to
press in which order on Babel to make it happen.

The whole pipeline is broken into 8 phases (numbered 0 through 7) with
checkpointed outputs and explicit gate criteria between phases. Every
phase is resumable. The wall-time budget is ~24 GPU-hours plus
~3 CPU-hours; total elapsed time including SLURM queueing is
~3–5 days.

---

## 0.1 Why Babel, not Colab

We tried Colab Pro+ first and ran into:
- Browser-only UI; no integration with VS Code where the rest of
  development happens.
- Session timeouts and intermittent secrets-fetch failures.
- Stateful kernels with module caching that hide "did I actually pull
  the new code?" issues.
- 40 GB GPU ceiling — tight for Llama 3.1 8B with the activation hooks
  we need.

Babel gives us:
- VS Code Remote-SSH connection — full IDE, breakpoints, file tree,
  terminal.
- Persistent storage — activation caches and model weights survive
  across sessions.
- A100 80 GB (vs Colab's 40 GB) — comfortable for Llama 3.1 8B with
  any reasonable batch size.
- SLURM batch submission — fire-and-forget for long phases.
- Reproducibility — pinned conda env, lock files, deterministic runs.

The trade-off is SLURM queue waits, typically minutes but occasionally
hours during high demand. For our phases this is acceptable. If the
queue is consistently ≥ 2 hours, fall back to interactive jobs via
`srun --partition=general --gres=gpu:A100_80GB:1 --time=04:00:00 --pty bash`
for short phases.

---

## 1. The execution graph

```
[Phase 0: Local + Babel setup]    laptop + login node    ~30 min one-time
       │
       ▼
[Phase 1: Tokenizer audit]         login node, CPU       ~5 min
       │   gate: intersection ≥ 7000 single-token problems
       │  -> $BLACKBOX_DATA/tokenizer_audit/{model}.json
       ▼
[Phase 2: Accuracy reproduction]   A100 ×3 (parallel)    ~3 h total
       │   gate: empirical accuracy within 5pp of KT
       │  -> $BLACKBOX_DATA/correctness/{model}.parquet
       ▼
[Phase 3: Activation extraction]   A100 ×3 (parallel)    ~9 h total
       │   gate: activation cache complete for all 4 candidate layers
       │  -> $BLACKBOX_DATA/activations/{model}/layer_{ℓ}.npz
       ▼
[Phase 4a: REG diagnostics]        CPU                   ~30 min
       │   gate: σ̂_eff·κ̂_max ≤ 0.1 AND η̂ ≤ η_0 OR fall back to (L)-only
       │  -> $BLACKBOX_DATA/diagnostics/{model}/{layer}.json
       │
[Phase 4b: Manifold fit + select]  CPU                   ~30 min
       │   gate: held-out R² ≥ 0.9 OR switch to Diffusion Maps fallback
       │  -> $BLACKBOX_DATA/manifolds/{model}/{primary,secondary,...}.npz
       ▼
[Phase 5a: T_n statistics]         CPU                   ~30 min
       │   gate: at least one of (T_n^matched, T_n^V) significant
       │  -> $BLACKBOX_DATA/stats/{model}/{T_n_*, failure_modes}.json
       │
[Phase 5b: Localization]           CPU                   ~30 min
       │  -> $BLACKBOX_DATA/stats/{model}/T_n_V.json
       ▼
[Phase 6: Causal interventions]    A100 ×3 (parallel)    ~9 h total
       │   gate: all four ACE pre-registered criteria pass on at least 2 of 3 models
       │  -> $BLACKBOX_DATA/aces/{model}.json
       ▼
[Phase 7: Aggregation + figures]   CPU                   ~30 min
          -> $BLACKBOX_DATA/paper_figs/figure_{1..6}.pdf
```

**Total wall time: ~24 h compute + 3 h CPU.** With SLURM queue waits
plan ~3–5 days end-to-end. Phases 2, 3, 6 run in parallel across the
three models (1 SLURM job per model), so the per-model 3 hours dominates,
not 9 hours summed.

---

## 2. Storage paths

All paths assume `$USER = anshulk`. The repo defines `BLACKBOX_DATA` env
var defaulting to `/data/user_data/$USER/blackbox/`; export it
differently if your allocation lives elsewhere.

```
/data/user_data/$USER/blackbox/         # $BLACKBOX_DATA
├── tokenizer_audit/
│   ├── gpt-j-6b.json                  # cols: a, b, s, ok_a, ok_b, ok_s
│   ├── pythia-6.9b.json
│   ├── llama-3.1-8b.json
│   └── intersection.json              # set retained by all 3 tokenizers
├── correctness/
│   ├── gpt-j-6b.parquet               # cols: a, b, s, predicted, correct
│   ├── pythia-6.9b.parquet
│   └── llama-3.1-8b.parquet
├── activations/
│   ├── gpt-j-6b/layer_{07,14,21,27}.npz
│   ├── pythia-6.9b/layer_{08,16,24,31}.npz
│   └── llama-3.1-8b/layer_{08,16,24,31}.npz
│   (each npz contains: h (n × 4096) fp32, a, b, s, correct)
├── diagnostics/
│   └── {model}/{layer}.json           # τ̂, κ̂_max, σ̂_eff, η̂ per layer
├── manifolds/
│   └── {model}/
│       ├── M_hat_param.npz            # parametric helix, primary
│       ├── M_hat_dm.npz               # diffusion maps fallback
│       ├── M_hat_pca_means.npz        # PCA on class means
│       ├── M_hat_local_pca.npz        # local PCA
│       ├── M_hat_kpca.npz             # kernel PCA
│       ├── principal_angle_matrix.npz # 5x5 between methods
│       └── selection.json             # primary M̂ chosen + R² values
├── stats/
│   └── {model}/
│       ├── T_n_naive.json             # version 1 (biased baseline)
│       ├── T_n_cross.json             # version 2 (cross-fitted)
│       ├── T_n_matched.json           # version 3 (matched + cross-fit)
│       ├── T_n_V.json                 # localized for V_1..V_5
│       ├── T_curve_T_span.json        # 3-mode failure decomposition
│       └── failure_modes.json
├── aces/
│   └── {model}.json                   # ACE_N, ACE_NV, ACE_S, ACE_R + Prop. 4
├── paper_figs/
│   ├── figure_1.pdf  # 3 failure-mode schematic
│   ├── figure_2.pdf  # 5×5 principal-angle matrix per model
│   ├── figure_3.pdf  # T_n^naive vs T_n^cross vs T_n^matched table
│   ├── figure_4.pdf  # T_n^V_k localization across V_1..V_5
│   ├── figure_5.pdf  # 4 ACE bar chart with bootstrap CIs per model
│   ├── figure_6.pdf  # Proposition 4 predicted vs measured ACE_S
│   └── master_results_table.csv
└── logs/
    └── {phase}_{model}_{slurm_job_id}.{out,err}
```

**Quota check.** Activations dominate: 3 models × 4 layers ×
~10000 samples × 4096 dim × 4 bytes ≈ 2 GB. Total project footprint
including logs and figures: ~5 GB. Confirm with `du -sh /data/user_data/$USER/`
during Phase 0. Babel default user-data quota is typically 100 GB which
is comfortably above this.

**Repo location.** Clone the GitHub repo to `~/blackbox-nlp-2026/`:

```bash
cd ~ && git clone https://github.com/anshulk-cmu/blackbox-nlp-2026.git
cd ~/blackbox-nlp-2026
```

This is *separate from* `$BLACKBOX_DATA`. The repo holds code and docs;
the data dir holds outputs. The repo never holds activations or model
weights.

---

## 3. Phase 0: Local + Babel one-time setup

**Owner: Anshul. ~30 min one time.** All steps in this phase are
one-time; you do not redo them between runs.

### 3.1 Connect VS Code to Babel via Remote-SSH

1. Install the **Remote - SSH** extension in VS Code (publisher:
   Microsoft).
2. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and run
   **Remote-SSH: Connect to Host...** → **Add New SSH Host...**
3. Enter `ssh anshulk@babel.lti.cs.cmu.edu` (or the actual login host).
   Save to your `~/.ssh/config`.
4. Connect. VS Code installs a server-side helper, then opens a new
   window running on Babel.
5. **File → Open Folder** → `/home/anshulk/blackbox-nlp-2026` (after
   the clone in §3.2 below).

From this point on, you edit code and run terminals *inside* VS Code,
but everything executes on Babel. Breakpoint debugging works via VS
Code's Python debugger over Remote-SSH.

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

The repo includes [environment.yml](environment.yml) pinning every
dependency. The toy validation here is the same validation referenced
in `full_paper_plan.md §0a`; if it does not produce 45/45 PASS, stop
and fix the environment before continuing.

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
mkdir -p /data/user_data/$USER/blackbox/{tokenizer_audit,correctness,activations,diagnostics,manifolds,stats,aces,paper_figs,logs}
echo "export BLACKBOX_DATA=/data/user_data/$USER/blackbox" >> ~/.bashrc
source ~/.bashrc
```

### 3.6 Llama 3.1 license

Llama 3.1 requires accepting the community license on HuggingFace.
Already accepted on the same HF account per user. If you are running
on a fresh account, visit
`https://huggingface.co/meta-llama/Meta-Llama-3.1-8B` while logged in
and click "Accept license." Wait ~5 min for the accept to propagate.

### 3.7 Pre-flight checklist (must all pass before Phase 1)

- [ ] VS Code connected to Babel via Remote-SSH; can edit files and
      run terminals.
- [ ] Repo cloned at `~/blackbox-nlp-2026/`.
- [ ] Conda env `blackbox` exists; `python toy/run_toy.py` produces
      `45 PASS / 0 FAIL`.
- [ ] `huggingface-cli whoami` shows your HF username.
- [ ] `$BLACKBOX_DATA` is set; the directory tree exists.
- [ ] Llama 3.1 license accepted on the HF account.
- [ ] `groups` on Babel shows your storage allocation membership; quota
      check (`du -sh /data/user_data/$USER/`) shows < 50 GB used so we
      have room to grow.

---

## 4. Phase 1: Tokenizer audit

**Goal.** Determine which `(a, b)` pairs in `{0,...,99}²` produce
single-token operands `a`, single-token operands `b`, AND single-token
answer `s = a + b ∈ {0,...,198}` in each tokenizer; compute the
three-way intersection across models.

**Why this matters (math link).** `paper_math.md §1.1` defines
activations `h(a, b)` at the equals-sign token, with the correctness
label determined by greedy decoding of the *first* answer token.
`full_paper_plan.md §3.2` requires that the analysis be restricted to
single-token operands AND single-token answers in all three models so
that activations are directly comparable. If a tokenizer splits the
operand or the answer the geometric structure changes and the analysis
becomes apples-to-oranges.

**Why this matters (KT link).** KT 2025 §3 (Experimental setup)
restricts to operands in `[0, 99]` because all three of their models
tokenize 0 through 99 as single tokens. KT do not formally require the
answer `s = a + b` to be single-token (they just check it is single in
the evaluation), but for our geometric test the answer-token
representation matters, so we add this constraint.

**Runtime.** Login node, CPU-only. ~5 min total. No SLURM job needed.

```bash
cd ~/blackbox-nlp-2026
conda activate blackbox
python code/run_phase1_audit.py
```

The script `code/run_phase1_audit.py` should:
1. Load each tokenizer via `transformers.AutoTokenizer.from_pretrained`.
2. For each `a ∈ [0, 99]`, `b ∈ [0, 99]`, `s = a + b`, encode
   `str(a)`, `str(b)`, `str(s)` and check `len(tokens) == 1` for each.
3. Save a per-model JSON with cols `a, b, s, ok_a, ok_b, ok_s` indicating
   single-token status.
4. Compute the three-way intersection (problems where all three are
   single-token in all three models) and save to `intersection.json`.

**Llama 3.1 special handling.** Llama 3.1's tokenizer (BPE with merges
up to 999 in some ranges) may produce digit-by-digit splits for some
numbers. The script's docstring documents the relaxed protocol:
- Primary: require all three of `a`, `b`, `s` to be single-token. Drop
  the pair if any is multi-token.
- Fallback (Llama only, if primary keeps < 5000 pairs): require operands
  single-token but allow the answer to be multi-token, and treat the
  *first* answer token as the prediction in the correctness check
  (consistent with KT Llama prompt-template handling).

**Outputs.**
- `$BLACKBOX_DATA/tokenizer_audit/{model}.json` — one per model.
- `$BLACKBOX_DATA/tokenizer_audit/intersection.json`.

**Pre-registered gate.** Intersection ≥ 7000 pairs out of 10,000
ordered pairs.

**Expected results.** Based on KT 2025 §3 single-token ranges (GPT-J:
0–361, Pythia: 0–557, Llama: 0–999), the primary intersection should be
at most 9801 (the limit where `s ≤ 198 < 361 < 557 < 999`) reduced by
any tokenizer special-case quirks. Empirically expect:
- GPT-J: ~9800 retained (tiny attrition from BPE quirks).
- Pythia: ~9800 retained.
- Llama: 5000–9800 retained depending on whether digit-split kicks in.
- Three-way intersection: ~5000–9800 typical.

**Failure modes and fixes.**
- *Intersection < 7000 with Llama as bottleneck*: switch to relaxed Llama
  protocol described above; document this deviation in Appendix H of
  the paper.
- *Intersection < 7000 even with relaxation*: stop and inspect manually.
  This would mean the tokenizer behavior is fundamentally different
  from what KT reported. Open an issue and check tokenizer version
  pinning.
- *Tokenizer load fails with auth error*: re-run `huggingface-cli login`
  with a fresh token, then retry.

---

## 5. Phase 2: Accuracy reproduction (KT replication)

**Goal.** Reproduce KT 2025's published accuracies (80.5% / 77.2% / 98%)
on the intersection set. Confirms the prompt + tokenizer + decode
pipeline is set up correctly before the much more expensive activation
extraction in Phase 3.

**Why this matters (math link).** `paper_math.md §1.2` defines the
correctness label `y(a, b)` based on the *final-layer* greedy decode at
temperature 0 of the model. Phase 2 produces these labels; the
correctness label is the only feature distinguishing the correct and
wrong populations in the rest of the analysis.

**Why this matters (KT link).** KT 2025 Appendix Figure 11 reports
80.5% / 77.2% / 98% accuracies on 10,000 problems with the prompts:
- GPT-J / Pythia: `"Output ONLY a number. {a}+{b}="`.
- Llama: `"The following is a correct addition problem.\n{a}+{b}="`.

The prompt asymmetry is real and not justified in KT; we preserve it
exactly so our correctness numbers are directly comparable to theirs.
If we deviate from the KT prompts our accuracies will not match and we
will have a difficult-to-debug methodological discrepancy. This is also
one of the reviewer-defense points in `full_paper_plan.md §8.4`.

**Runtime.** A100 80 GB per model, bf16 forward-only. ~1 h each.
Submit all three SLURM jobs in parallel.

```bash
sbatch scripts/phase2_accuracy.sbatch gpt-j-6b
sbatch scripts/phase2_accuracy.sbatch pythia-6.9b
sbatch scripts/phase2_accuracy.sbatch llama-3.1-8b
```

The single sbatch script takes the model key as `$1`. SLURM template
in [scripts/phase2_accuracy.sbatch](scripts/phase2_accuracy.sbatch).
The Python script `code/run_phase2_accuracy.py` should:
1. Load the model in bf16 to A100.
2. For each `(a, b)` in the intersection set, format the prompt
   per-model, run a single forward pass with greedy decode for 1
   token (or up to 4 for Llama if multi-token answer mode is on),
   record the predicted answer and whether it matches `s = a + b`.
3. Save a parquet file with cols `a, b, s, predicted, correct`.
4. Print summary accuracy and per-bin accuracy stats.

**Outputs.** `$BLACKBOX_DATA/correctness/{model}.parquet`.

**Pre-registered gate.** Per-model `|empirical_accuracy - KT_reported|
≤ 5pp` (5 percentage points). Specifically:
- GPT-J: 75.5% ≤ acc ≤ 85.5%.
- Pythia: 72.2% ≤ acc ≤ 82.2%.
- Llama: 93% ≤ acc ≤ 100%.

**Expected results and what they mean.**
- All three within tolerance: pipeline reproduces KT, proceed to
  Phase 3.
- Llama < 93%: most likely cause is the multi-digit tokenization being
  handled wrong. Switch to the multi-token decode mode
  (`generate(max_new_tokens=4)` and string-compare against `str(s)`)
  and re-run.
- GPT-J or Pythia far off: likely a prompt typo or wrong tokenizer.
  Diff your prompt against KT Table 2 byte-for-byte.

**Failure modes and fixes.**
- *bf16 produces NaN logits on Llama*: the residual-stream activations
  in Llama 3.1 sometimes have outliers that overflow bf16. Cast logits
  to fp32 before argmax. This is well-known in the Llama community.
- *OOM on Llama batch=32*: Llama 3.1 8B in bf16 is ~16 GB plus
  ~2 GB activations per batch=32. Should fit in 80 GB easily; if it
  doesn't, drop to batch=16. If still OOM, reload `transformers` to
  the pinned version in `environment.yml`.
- *SLURM job preempted mid-run*: the script writes per-batch partials
  every 50 batches; resubmit and the script picks up from the last
  saved partial.

**Wrong-population sample-count math.** With the KT-published
accuracies and a 10,000-pair primary set, the expected wrong-sample
counts are:
- GPT-J: ~1950 wrong (80.5% accurate).
- Pythia: ~2280 wrong (77.2% accurate).
- Llama: ~200 wrong (98% accurate).

`full_paper_plan.md §3.1` notes that Llama's small `n_w ≈ 200` is below
the unprojected `T_n` detection threshold of ~340 derived from
`paper_math.md §6.6` (Theorem 6.6 composition bound), but is well above
the localized `T_n^V` threshold of ~10 for `r = 1, σ = 0.1, Δ = 1` from
`paper_math.md §5.5`. So Llama's headline test is the localized one.

---

## 6. Phase 3: Activation extraction

**Goal.** Cache residual-stream activations at the equals-sign token at
each candidate analysis layer per model:
- GPT-J (28 layers): candidates `{7, 14, 21, 27}` = `{L/4, L/2, 3L/4, L-1}`.
- Pythia (32 layers): candidates `{8, 16, 24, 31}`.
- Llama (32 layers): candidates `{8, 16, 24, 31}`.

The candidate layer set follows `full_paper_plan.md §3.6` and matches
KT 2025 §5.1's per-layer dynamics findings (KT identify layer 17 in
GPT-J as where the 9-parameter helix(a+b) beats 27-dim PCA, layers
14–18 as builder MLPs, 19–22 as the answer-helix plateau).

**Why this matters (math link).** `paper_math.md §1.1` defines `h^ℓ`
as the residual-stream activation at the equals-sign token at layer
`ℓ*` selected by Phase 4b. Phase 3 caches all four candidates for each
model so Phase 4b can pick the best held-out R² without re-running
forward passes.

**Why this matters (KT link).** KT 2025 §5.1 shows that the
last-token (equals-sign) residual stream is where the answer-helix
appears, peaking around layers 19–22 in GPT-J. We cache all four
candidates per model so we can verify the layer-selection rule
empirically rather than committing to one layer up front.

**Runtime.** A100 80 GB per model. ~3 h each. Submit all three SLURM
jobs in parallel.

```bash
sbatch scripts/phase3_extract.sbatch gpt-j-6b
sbatch scripts/phase3_extract.sbatch pythia-6.9b
sbatch scripts/phase3_extract.sbatch llama-3.1-8b
```

The Python script `code/run_phase3_extract.py` should:
1. Load the model in bf16 to A100.
2. Register HuggingFace `forward_hook`s at the four candidate layers'
   block outputs. Per-model HuggingFace paths
   (`full_paper_plan.md §3.9.6`):
   - GPT-J: `transformer.h.{ℓ}` (28 layers).
   - Pythia: `gpt_neox.layers.{ℓ}` (32 layers, parallel attention/MLP
     architecture; we patch at the *block input*, matching KT).
   - Llama: `model.layers.{ℓ}` (32 layers, sequential attention then MLP
     with RMSNorm; we patch at the block output before the next
     block's input).
3. For each `(a, b)` in the intersection set, run a forward pass and
   record the residual stream at the equals-sign token at each
   candidate layer.
4. After every 50 batches, flush partials to
   `$BLACKBOX_DATA/activations/{model}/layer_{ℓ}_partial_{batch}.npz`.
5. After all batches, concatenate partials into final
   `$BLACKBOX_DATA/activations/{model}/layer_{ℓ}.npz` files. Each
   contains `h: (n, 4096) fp32, a, b, s, correct`.

**Memory math.** Per-model footprint:
- Model bf16 weights: 12–16 GB.
- Per-token activations bf16 hooked: ~16 KB per sample per layer hook.
- Batch 32 with 4 hooks: ~2 MB extra per forward.
- Comfortable margin within 80 GB even at batch=64 if needed.

**Outputs.** Per-model, per-layer npz: ~10000 × 4096 × 4 bytes ≈ 160 MB.
Total per model: ~640 MB across 4 layers; total project: ~1.9 GB.

**Pre-registered gate.** All four candidate layers' npz files exist
and have row counts equal to the tokenizer-audit retained count from
Phase 1.

**Expected results and what they mean.**
- All four layers cache cleanly: proceed to Phase 4.
- Activations have NaNs: cast hook outputs to fp32 *inside* the hook
  before saving (the hook captures the bf16 tensor; we need to clone
  and `.float()` before stashing). Re-run with this fix.
- Llama RMSNorm rescaling concern: RMSNorm is a per-sample rescale, so
  a fixed V-direction injection still acts in the same direction
  post-norm. We measure the per-layer rescale factor and report it in
  Appendix J; it does not affect the geometric tests because we fit
  `M̂` on the same rescaled activations.

**Failure modes and fixes.**
- *SLURM time-limit hit before all batches done*: increase `--time` to
  `06:00:00` and resubmit; partials will be picked up.
- *Disk quota exceeded*: check `du -sh $BLACKBOX_DATA`; partials may
  have piled up. After successful concat, the per-batch partials can
  be deleted.
- *Per-layer npz row counts don't match across layers*: shouldn't
  happen because all four layers are hooked in the same forward pass,
  but if it does, restart the model and re-extract — likely a hook
  registration race.

---

## 7. Phase 4a: REG diagnostics (manifold regularity)

**Goal.** Empirically estimate the four diagnostic quantities required
by `paper_math.md` Assumption REG (§3.2) and Theorem 5.2 case (C)
(§5.2): reach `τ̂`, curvature `κ̂_max`, effective noise scale
`σ̂_eff`, and normal-frame drift `η̂`. Pre-registered as a gate before
the headline test in Phase 5.

**Why this matters (math link).** `paper_math.md §3.2` Remark 3.6
states: "We pre-register reporting `τ̂, κ̂_max, σ̂_eff, η̂` for each
(model, layer) pair before computing the test statistic, and treat the
test as conditional on the joint event that `σ̂_eff κ̂_max ≤ c_0` (the
small-noise regime) and `η̂ ≤ η_0`. Layers failing these diagnostics
are excluded from the analysis." Without this gate, Theorem 1's
linearization remainder `R_1 = O(σ² κ_max²)` (paper_math.md §4) is no
longer small and the test is biased.

**The four diagnostics, explained simply.**

1. *Reach `τ̂` (paper_math.md §1.3).* The reach `τ(M)` is the largest
   tube radius around `M` within which the closest-point projection
   `Π_M(h) = arg min_p ‖h - p‖` is unique. We need `τ(M)` to be
   bounded below by a positive constant `τ_min` for the residual
   `r(h) = h - Π_M(h)` to be well-defined. Estimator
   (Aamari–Levrard 2019):
   ```
   τ̂ ≥ min_{i ≠ j} ‖h_c^(i) - h_c^(j)‖² / (2 · d(h_c^(i) - h_c^(j), T_{p_j} M))
   ```
   over a sample of pairs `(i, j)`, where `T_{p_j} M` is estimated via
   local PCA at `p_j`. In plain terms: the distance from one correct
   activation to another, divided by twice the distance from the gap
   vector to the local tangent plane. If two close points project to
   the same place on `M`, the reach is small there.

2. *Curvature `κ̂_max`.* Operator norm of the second fundamental form
   of `M`, controlling how curved `M` is. For our parametric helix
   with `g(t) = u_0 + t u_lin + Σ_T (cos · u_cos^T + sin · u_sin^T)`,
   the closed-form bound is
   ```
   κ̂_max = max_t sup_{u ⊥ g'(t), ‖u‖=1} |⟨u, g''(t)⟩|
   ```
   evaluated at integer `t`. For the helix with periods `T ∈
   {2, 5, 10, 100}`, the highest-frequency component (T=2) dominates
   the curvature; we expect `κ̂_max ≈ 4π² / 4 ≈ 9.87` × max amplitude
   of T=2 component.

3. *Effective noise scale `σ̂_eff` (paper_math.md Theorem 4.10).* For
   anisotropic noise the relevant effective rank is
   ```
   k̂_eff = (tr Σ̂_r)² / tr(Σ̂_r²)
   σ̂_eff² = tr(Σ̂_r) / k̂_eff = tr(Σ̂_r²) / tr(Σ̂_r)
   ```
   where `Σ̂_r = (1/n_c) Σ r(h_c^(i)) r(h_c^(i))^T` is the empirical
   normal-bundle residual covariance. For isotropic noise
   `k̂_eff = k = d - dim(M)`. Real activations after layer
   normalization are anisotropic; we expect `k̂_eff ≪ k`.

4. *Normal-frame drift `η̂` (paper_math.md §5.2 case C).* For the
   curve case (as opposed to the linear span case), the normal space
   `N_p M` rotates as `p` moves along the curve, so the strict
   condition `V ⊆ ∩_p N_p M` of earlier drafts fails. Replaced by:
   ```
   η̂ = max_k ‖P_V (P^N_{p_k} - P^N_{p_0}) P_V‖_op
   ```
   over a sample of base points `{p_k}` along the curve, where `V` is
   the chosen localized subspace and `p_0` a reference point. We
   require `η̂ ≤ η_0` for some small constant; `paper_math.md §11
   item 5` flags that an analytic bound `η_0 = O(κ_max · diam(M))` is
   open. Empirically we need `η̂ ≤ 0.1` for the additional
   `O(η_0 D_max²)` and `O(η_0 r σ²)` terms in `E[T_n^V]` to be small.

**Runtime.** ~30 min, CPU only.

```bash
python code/run_phase4a_diagnostics.py
```

The script `code/run_phase4a_diagnostics.py` should, for each model and
each candidate layer:
1. Load the activations from Phase 3.
2. Restrict to correct-population activations only.
3. Compute the four diagnostics above.
4. Save `$BLACKBOX_DATA/diagnostics/{model}/{layer}.json` with all four
   values plus the `(σ̂_eff κ̂_max ≤ c_0, η̂ ≤ η_0)` decision.

**Outputs.** `$BLACKBOX_DATA/diagnostics/{model}/{layer}.json`.

**Pre-registered gate.** Per `(model, layer)` pair:
- `σ̂_eff · κ̂_max ≤ 0.1`: small-noise regime.
- `η̂ ≤ 0.1`: drift-controlled (curve case only; trivially zero for
  linear-span case (L) since `P^N_p = I - U_S U_S^T` is constant in
  `p`).

A `(model, layer)` pair failing either gate is *excluded* from
Phase 5/6 analysis. Pairs passing both proceed.

**Expected results and what they mean.**
- All four candidate layers pass for all three models: ideal; Phase 4b
  picks the best one by held-out R².
- Some layers fail, others pass: typical; we use only the passing
  layers.
- All layers fail for one model: the test's small-noise regime is
  violated for that model. Two responses: (1) report the model as
  "diagnostic-failed" in the appendix and proceed only with the other
  two; (2) relax the gate empirically and report the additional bias
  term explicitly. Decision is pre-registered in `full_paper_plan.md
  §6.4`: at least 2 of 3 models must pass for a positive result; if
  fewer pass, fall back to negative-result publication.

**Failure modes and fixes.**
- *`τ̂` is small (< 0.1)*: indicates near-self-intersection of the
  manifold or noisy local-PCA tangent estimates. Sub-sample more
  aggressively, raise the local-PCA neighborhood size, and recompute.
- *`κ̂_max` is huge (> 100)*: the parametric helix derivative is
  blowing up at some integer. Check for tokenizer artifacts (a single
  outlier integer's activation differs sharply). Drop that integer
  from the helix fit and recompute.
- *`η̂` is large (> 0.5)*: the normal-frame is drifting heavily on the
  curve. Either fall back to the linear-span case (L), where
  `M = M_S` and `η = 0` automatically, or pick a different reference
  point `p_0`. Linear-span fallback is the practical default and is
  what most of the analysis runs on.

---

## 8. Phase 4b: Manifold fit and layer selection

**Goal.** For each model, fit five candidate manifold estimators
(parametric helix, PCA on class means, local PCA, Diffusion Maps, Kernel
PCA) at every passing candidate layer, compute held-out R² and pairwise
principal angles, select the primary `M̂` per the pre-registered rule.

**Why this matters (math link).** `paper_math.md §6` (Theorem 6.2)
gives the parametric estimator's finite-sample concentration:
```
sin θ_max(M̂_param, M) ≤ C_3 σ √(κ^B) / (√(λ_min^B) σ_K(C*)) · √(d log(d/δ)/n_c)
```
with the misspecification radius `b(M)` from Theorem 6.5 added when
`E[H_c | B] = B(C*)^T` is not exact. The composed bound from
Theorem 6.6 then propagates this error through the test statistic.
Diffusion Maps and Kernel PCA do not have a clean composed rate
(`paper_math.md §10`); they serve as cross-method validation.

**Why this matters (KT link).** KT 2025 §4 fits the parametric helix
with `B(a) = [a, cos(2π a/T), sin(2π a/T) for T ∈ {2,5,10,100}]`,
9 columns. We adopt the same basis with one critical correction from
`paper_math.md` Remark 2.2 and `full_paper_plan.md §0a.1`: on integer
inputs the `sin(2π·a/2) = sin(π a) = 0` column is identically zero, so
the basis has only `K = 8` non-degenerate functions. KT note this
fragility in their Figure 12 but keep `T = 2` because downstream
neurons read at period 2; we drop the degenerate `sin(πa)` column from
the OLS design matrix to keep `B` full-rank, but retain the `cos(πa)`
column.

**Why this matters (plan link).** `full_paper_plan.md §3.5` requires
fitting three manifold *parameterizations* (`M_S^answer`, `M_S^union`,
`M_S^joint`) and selecting the simplest one with R² ≥ 0.9. For the
BlackboxNLP submission the headline test uses `M_S^answer` (the
last-token answer-helix); the other two are appendix robustness checks.

**Runtime.** ~30 min, CPU only.

```bash
python code/run_phase4b_manifolds.py
```

The script should, for each model and each *passing* candidate layer
from Phase 4a:
1. Load the correct-population activations.
2. Mean-center: `H_c ← H_c - mean(H_c)` (paper_math.md Remark 2.4 makes
   the affine `M_S` linear after centering).
3. Fit five `M̂` methods listed below, with a held-out 20% split for R²
   computation.
4. Compute the 5×5 pairwise principal-angle matrix.
5. Save per-method `M̂` plus the principal-angle matrix and a selection
   JSON.

**The five methods, summarized.**

1. *Parametric helix (primary, `paper_math.md §6.1`).* Build the
   integer-only basis `B ∈ R^{n_c × 8}` with columns `[a, cos(πa),
   cos(2π a/5), sin(2π a/5), cos(2π a/10), sin(2π a/10),
   cos(2π a/100), sin(2π a/100)]`. Solve OLS `Ĉ = (B^T B)^{-1} B^T H_c`,
   `M̂_param = QR(Ĉ^T)[Q]`. Held-out R² is computed on the 20% fold.

2. *PCA on class means (cautionary baseline, `paper_math.md §10`).* Bin
   `a` into `K_bins = 100` bins (single-integer bins, since we have
   100 distinct integers 0–99), compute the per-bin mean activation,
   SVD the matrix of bin-means minus grand mean. Take the top-`m`
   left singular vectors as `M̂_pca-means`. The toy validation
   (`full_paper_plan.md §0a`) caught a critical bug: `K_bins = 20`
   over a 199-point range gives bin width 10 = T=10 period, which
   cancels the T=10 helix component entirely; we use `K_bins ≥ 100`
   throughout.

3. *Local PCA (negative result, `paper_math.md §10`, Singer–Wu 2012).*
   At each `h_c^(i)`, find `k = 50` nearest neighbors in `H_c`, do
   local SVD on the centered neighborhood, extract top-`m` directions.
   Average the projection matrices `P_T(h_i) P_T(h_i)^T` and
   eigendecompose. Expected to fail on the high-curvature helix
   (curvature × bandwidth too large for Singer–Wu's small-curvature
   assumption).

4. *Diffusion Maps + regression lift (`paper_math.md §10`,
   Coifman–Lafon 2006).* Build affinity matrix `K_{ij} = exp(-‖h_i -
   h_j‖² / (2 σ_DM²))` with median-bandwidth `σ_DM`, row-normalize to
   `P`, eigendecompose. Take top non-trivial eigenvectors as
   embedding `Φ ∈ R^{n × m}`. Lift back via regression: solve
   `H_c ≈ Φ A` for `A ∈ R^{m × d_m}`. `M̂_dm` is the column span of
   `A^T`.

5. *Kernel PCA + regression lift (`paper_math.md §10`,
   Schölkopf–Smola–Müller 1998).* Build double-centered RBF kernel,
   eigendecompose. Same regression-lift as method 4. Expected to
   agree with method 4 to within 0.5° in our setting.

**The three manifold parameterizations.** All five methods are run for
each of:
- `M_S^answer` (1D parameterization in `s = a + b`, dim 8 after
  centering): primary. Last-token activation regressed against the
  answer label.
- `M_S^union` (joint operand-answer, dim up to 24): operand `a`
  helix, operand `b` helix, and answer `s = a + b` helix concatenated;
  retain unique directions via QR.
- `M_S^joint` (full arithmetic state, dim up to 30+): basis includes
  carry indicator and operand-decile dummies in addition to the helix.

**Selection rule (pre-registered in `full_paper_plan.md §3.5–§3.6`).**
1. For each candidate layer that passed Phase 4a's REG gate, compute
   held-out R² of `M̂_param` against `M_S^answer`.
2. Choose the primary layer `ℓ*` as the one with the highest R².
3. If `R² ≥ 0.9` at `ℓ*`, primary `M̂ = M̂_param` for `M_S^answer`.
4. Else if `R² ≥ 0.9` for `M_S^union`, switch primary to that.
5. Else if `R² ≥ 0.9` for `M_S^joint`, switch primary to that.
6. Else if no parameterization reaches R² = 0.9, fall back to
   Diffusion Maps as the primary method (likely for Llama 3.1 8B per
   KT Figure 23, where the last-token helix fit is markedly weaker).

**Outputs.** Per-model:
- `M_hat_param.npz, M_hat_dm.npz, M_hat_pca_means.npz,
  M_hat_local_pca.npz, M_hat_kpca.npz`.
- `principal_angle_matrix.npz` (5×5).
- `selection.json` with primary method, primary layer, R² values,
  fallback flags.

**Pre-registered gate.** At least one parameterization reaches
R² ≥ 0.9 OR the Diffusion Maps fallback is invoked.

**Expected results.**
- GPT-J: `M_S^answer` parametric R² ≥ 0.95 at layer 21 (KT Figure 5
  plateau).
- Pythia: similar to GPT-J at layer 24 or 25.
- Llama: parametric R² may be ~0.7–0.85 (KT Figure 23 weaker fit).
  Likely scenario: `M_S^union` or Diffusion Maps fallback engaged.

**Pre-registered findings to record.** Per
`full_paper_plan.md §3.5` and `§4.3`, we expect methods 1, 4, 5 to
agree to within 5° on real LLM data; method 2 to be 5–10° off due to
binning bias; method 3 to either agree on small `n` or break on large
`n` due to curvature.

**Failure modes and fixes.**
- *Parametric R² < 0.9 on all three layers for some model*: fallback
  to Diffusion Maps as primary; this is the pre-registered Llama
  scenario.
- *Diffusion Maps σ_DM too small or too large*: median bandwidth is
  the default; if R² is low, sweep σ_DM ∈ {0.5, 1, 2, 4} × median and
  pick the best.
- *Local PCA θ_max > 80°*: confirm the prediction in the paper, no
  action — this is reported as a negative result per
  `full_paper_plan.md §4.5`.
- *Principal-angle matrix is asymmetric*: bug in the angle
  computation; principal angles must be symmetric. Use Björck-Golub
  1973 SVD-of-cross-Gram.

---

## 9. Phase 5a: Test statistics computation

**Goal.** For each model at the chosen layer (and its primary `M̂`):
compute three versions of `T_n` (naive / cross-fitted / matched +
cross-fitted), plus the curve-vs-span decomposition `(T_curve, T_span,
T_within-span)` for the three failure modes.

**Why this matters (math link).** Three theorems power this phase:
- *Theorem 4.2 (validity, paper_math.md §4.2).* Under Assumptions GM
  + BD + REG, `E[T_n] = ‖μ_ξ‖² + tr(Σ_ξ) + R_1(σ, κ_max)` with
  `R_1 = O(σ² κ_max²)` vanishing for linear `M`. Concentration
  bound: for any `u > 0`,
  `P[T_n - E[T_n] ≥ 2σ²√(2ku/n) + 2σ²u/n] ≤ 2e^{-u}`.
- *Theorem 4.10 (anisotropic, paper_math.md §4.6).* For anisotropic
  `Σ_ε`, the variance term is `2 k_eff σ_eff⁴` and the linear-tail
  term carries `σ_op²` per Hanson–Wright.
- *Theorem 8.3 (matched permutation, paper_math.md §8).* The
  bin-restricted permutation distribution gives exact conditional
  Type I control under the pooled exchangeability hypothesis
  `H_0^cond: h ⊥ y | φ(a, b)`.

**Why this matters (plan link).** `full_paper_plan.md §3.7` requires
reporting all three versions transparently:
- *V1 naive*: `M̂` fit on all correct samples, residuals computed on
  the same correct samples (the correct-residual baseline is
  artificially small → biased upward).
- *V2 cross-fitted*: K=5-fold cross-fitting (`paper_math.md §7.1`).
  The correct-residual baseline is unbiased; this is Neyman-orthogonal
  in the sense of Chernozhukov 2018, with the explicit Gateaux-
  derivative verification in `paper_math.md §7.3`.
- *V3 matched + cross-fitted (the headline test)*: V2 plus the
  bin-stratified permutation null from `paper_math.md §8` and
  `full_paper_plan.md §3.3`. This addresses the difficulty confound
  (wrong samples are systematically harder than correct).

**Why this matters (failure-mode link).** `paper_math.md §2.3` and
`full_paper_plan.md §1.5` define three failure modes via the
curve/span Pythagorean decomposition `‖r_C(h)‖² = ‖r_S(h)‖² +
‖r_within(h)‖²` (Lemma 2.7):
- On-curve, in-span: both `T_curve` and `T_span` low. Position error,
  not geometric error.
- Off-curve, in-span: `T_curve` high, `T_span` low. Drift along the
  span.
- Off-span: both high. Geometric structure has broken (the Llama
  scenario per KT Figure 23).

**Runtime.** ~30 min, CPU only.

```bash
python code/run_phase5a_stats.py
```

The script should, for each model:
1. Load the primary `M̂` and the correct/wrong activations from the
   chosen layer.
2. Compute `T_n^naive`: single-fit `M̂` on all correct, residuals on
   all. Bootstrap 1000 samples for CI.
3. Compute `T_n^cross`: 5-fold cross-fitting per
   `paper_math.md Definition 7.1`. Bootstrap 1000 samples for CI.
4. Define the bin function `φ(a, b) = (sum_bin, carry_pattern,
   a_decile, b_decile, answer_token_class)` per
   `full_paper_plan.md §3.3` (4 × 4 × 10 × 10 × 2 = 3200 nominal
   cells, ~200–300 non-empty empirically).
5. Compute `T_n^matched`: per-bin within-bin permutation null over
   1000 shuffles, weighted by `min(n_c^bin, n_w^bin)`. Use only
   bins with `n_w^bin ≥ 2 AND n_c^bin ≥ 2`.
6. Compute `T_curve` and `T_span` separately per
   `full_paper_plan.md §1.5`. For `T_curve`, project onto the
   1D helix curve via the integer-search
   `â(h) = arg min_a ‖h - g(a)‖`.
7. Save all results.

**Outputs.** `$BLACKBOX_DATA/stats/{model}/`:
- `T_n_naive.json` — value, bootstrap CI, p-value vs unconditional
  null.
- `T_n_cross.json` — value, bootstrap CI, p-value vs cross-fitted
  null.
- `T_n_matched.json` — value, matched null distribution, p-value.
- `T_curve_T_span.json` — both decomposition values.
- `failure_modes.json` — primary failure-mode classification per the
  table in `full_paper_plan.md §1.5`.

**Pre-registered gate.** At least one of `T_n^matched > 0` or
`T_n^V > 0` (Phase 5b) is significant at `α = 0.05` after BH-FDR
correction.

**Expected results and what they mean.**

GPT-J / Pythia (KT-friendly models per KT Figure 5):
- `T_n^naive`: large positive (biased upward, 0.5–2.0 in normalized
  units).
- `T_n^cross`: moderate positive (0.2–1.0); confirms cross-fit
  removes ~50% of the naive bias.
- `T_n^matched`: smaller but still positive (0.1–0.5); confirms the
  effect is not purely difficulty confounding.
- Failure mode: most likely "off-curve, in-span" — wrong activations
  are at non-integer helix positions but still in the span.

Llama 3.1 8B (KT Figure 23 weaker fit):
- All three `T_n` versions positive, but smaller in absolute terms.
- Failure mode most likely "off-span" — Llama's gated MLPs implement
  algorithms outside the helix subspace, so wrong activations leave
  the span entirely. This is the headline empirical finding if it
  holds.

**If V3 (matched) is null while V1 (naive) is large.** Per
`full_paper_plan.md §3.3`: report this honestly as "the unconditional
geometric difference between correct and wrong populations is largely
explained by difficulty distribution and overfitting bias rather than
intrinsic geometric divergence." Still publishable as a methodology
paper with negative-result framing.

**Failure modes and fixes.**
- *Permutation distribution has too few non-trivial bins (< 50)*: fall
  back to the coarser scheme `(sum_bin, carry_pattern,
  answer_token_class)` (~32 cells). Pre-registered in
  `full_paper_plan.md §3.3`.
- *`T_n^matched` is significantly *negative**: indicates correct
  activations have *larger* residuals than wrong ones within bins.
  Most likely cause: wrong samples disproportionately in low-noise
  bins. Sanity-check `σ̂_eff` per bin; if heteroskedasticity is the
  cause, switch to studentized statistic.
- *Bootstrap CI fails to converge*: increase bootstrap to 5000;
  if still unstable, the residual distribution is heavy-tailed —
  switch to permutation-based CI.

---

## 10. Phase 5b: Localization across pre-registered subspaces

**Goal.** For each model at the chosen layer: compute `T_n^V_k` for the
five pre-registered failure subspaces V_1..V_5, apply BH-FDR correction
across them, identify the top-V (the one with largest significant
`T_n^V`).

**Why this matters (math link).** `paper_math.md §5` (Theorem 5.2) is
the headline localization theorem:
- *(a) Achievability*: `n ≥ C_1 · r σ⁴/Δ⁴ · log(1/β)` for power
  `1 - β`. For `r = 1, σ = 0.1, Δ = 1`, this is ~10 samples — easily
  achievable even on Llama's small wrong population.
- *(b) Two-point Le Cam lower bound (proven)*: at rate `σ²/Δ²`. The
  matching minimax rate `r σ⁴/Δ⁴` (b') is conjectural in this
  submission.
- *(c) Exact null distribution*: scaled-chi-squared with rate
  `O(1/√(nr))` Berry–Esseen approximation
  (`paper_math.md` Remark 4.11; the rate *improves* with `r`).

The case (L) / (C) split (`paper_math.md §5.2`) determines which
projection logic applies:
- *(L) Linear span*: `M = M_S` is linear, `N_p M = M_S^⊥` is constant
  in `p`. The condition `V ⊆ ∩_p N_p M = M_S^⊥` is automatic.
- *(C) Curve case*: `V ⊆ N_{p_0} M` at a reference point `p_0`, with
  drift bound `η̂ ≤ η_0` from Phase 4a. Conclusions hold with an
  extra `O(η_0 D_max²)` term in mean and `O(η_0 r σ²)` in variance.

**Why this matters (plan link).** `full_paper_plan.md §3.8` defines
the five subspaces:
- *V_1 (carry)*: LDA direction separating `a + b ≥ 100` (carry) from
  `a + b < 100`. `r_1 = 1`.
- *V_2 (fragile T=2)*: cos/sin pair for period T=2 from KT's basis.
  `r_2 = 2`. Per KT Figure 12 this is the fragile component.
- *V_3 (higher-order Fourier)*: cos/sin pairs for periods
  T ∈ {3, 4, 6, 7, 8} that KT did *not* include. `r_3 = 10`. Tests
  whether wrong activations have structure outside KT's basis.
- *V_4 (random-direction baseline)*: `r = 1` random unit vector
  orthogonal to `M̂`. Repeated 100 times to estimate the baseline
  distribution.
- *V_5 (full orthogonal complement of M̂)*: `r_5 = d_m - dim(M̂) ≈
  4087`. The "no-localization" baseline equivalent to `T_n` itself.

**Adaptive V via Romano–Wolf.** If we want to claim "V_k is the
specific failure subspace" and V_k was selected by data, the joint
Type I error over V_1..V_5 is controlled via Romano–Wolf step-down
with studentized maxT (`paper_math.md` Remark 5.7). We pre-register the
5 V's so this is not a post-hoc selection.

**Runtime.** ~30 min, CPU only.

```bash
python code/run_phase5b_localization.py
```

The script should, for each model:
1. Identify V_1..V_5 from correct samples only (using the 4-way
   data split per `full_paper_plan.md §3.4`: identify on Split B,
   evaluate on Split D).
2. For each V_k, project residuals via `P_V` and compute `T_n^V_k`.
3. Bootstrap 1000 CI per V_k, compute permutation p-value.
4. Apply BH-FDR at `q = 0.05` across the 5 V's.
5. Identify the top-V (largest significant `T_n^V`).

**Outputs.** `$BLACKBOX_DATA/stats/{model}/T_n_V.json` with per-V
results, BH-corrected significance, and top-V identification.

**Pre-registered gate.** At least one V_k significant after
BH-correction.

**Expected results and what they mean.**

GPT-J / Pythia (KT-aligned):
- V_1 (carry): probably significant; KT Figure 5 shows MLPs 19–22 do
  the answer-helix readout, errors should localize to the carry
  mechanism.
- V_2 (T=2): possibly significant but small effect (KT Figure 12: T=2
  is fragile, so wrong cases may have larger T=2 perturbation than
  correct).
- V_3 (higher Fourier): probably null; outside KT's basis.
- V_4 (random): null by construction.
- V_5 (full): equals `T_n` from Phase 5a.

Llama 3.1 8B:
- More uncertain; KT do not localize the gap. Expected: V_3 (higher
  Fourier) significant, indicating Llama uses Fourier components KT
  did not find.

**Failure modes and fixes.**
- *No V_k significant after BH*: the failure is geometrically diffuse;
  publishable as "wrong activations are off-manifold but not localized"
  per `full_paper_plan.md §11.1`.
- *V_4 (random) is significant*: the test is not properly calibrated
  against random subspaces. Re-run V_4 with more random draws (1000
  instead of 100) and verify the calibration distribution is
  zero-centered.
- *V_5 doesn't equal `T_n`*: bug in the projection. V_5 should give
  the unprojected statistic exactly; if it doesn't, check `P_V_5 = I -
  U_M̂ U_M̂^T`.

---

## 11. Phase 6: Causal interventions (4 ACEs + Proposition 4)

**Goal.** Run all four interventions (Necessity, Localized Necessity,
Sufficiency, Random-baseline Specificity) and the Proposition 4
quantitative check. This is the load-bearing causal claim of the paper.

**Why this matters (math link).** `paper_math.md §9` (Proposition 9.3)
is the causal-sufficiency proposition with second-order Hessian
remainder:
- The patch displacement is `Δh_c = U_V μ̂_ξ - P_V h_c` (the patch
  *replaces* the V-component, so the displacement subtracts off the
  original V-component; this corrects an earlier draft that omitted
  the `-P_V h_c` term).
- Hessian remainder: `|ACE_S - ACE_S^linear| ≤ (L/2) E‖Δh_c‖²` with
  `L` the operator-norm bound on `∇² LD`.
- Linear prediction:
  `ACE_S^linear = μ̂_ξ^T U_V^T E[∇LD(h_c)] - E[h_c^T P_V ∇LD(h_c)]`.
  Under linear-span GM with V ⊥ M_S, the second term vanishes in
  expectation, leaving
  `ACE_S^linear = μ̂_ξ^T U_V^T E[∇LD(h_c)]`.
- Magnitude lower bound under the *Alignment Assumption* (ALN, eq 9.3):
  `|ACE_S^linear| ≥ c_0 · ‖μ̂_ξ‖ · ‖U_V^T E[∇LD]‖` for `c_0 = 0.3`.
  Cauchy-Schwarz alone gives an *equality* with `|cos θ|`, not a lower
  bound; the ALN converts this to an actual lower bound.

**Why this matters (plan link).** `full_paper_plan.md §3.9` defines:
- *N (necessity)*: project wrong activations onto `M̂` (zero out off-
  manifold residual). `Patch(h_w, M̂^⊥, 1, 0) = Π_{M̂}(h_w)`.
- *NV (localized necessity)*: zero out only the V-component:
  `Patch(h_w, V, 1, 0)`.
- *S (sufficiency)*: inject `μ̂_ξ` into correct activations:
  `Patch(h_c, V, 1, μ̂_ξ)`.
- *R (random-baseline specificity)*: `Patch(h_c, V_random, 1,
  δ_random)` with random V of matching dim and δ of matching norm.
  Repeated 100 times.

**Why this matters (KT link).** KT 2025 §5 (the entire Clock-algorithm
section) is built on activation patching. KT patch *correct* activations
to verify the helix-fit hypothesis (Figure 4, 5). Our use of patching
is symmetric — we patch *wrong* activations to measure causal
necessity and *correct* activations to measure causal sufficiency. The
Patching protocol (KT Section 5, Appendix C) is preserved exactly:
1. Pick a clean prompt and a corrupted prompt.
2. Run the model on the clean prompt; save residuals.
3. Run on the corrupted prompt; save residuals.
4. Re-run corrupted, but at layer ℓ_m at the equals-sign token,
   overwrite the residual with `Patch(h_corrupted, V, α, δ)`.
5. Measure the change in `LD` (downstream forward-pass logit-difference
   per `paper_math.md` Definition 9.2).

**Logit-difference as a downstream forward-pass functional.** For
`h ∈ R^{d_m}` representing a candidate activation at the analysis
layer `ℓ_m` on a problem `(a, b)` with answer token `τ(s)`:
```
LD(h; a, b) = logit^final_{τ(s)}(forward_{ℓ_m → L}(h; a, b))
            - max_{t ≠ τ(s)} logit^final_t(forward_{ℓ_m → L}(h; a, b))
```
where `forward_{ℓ_m → L}(h; a, b)` is the model's downstream forward
pass holding the rest of the network fixed (paper_math.md Def. 9.2;
Conmy et al. 2023). LD is well-defined for any `h`, not only the
natural activation.

**Smooth-max regularization for argmax flips.** `LD` is non-smooth at
points where the argmax over `t ≠ s` flips, and the Hessian bound `L`
is unbounded there. Replace `max` with log-sum-exp at temperature `τ`:
`LD_τ(h) = logit_s(h) - τ log Σ_{t ≠ s} exp(logit_t(h)/τ)`. We apply
Proposition 9.3 to `LD_τ` and report sensitivity to `τ` in the
appendix. Default `τ = 0.5`.

**Runtime.** A100 80 GB per model. ~3 h each. Submit all three SLURM
jobs in parallel.

```bash
sbatch scripts/phase6_causal.sbatch gpt-j-6b
sbatch scripts/phase6_causal.sbatch pythia-6.9b
sbatch scripts/phase6_causal.sbatch llama-3.1-8b
```

The Python script `code/run_phase6_causal.py` should, for each model:
1. Load the model in bf16; identify the analysis layer and primary M̂
   from Phase 4b; identify the top-V from Phase 5b.
2. Re-register HuggingFace `forward_pre_hook` on the *next* layer to
   capture and modify the residual stream output of `ℓ_m`.
3. Estimate `μ̂_ξ` from cross-fitted wrong-population residuals
   projected onto top-V (avoiding double-dipping per
   `full_paper_plan.md §3.9.3`).
4. Run intervention N: for each wrong sample, `h_w → Π_{M̂}(h_w)`,
   compute `LD_τ(Π_{M̂}(h_w)) - LD_τ(h_w)`. Average → ACE_N.
5. Run NV: for each wrong sample, `h_w → Patch(h_w, V, 1, 0)`,
   compute LD difference. Average → ACE_NV.
6. Run S: for each correct sample, `h_c → Patch(h_c, V, 1, μ̂_ξ)`,
   compute LD difference. Average → ACE_S.
7. Run R: for each correct sample, repeat S with V replaced by 100
   random subspaces of matching dim and δ_random of matching norm.
   Per-replicate ACE_R.
8. Bootstrap 1000 CI for each.
9. Compute Proposition 4's predicted `ACE_S^linear` via PyTorch
   autograd of LD with respect to h, and the alignment cosine
   `cos θ_{ξ,∇} = ⟨μ̂_ξ, U_V^T E[∇LD]⟩ / (‖μ̂_ξ‖ ‖U_V^T E[∇LD]‖)`.
10. Save all to `$BLACKBOX_DATA/aces/{model}.json`.

**Outputs.** Per model:
```json
{
  "ACE_N":   {"mean": ..., "ci_low": ..., "ci_high": ..., "p": ...},
  "ACE_NV":  {"mean": ..., "ci_low": ..., "ci_high": ..., "p": ...},
  "ACE_S":   {"mean": ..., "ci_low": ..., "ci_high": ..., "p": ...},
  "ACE_R":   {"mean": ..., "ci_low": ..., "ci_high": ..., "p": ...},
  "ACE_S_predicted":  ..., "ACE_S_measured": ..., "ratio": ...,
  "cos_theta_xi_grad": ..., "ALN_pass": ...,
  "L_hessian_estimate": ..., "tau_smooth_max": 0.5,
  "top_V": "V_1"
}
```

**Pre-registered ACE criteria** (`full_paper_plan.md §3.9.4`,
BH-corrected at `q = 0.05` across all four):
1. **ACE_N significantly positive** — necessity of the off-manifold
   component.
2. **ACE_NV significantly positive AND ≥ 0.7 × ACE_N** — the
   localized component captures most of necessity.
3. **ACE_S significantly negative** — sufficiency of injecting `μ̂_ξ`
   to break correct behavior.
4. **ACE_NV significantly exceeds ACE_R** — specificity to V, not a
   generic perturbation effect.

A positive causal result requires **all four** on at least 2 of 3
models. Partial positives are reported with hedging per
`full_paper_plan.md §6.4`.

**Expected results.**
- GPT-J / Pythia: all four likely pass.
- Llama 3.1 8B: ACE_N likely passes (geometric divergence is causal),
  ACE_NV less certain (depends on whether the top-V actually carries
  most of the signal; Llama may have diffuse off-span structure).

**Proposition 4 prediction.**
- `ACE_S^predicted = μ̂_ξ^T U_V^T E[∇LD]` per the linear-span
  simplification.
- Ratio `|ACE_S_measured| / |ACE_S^linear|` ∈ [0.2, 5.0] is consistent
  with the Hessian-bound prediction (the toy validation found
  `ratio ≈ 5×` in one calibrated case per `full_paper_plan.md §0a`).
- ALN pass: `|cos θ_{ξ,∇}| ≥ 0.3`. If ALN fails, the magnitude lower
  bound is uninformative; report Proposition 4 as "qualitatively
  consistent but quantitative bound inconclusive."

**Failure modes and fixes.**
- *PyTorch autograd of LD blows up (Hessian L → ∞)*: argmax flip near
  the patched activation. Increase `τ` from 0.5 to 1.0 in the smooth
  max; report sensitivity. Per `paper_math.md §11 item 9` this is an
  open question.
- *ACE_NV < 0.7 × ACE_N*: the top-V does *not* capture most of
  necessity. Report as "off-manifold drift is not subspace-localized
  in V_k for k=1..3"; consider trying additional V's in a follow-up.
- *ACE_R has fat tails (some random subspaces give large effects)*:
  the random-direction baseline is mis-calibrated. Sub-sample more
  random subspaces (1000 instead of 100); use the empirical 95th
  percentile rather than mean as the comparator.
- *RMSNorm scaling concern on Llama*: each block has RMSNorm on the
  input, so a fixed V-direction injection is rescaled by a per-sample
  factor. We measure this factor and report it; the V-direction is
  preserved, just rescaled. ACE_NV should not be affected at the
  qualitative level.
- *Out of memory on Llama 3.1 patching with autograd enabled*: turn
  off autograd outside the gradient computation step
  (`torch.no_grad()` for the ACE evaluation, then a separate pass with
  gradient enabled for Proposition 4 prediction).

**Compute budget.** Per model with `n_c ≈ 4500` correct and
`n_w ≈ 500` wrong, four interventions per sample:
- N: 500 wrong × 1 forward = 500.
- NV: 500.
- S: 4500.
- R: 4500 × 100 random subspaces; sub-sample to 4500 × 10 = 45000 if
  full is too expensive, with smaller CIs to compensate.
- Total ~10000–55000 forward passes per model. At A100 batch 32, ~15
  minutes to 3 hours per model. Comfortable within 4-hour SLURM time
  limit.

---

## 12. Phase 7: Aggregation and figures

**Goal.** Build the six paper figures plus the master results table for
the BlackboxNLP submission.

**Runtime.** ~30 min, CPU only.

```bash
python code/run_phase7_figures.py
```

The script generates:
- *Figure 1 (intro schematic)*: cartoon of the three failure modes
  (on-curve, off-curve in-span, off-span). Hand-drawn or matplotlib.
- *Figure 2 (5×5 principal-angle matrix per model)*: heatmap of
  pairwise principal angles between the five `M̂` methods. Shows
  parametric / DM / KPCA agree to within 5°.
- *Figure 3 (three-version `T_n` table)*: T_n^naive / T_n^cross /
  T_n^matched per model, with bootstrap CIs.
- *Figure 4 (T_n^V localization)*: per-V, per-model `T_n^V` with
  BH-corrected significance flags.
- *Figure 5 (4 ACE bar chart)*: ACE_N, ACE_NV, ACE_S, ACE_R per model
  with bootstrap CIs.
- *Figure 6 (Proposition 4)*: predicted vs measured ACE_S scatter
  across models, with the ALN cosine annotated.
- *Master results table* (`master_results_table.csv`): one row per
  model, columns: best layer, primary method, R², T_n^matched, top-V,
  ACE_N, ACE_NV, ACE_S, ACE_R, ACE_S_predicted, cos θ_{ξ,∇},
  positive-result flag.

**Outputs.** `$BLACKBOX_DATA/paper_figs/figure_{1..6}.pdf`,
`master_results_table.csv`.

**Pre-registered headline numbers.** Per `full_paper_plan.md §6.4`:
positive result = at least 2 of 3 models pass all four ACE criteria
AND have significant `T_n^matched`.

**What success looks like.**
- All three models show the same qualitative pattern: significant
  off-manifold drift, localized to a specific V, causally necessary
  AND sufficient.
- Llama-specific finding: the top-V is *different* from GPT-J/Pythia,
  and the failure mode is "off-span" rather than "off-curve in-span."

**What partial success looks like.**
- 2 of 3 models pass: still publishable, with discussion of the
  asymmetry. Most likely scenario: GPT-J + Pythia pass, Llama partial.
- 1 of 3 models pass: report as a model-specific finding rather than
  a general claim.

**What failure looks like.**
- 0 of 3 models show significant `T_n^matched`: publishable as "the
  helix is sufficient on these models for these errors; failures are
  on-manifold position errors." Per `full_paper_plan.md §6.4`, this
  is the graceful fallback.

---

## 13. SLURM script template

All Phase-2/3/6 SLURM scripts share the same template. Concrete files
in [scripts/](scripts/). Skeleton:

```bash
#!/bin/bash
#SBATCH --job-name=blackbox_{phase}_{model}
#SBATCH --partition=general
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

Per-phase variants override `--gres`, `--time`, and the python script
name. Phase 3 (extraction) uses `--time=04:00:00`; Phase 6 (causal)
uses `--time=06:00:00` to allow buffer for the 100-replicate ACE_R.

**Submitting all three models in parallel.** A small bash helper:

```bash
#!/bin/bash
# scripts/submit_all_models.sh PHASE
PHASE=$1
for MODEL in gpt-j-6b pythia-6.9b llama-3.1-8b; do
  sbatch scripts/phase${PHASE}_*.sbatch $MODEL
done
```

Usage: `./scripts/submit_all_models.sh 2` submits Phase 2 for all
three models.

**Monitoring.** `squeue -u $USER` to see queue; `tail -f
$BLACKBOX_DATA/logs/*_${SLURM_JOB_ID}.out` to watch the run live.

---

## 14. Failure recovery and resumability

Every phase that runs > 5 minutes follows the same resumability
pattern:

1. **Drive flush every 50 batches.** Partial outputs go to
   `*_partial_{batch_idx}.npz`; final concat reads all partials.
2. **Resume by checking output first.** Each script starts with
   ```python
   if Path(out).exists():
       print(f'cached: {out}, skipping')
       return
   ```
   so re-running is idempotent.
3. **SLURM preemption.** If a job is preempted, just resubmit; the
   script picks up from the last cached partial.
4. **Cache verification.** A `code/verify_cache.py` utility script
   reports the state of every expected output file across all phases,
   to diagnose half-completed pipelines after preemption storms.

Specific resume scenarios:
- *Phase 3 preemption mid-extraction*: partials at
  `activations/{model}/layer_{ℓ}_partial_{batch}.npz` are picked up;
  resubmit the same SLURM job.
- *Phase 4b parameter sweep partially done*: each method's M̂ is
  cached separately; resubmit and only the missing methods run.
- *Phase 6 ACE_R partially done (one of 100 random subspaces)*:
  individual random-subspace results are cached as
  `aces/{model}_random_{i}.json` and concatenated at the end.

---

## 15. VS Code workflow

The recommended development cycle:

1. **Edit code locally** via VS Code Remote-SSH file editor on Babel.
2. **Commit and push** from the local clone (or via VS Code's git
   integration on Babel directly).
3. **In the Babel terminal (inside VS Code Remote)**: `git pull` and
   run the phase script.
4. **Browse outputs** via the VS Code Remote file tree at
   `/data/user_data/$USER/blackbox/`.
5. **For interactive Python**: open a `.py` file in VS Code, click
   "Run Python File in Terminal" — it executes on Babel with the
   conda env.
6. **For figures**: VS Code can preview PDFs inline; click the figure
   in the Remote file tree.

Breakpoint debugging works via VS Code's Python debugger over Remote-
SSH; especially useful for Phase 6 causal interventions where the
forward-hook logic is intricate.

---

## 16. Pre-registration acceptance criteria summary

This single subsection summarizes every pre-registered gate so a
reviewer can audit the protocol from one place.

**Phase 1 gate.** Three-way tokenizer intersection ≥ 7000 pairs.
Failure → switch Llama to relaxed multi-token-answer protocol.

**Phase 2 gate.** Per-model accuracy within 5pp of KT 2025:
- GPT-J ∈ [75.5%, 85.5%].
- Pythia ∈ [72.2%, 82.2%].
- Llama ∈ [93%, 100%].

**Phase 4a gate.** Per (model, layer): `σ̂_eff · κ̂_max ≤ 0.1` AND
`η̂ ≤ 0.1`. Failing layers are excluded from analysis. At least one
candidate layer per model must pass.

**Phase 4b gate.** Held-out R² ≥ 0.9 for at least one of `M_S^answer`,
`M_S^union`, `M_S^joint` on the parametric estimator, OR fall back to
Diffusion Maps as primary (pre-registered for Llama if needed).

**Phase 5a gate (test validity).** At least one of `T_n^matched > 0`
or `T_n^V > 0` (5b) significant at α = 0.05 after BH-FDR.

**Phase 5b gate (localization).** At least one V_k significant after
BH-FDR at q = 0.05. V_4 (random) must be null (calibration check).

**Phase 6 gate (causal).** All four ACE criteria pass on at least 2 of
3 models:
1. ACE_N significantly positive.
2. ACE_NV significantly positive AND ≥ 0.7 × ACE_N.
3. ACE_S significantly negative.
4. ACE_NV significantly exceeds ACE_R.

**Phase 6 quantitative check (Proposition 4).** Ratio
`|ACE_S_measured| / |ACE_S^linear| ∈ [0.2, 5.0]` AND
`|cos θ_{ξ,∇}| ≥ 0.3` (Alignment Assumption).

**Headline positive result.** All five gates pass on at least 2 of 3
models; partial positives are reported with hedging per
`full_paper_plan.md §6.4`.

---

## 17. Known issues and common fallbacks

### Issue 1: Babel queue is full
*Symptom*: `sbatch` reports queue depth > 50 jobs.
*Fix*: Use the priority partition if your account has access (`-p
priority`); otherwise use shorter time limits to qualify for the
short-job queue (`-p short --time=01:00:00`). For long phases this is
not an option; just wait.

### Issue 2: HuggingFace download throttling
*Symptom*: First model load on each new node downloads weights and
hits HF rate limit.
*Fix*: Pre-stage the weights to Babel scratch:
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('EleutherAI/gpt-j-6B', cache_dir='/data/user_data/$USER/blackbox/hf_cache')"
```
Then in subsequent runs, set `HF_HOME=/data/user_data/$USER/blackbox/hf_cache`.

### Issue 3: bf16 NaN cascades on Llama
*Symptom*: Logits contain NaN; correctness label is undefined.
*Fix*: Cast to fp32 at the unembedding step:
```python
logits = (residual.float() @ unembed.weight.float().T)
```
This is well-known in the Llama community.

### Issue 4: Diffusion Maps σ_DM is wrong scale
*Symptom*: DM embedding has ~all eigenvalues equal (kernel too narrow)
or one dominant eigenvalue (kernel too wide).
*Fix*: Sweep σ_DM ∈ {median, 2×median, 0.5×median, 4×median} of
pairwise distances; pick the one with cleanest spectral gap.

### Issue 5: Local PCA neighborhoods cross the helix bottleneck
*Symptom*: Local PCA θ_max > 80° catastrophically.
*Fix*: This is the predicted negative result per
`full_paper_plan.md §4.5`. No fix; report as expected.

### Issue 6: Permutation null has too few non-trivial bins
*Symptom*: Phase 5a `T_n^matched` p-value is undefined or always 1.
*Fix*: Drop to the coarser bin scheme `(sum_bin, carry_pattern,
answer_token_class)` (~32 cells). Pre-registered fallback in
`full_paper_plan.md §3.3`.

### Issue 7: ACE_R has fat tails
*Symptom*: A few random subspaces give effects comparable to or
larger than ACE_NV.
*Fix*: Increase to 1000 random subspaces for tighter percentile
estimation. Use the empirical 95th percentile rather than the mean
of |ACE_R| as the comparator with ACE_NV.

### Issue 8: Smooth-max temperature τ is wrong
*Symptom*: PyTorch autograd produces NaN gradients or huge `L`
estimates near argmax flips.
*Fix*: Increase τ from 0.5 to 1.0 or 2.0; report sensitivity per
`paper_math.md §11 item 9`.

### Issue 9: Cross-fitting variance estimate is unstable
*Symptom*: K-fold variance estimate has CI[0] < 0 < CI[1] when the
mean is clearly nonzero.
*Fix*: Increase K from 5 to 10. If still unstable, switch to the
4-way data split protocol (`full_paper_plan.md §3.4`) which separates
fit / identify-V / calibrate / test more aggressively.

---

## 18. Glossary and cross-reference

| Term / symbol | Defined in | Plain meaning |
|---|---|---|
| `M` | paper_math.md §1.3 | The true manifold of correct activations |
| `M_S` | paper_math.md §2.2 | The helix span (linear), dim 8 (integer) or 9 (continuous) |
| `M_C` | paper_math.md §2.1 | The helix curve (1D), `{g(t) : t ∈ [0, A]}` |
| `M̂` | full_paper_plan.md §1.1 | Empirical estimator of `M` from correct samples |
| `r(h)` | paper_math.md §1.4 | Off-manifold residual `h - Π_M(h)` |
| `T_n` | paper_math.md §4.1 | Mean residual-norm² difference, wrong - correct |
| `T_n^V` | paper_math.md §5.1 | Localized version, residuals projected to V |
| `T_curve` / `T_span` | full_paper_plan.md §1.5 | Curve and span versions |
| `μ_ξ`, `Σ_ξ` | paper_math.md §3 | Mean and covariance of the wrong-population perturbation |
| `σ`, `Σ_ε` | paper_math.md §3 | Within-population noise scale and covariance |
| `k` | paper_math.md §4.1 | `d - dim(M)`, residual-bundle dimension |
| `k_eff`, `σ_eff²` | paper_math.md §4.6 | Effective rank and scale for anisotropic noise |
| `κ_max`, `τ_min` | paper_math.md §3.2 | Curvature and reach bounds (Assumption REG) |
| `λ_min^B`, `κ^B` | paper_math.md §6.2 | Design-matrix conditioning |
| `σ_K(C*)` | paper_math.md §6.2 | Smallest singular value of the population coefficients |
| `D_max`, `D_max^ctr` | paper_math.md §3.1, §6.7 | Activation norm bound, uncentered and centered |
| `η`, `η_0` | paper_math.md §5.2 | Normal-frame drift constant (curve case) |
| `V_1..V_5` | full_paper_plan.md §3.8 | Pre-registered failure subspaces |
| `φ(a, b)` | paper_math.md §8.1 | Coarse bin function for matched permutation |
| `LD(h)` | paper_math.md §9.2 | Logit-difference (downstream forward-pass functional) |
| `Patch(h, V, α, δ)` | paper_math.md §9.1, full_paper_plan.md §3.9.1 | Patching operation |
| `ACE_N, ACE_NV, ACE_S, ACE_R` | full_paper_plan.md §3.9.4 | The four causal effects |
| ALN | paper_math.md §9.3 (eq 9.3) | Alignment assumption `\|cos θ_{ξ,∇}\| ≥ c_0` |
| `R_1` | paper_math.md §4.2 | Linearization remainder, `O(σ² κ_max²)` |

**Phase ↔ math/plan crosswalk.**

| Phase | Math support | Plan support | KT support |
|---|---|---|---|
| 1 (audit) | §1.2 (correctness label) | §3.2 (single-token rule) | §3 (KT operand range) |
| 2 (accuracy) | §1.2 (`y(a, b)`) | §3.2 (KT prompts) | §3, App. Fig 11 (KT acc) |
| 3 (extract) | §1.1 (h^ℓ definition) | §3.6 (layer set), §3.9.6 (HF paths) | §5.1 (last-token analysis) |
| 4a (REG diagnostics) | §3.2 (REG), Remark 3.6 (estimators) | §1.4 (REG paragraph) | — |
| 4b (manifolds) | §6 (Theorem 6.2/6.5/6.6) | §3.5–§3.6, §4 (5-method comparison) | §4 (helix fitting) |
| 5a (T_n stats) | §4 (Theorems 4.2, 4.10), §7 (cross-fit) | §3.3, §3.4, §3.7 | — |
| 5a (failure modes) | §2.3 (Pythagorean decomp) | §1.5 (3 modes) | §5.1 + Fig 23 (Llama gap) |
| 5a (matched perm) | §8 (Theorem 8.3) | §3.3 (matched perm) | — |
| 5b (localization) | §5 (Theorem 5.2) | §3.8 (V_1..V_5) | Fig 12 (T=2 fragility) |
| 6 (causal) | §9 (Proposition 9.3) | §3.9 (4 ACEs) | §5 (Clock algorithm patching) |
| 7 (figures) | — | §9.5 (figure plan) | — |

---

## 19. Final pre-flight checklist before kickoff

When you sit down to actually run the pipeline end-to-end, verify
this list line by line:

- [ ] Phase 0 setup signed off (§3.7 above).
- [ ] Toy validation passes 45/45 (`python toy/run_toy.py`).
- [ ] Babel allocation confirmed (`groups`, quota check).
- [ ] HuggingFace login active (`huggingface-cli whoami`).
- [ ] All three model weights pre-staged or downloadable
      (`huggingface-cli download EleutherAI/gpt-j-6B` etc.).
- [ ] `$BLACKBOX_DATA` directory tree exists.
- [ ] Code repository at `~/blackbox-nlp-2026/` clean
      (`git status` clean, on the right branch).
- [ ] All Phase X SLURM scripts present in `scripts/`
      (`ls scripts/phase*.sbatch` shows phases 2, 3, 6).
- [ ] All `code/run_phaseN_*.py` scripts present (`ls code/run_phase*.py`).

Once all boxes are ticked, kick off Phase 1 with the single command:

```bash
python code/run_phase1_audit.py
```

and follow the phase-by-phase recipes above.

---

## 20. Documentation completeness

### Math (paper_math.md): complete (45 / 45 toy PASS validates).

### Methodology (full_paper_plan.md): complete (revised 2026-05-06 to
align with paper_math.md across all theorems).

### Toy (toy/README.md): complete; runs locally or on Babel CPU node
in ~4 min.

### Operations (this document): complete (rewritten 2026-05-06 with
full integration of paper_math.md theorems, full_paper_plan.md
methodology, and KT_paper.md prior-work references).

### Outstanding before Phase 1:
- [ ] Pre-flight checklist (§19) signed off.
- [ ] Babel allocation confirmed.

---

*End of Babel execution plan.*
