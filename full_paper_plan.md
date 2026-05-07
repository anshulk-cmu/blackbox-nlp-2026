# Off-Manifold Failure: A Geometric Test for Localized Computational Errors in Language Models

## Complete End-to-End Paper Plan

**Status.** Locked methodological plan. Every mathematical step written out
in detail. Three models throughout: GPT-J 6B, Pythia 6.9B, Llama 3.1 8B,
matching Kantamneni and Tegmark 2025.

**Target.** BlackboxNLP 2026 archival track, 8 pages plus references plus
optional appendix (camera-ready allows +1 page → 9 total). Co-located with
EMNLP 2026 in Budapest, Hungary, October 28th, 2026.

**Submission window from today (May 3, 2026).**
- **July 17, 2026** — direct paper submission deadline (OpenReview, AoE).
- **August 28, 2026** — ARR pre-reviewed paper commitment deadline (only
  useful if we have ARR reviews in hand by then; we do not, so this is
  not our target).
- **September 8, 2026** — notification of acceptance.
- **September 20, 2026** — camera-ready due.
- **October 28, 2026** — workshop date.

That gives us ~10.5 weeks (75 days) from today to the direct-submission
deadline. The plan below is methodological and does not move with the
calendar; the timeline is reproduced here for reference only. A separate
execution schedule will be tracked outside this document.

**Companion paper.** Anshul-Prasad EMNLP 2026 main on multi-digit
multiplication (separate, not previewed here).

**Authoritative math reference.** All theorem statements, assumptions, and
proof outlines in this plan are reconciled with [paper_math.md](paper_math.md)
(the standalone mathematical proposal). When the plan and paper_math.md
disagree, paper_math.md wins. Cross-references in this plan point to
sections of paper_math.md by their section number there
(e.g., "paper_math.md §4" for the validity theorem).

**Dual submission.** The 2026 BlackboxNLP CFP allows dual submissions to
the archival track ("please check the dual submissions policy for the
other venue that you are dual-submitting to"). This is a relaxation
relative to BlackboxNLP 2025, which prohibited EMNLP+BlackboxNLP dual
submissions outright. Practically: our two papers (multiplication for
EMNLP main, addition for BlackboxNLP) do not overlap on task, model
analysis, or theoretical contribution, so dual-submission risk was
already low; this update removes residual concern. We will still verify
EMNLP 2026's outbound policy before submission.

**Special track consideration.** The 2026 CFP includes a special track
on "Reproducibility and Reliability in Interpretability Analyses"
(6 pages max) for papers reproducing established results with rigorous
statistical evaluation. Our paper partially fits this — we replicate KT
on three models with cross-fitting, conditional null calibration, and
random-subspace controls. We are NOT submitting to the special track:
our scope exceeds reproduction (new theory, new failure-mode framework,
new causal pipeline). The 8-page archival track is the right fit. The
methodological rigor we adopt from special-track standards (random
baselines, effect-size reporting, generalization across model
configurations) strengthens the archival submission anyway.

---

## 0a. Synthetic-toy validation status (added 2026-05-04)

Before running on GPT-J / Pythia / Llama on the CMU Babel cluster, we
validated the entire pipeline against synthetic ground truth in
[toy/](toy/). The toy plants a known helix with known perturbation
directions in `R^{256}`, runs every analysis this paper claims to run,
and prints a pre-registered PASS/FAIL line per metric. **Final result:
45 PASS / 0 FAIL in 244 seconds on a CPU after the 2026-05-06
paper_math.md restructuring** (was 27/0 before E9–E16 were added; see
[toy/README.md](toy/README.md) for the full breakdown). Every component
of the paper's pipeline behaves as predicted on data that satisfies
Assumption GM exactly.

**Execution environment.** All real-model runs happen on **CMU Babel**,
operated through VS Code Remote-SSH for full IDE integration (file
browser, breakpoints, terminals). SLURM submits batch jobs for GPU phases;
non-GPU phases run on the login node or interactively. The full
step-by-step execution plan with SLURM scripts and resumable cache layout
is [babel_execution_plan.md](babel_execution_plan.md). An earlier draft
attempted Colab Pro but we pivoted back to Babel for IDE integration,
80 GB A100s (vs Colab's 40 GB), persistent storage, and SLURM's
fire-and-forget batch model.

**What passed numerically.**
- Theorem 1: `T_n` reads `0.012` under H_0 (predicted 0), `0.283` under
  mean-shift `‖μ_ξ‖² = 0.25` (within 0.05), `1.224` under variance-only
  `tr(Σ_ξ) = 1.25` (within 0.10). Empirical `1/√n` concentration verified
  (slope -0.62 vs theory -0.5; small-n bias absorbs the difference).
- Theorem 2: 18/18 power-curve cells (r ∈ {1,2,5}, Δ ∈ {0.3,1,3}, n ∈
  {100,500,2000}) reach empirical power ≥ 0.8.
- Theorem 3: parametric helix-basis OLS recovers `M_S` to `sin θ_max = 0.041`
  at `n_c = 2000`, scales `1/√n` cleanly. PCA-on-bins saturates at 0.111
  (bias-limited, as predicted). Diffusion Maps and Kernel PCA agree to within
  0.005 (cross-validation OK).
- Three failure modes (curve vs span): `(T_curve, T_span)` signatures match
  the table at lines 248-252 — on-curve gives `(0.004, 0.007)`, off-curve
  in-span gives `(0.81, -0.002)`, off-span gives `(0.98, 0.98)`.
- Cross-fitting (§3.4): naive `T_n` shows `+0.092` upward bias under H_0;
  K-fold cross-fitted `T_n` is `-0.0004` (CI contains 0). Cross-fit fix works.
- Matched permutation (§3.3): naive permutation Type I rate 100% under
  difficulty confound; matched permutation Type I rate 12% (vs nominal 5%,
  within tolerance). Matched fix works.
- Localization (§3.8): planted V_1 has per-dim effect 770× larger than random
  V_4, 247× larger than full-complement V_5. Pre-dim effect-size diagnostic
  cleanly separates signal from spillover.
- Causal pipeline (§3.9): ACE_N = +1.566, ACE_NV/ACE_N = 1.000, ACE_S =
  -1.555, ACE_R = +0.009 ± 0.013, ACE_NV - ACE_R = 1.557 (all four
  pre-registered criteria satisfied).
- Proposition 4 magnitude lower bound: `|ACE_S|/predicted = 5.35`. Bound
  holds with substantial slack.
- **PyTorch autograd matches the closed-form gradient to floating-point
  precision (1e-15).** This validates the gradient-computation infrastructure
  for the real-model causal pipeline.

**Bugs the toy caught (would have surfaced only on Babel otherwise):**
1. **T=2 fragility makes the design matrix rank-deficient** at integer inputs.
   `sin(2π·s/2) = sin(πs) = 0` identically on integer s, so the basis is
   effectively 8-D not 9-D. KT note this in their Figure 12; we missed it
   in the original plan. **Fix applied below: redefine `HELIX_DIM = 8`
   throughout.**
2. Linear-axis convention mismatch between data-generation and OLS basis.
   `helix(s, C)` and `basis_vector(s)` had divergent linear-coordinate
   conventions; refactored to share code.
3. `(n, n, d_m)` broadcast in pairwise-distance computation — at the paper's
   target `d_m = 4096, n_c = 4500` this would be ~660 GB of memory. Use
   `‖a-b‖² = ‖a‖² + ‖b‖² - 2 a·b` to compute distances in O(n²) memory.
4. `K_bins = 20` over a 199-point range gives bin width 10 = T=10 period
   exactly, cancelling the T=10 helix component entirely. Use `K_bins ≥ 100`.
5. Confound on the wrong axis defeats matched permutation. Confounds must
   live at the bin-marginal level, not the within-bin level, for matched
   permutation to control Type I rate.
6. Five Section-2 math ambiguities surfaced by careful audit (typical-p,
   ξ ∈ V vs μ_ξ ∈ V, Σ_ξ|_V notation, composition boundedness, missing √k).
   **Fixes applied directly to Sections 2.1, 2.3, 2.5, 2.7 below.**

**What the toy does NOT validate** (and therefore remains as risk for the
real-model run):
- Real activations have anisotropic, possibly non-Gaussian noise.
- Llama 3.1 8B's last-token helix fit is markedly weaker (KT Figure 23) —
  the toy plants a clean helix and gets clean recovery.
- Layer selection: the toy operates at one synthetic "layer."
- Real causal-pipeline plumbing: HuggingFace forward hooks on transformer
  blocks, lm_head logit-difference computation, resumable activation cache
  on Babel scratch
  (`/data/user_data/$USER/blackbox/activations/{model}/layer_{ℓ}.npz`).

**Practical implication.** The methodology is locked. The Babel run is now
a *systems* validation step, not a *math* validation step. The remaining
risk is whether the real-model assumptions hold, not whether the analysis
code does the right thing given clean inputs. The step-by-step Babel
execution plan with SLURM scripts, conda env, and resume logic is in
[babel_execution_plan.md](babel_execution_plan.md).

---

## 0b. The contribution in one paragraph

We introduce a localized geometric diagnostic for mechanistic failure modes
in language-model arithmetic. Given a candidate computational manifold
recovered from correct cases and a hypothesized failure subspace, the
diagnostic identifies whether model errors correspond to *structured
off-manifold drift* in the hypothesized subspace rather than on-manifold
mispositioning. We build on Kantamneni and Tegmark 2025 (arXiv:2502.00873),
who establish that GPT-J 6B, Pythia 6.9B, and Llama 3.1 8B encode integers
on a generalized helix and use a Clock-style algorithm for two-digit
addition. Their Figure 23 shows the algorithm claim transfers cleanly to
GPT-J and Pythia but only weakly to Llama 3.1 8B; they do not investigate
the gap. They also study only correct cases. Our paper formalizes the
distinction between three failure modes (on-curve in-span, off-curve
in-span, off-span), proves a finite-sample concentration bound for the
test under sample-estimated manifolds (Theorem 3), proves explicit power
against pre-registered directional alternatives (Theorem 2), and *plans to
test* (post-results: "establishes") a quantitative connection to
behavioral causal effects via a four-intervention pipeline (necessity,
localized necessity, sufficiency, specificity). The methodology is robust
to two key threats: input-difficulty confounding (addressed by stratified
matched permutation) and overfitting bias (addressed by K-fold
cross-fitting). To our knowledge this is the first application of
cross-fitted manifold-residual testing with conditional null calibration
to mechanistic localization in mid-sized open-weight transformer language
models. The output is a model-agnostic, theoretically grounded
diagnostic, with causal validation built into the design, that any
interpretability researcher can plug their own manifold model into.

*Note for the plan-to-paper transition:* the paragraph above is written
in the cautious, pre-results form. Phrases like "we plan to test"
become "we test" once experiments are run and "we demonstrate that"
only when ACE thresholds in Section 3.9.4 are met. Section 13 carries
both versions.

---

## 1. Setting up the problem

### 1.1 Notation we will use throughout

Throughout the paper, we use the following symbols. Reviewers should be able
to look up any symbol once and not lose track.

- `m`: model identifier, `m ∈ {GPT-J 6B, Pythia 6.9B, Llama 3.1 8B}`.
- `d_m`: residual-stream width for model `m`. `d_GPT-J = 4096`, `d_Pythia = 4096`,
  `d_Llama = 4096`. (All three happen to be 4096; we keep `d_m` notation to avoid
  conflating across models.)
- `L_m`: number of transformer blocks. `L_GPT-J = 28`, `L_Pythia = 32`, `L_Llama = 32`.
- `ℓ ∈ {0, 1, ..., L_m}`: layer index, where `ℓ = 0` is the embedding output and
  `ℓ = L_m` is the final residual stream before the unembedding.
- `h_m^ℓ ∈ R^{d_m}`: residual-stream activation at layer `ℓ` for model `m` at
  the last input token (the `=` sign).
- `(a, b)`: an addition problem with operands `a, b ∈ {0, 1, ..., 99}` and
  ground-truth answer `a + b ∈ {0, 1, ..., 198}`.
- `H_c, H_w`: matrices of stacked residual-stream activations from correct and
  wrong populations respectively. `H_c ∈ R^{n_c × d_m}` and `H_w ∈ R^{n_w × d_m}`.
- `M ⊂ R^{d_m}`: the *true* manifold of correct-population activations
  (unknown; recovered from data).
- `M̂`: an estimator of `M`, computed from `H_c` only.
- `ξ ∈ R^{d_m}`: a structured perturbation vector representing the difference
  between the wrong population's mean shape and `M`.
- `r(h) = h - π_M(h)`: the projection residual, where `π_M` is the orthogonal
  projection onto `M`.
- `T_n`: the test statistic.
- `T_n^V`: the localized test statistic with respect to a candidate failure
  subspace `V`.
- `θ_max(M̂, M)`: the maximum principal angle between two subspaces (Björck-Golub
  1973).
- `σ`: within-population noise standard deviation (per coordinate, sub-Gaussian).

### 1.2 The motivating question, restated

KT show that integers embed on a helical curve in residual-stream space.
Concretely, for the integer token `a ∈ {0, 1, ..., 99}` (the operand range
used throughout this paper, matching the 10,000-ordered-pair primary
dataset of Section 3.2), the activation `h^ℓ(a)` is well-approximated by

```
h^ℓ(a) ≈ a · u_lin + Σ_{T ∈ {2, 5, 10, 100}} [cos(2π a / T) · u_cos^T + sin(2π a / T) · u_sin^T] + noise
```

with KT's nine directions `u_lin, u_cos^2, u_sin^2, ..., u_cos^100, u_sin^100`
(one linear, four cosines, four sines = 9 vectors in `R^{d_m}`) constituting
a 9-parameter helix. The image of this fit is a smooth 1-dimensional manifold
(the helix curve) sitting inside an *up-to* 9-dimensional ambient subspace.
For *integer-only* data (the empirical setting), the basis function
`sin(2π · a / 2) = sin(π a) = 0` vanishes identically on integer `a`, so
the identifiable basis has `K = 8` non-degenerate functions and the
identifiable span dimension is `m = 8` (paper_math.md Remark 2.2; KT
Figure 12). We retain the 9-vector notation when discussing the ideal
continuous-input model, but use `m = 8` and `K = 8` in the empirical
pipeline throughout.

*Equivalence with KT's matrix form.* Equation (1) above can be written
compactly as `helix(a) = C · B(a)^T`, matching KT Equation 2, where

```
C  =  [u_lin | u_cos^2 | u_sin^2 | u_cos^5 | u_sin^5 | u_cos^10 | u_sin^10 | u_cos^100 | u_sin^100]  ∈ R^{d_m × 9}
B(a)  =  [a, cos(2π a / 2), sin(2π a / 2), cos(2π a / 5), sin(2π a / 5),
            cos(2π a / 10), sin(2π a / 10), cos(2π a / 100), sin(2π a / 100)]  ∈ R^9.
```

The columns of `C` are exactly our `u_*` direction vectors; `B(a)` is the
9-vector of basis functions evaluated at `a`. We use the sum-of-named-vectors
form in this paper because it makes the curve-vs-span decomposition of
Section 1.5 transparent (each `u_*` is one direction in `M_S`), but every
formula carries over to KT's matrix form by reading off the columns of `C`.

*Range footnote.* The answer helix `M_S^answer` (Section 3.5) re-uses the
same basis with `a` replaced by the sum `s = a + b ∈ {0, ..., 198}`; this is
the only place the wider 0–198 range appears. KT's still-wider 0–360 range
([KT Section 4.1, "Why 0 to 360"](KT_paper.md)) was a Fourier-discovery
choice (360 has many integer divisors, easing period detection) and is not
used for any fit, projection, or test in this paper.

KT then study only inputs where the model produces the correct sum. They report,
for each layer of each model, the helix's fit quality on correct samples. Their
Figure 23 documents that on Llama 3.1 8B the *last-token* helix fit is markedly
worse than on GPT-J or Pythia, suggesting the algorithm uses the helix for
operand encoding but performs the addition via additional Llama-specific
mechanism. They do not investigate further.

We ask: if we collect activations from problems where the model produces a
*wrong* answer, do those activations sit on the same helix as the correct ones?

Two structurally different scenarios are possible.

- **On-manifold position errors.** Wrong activations live on `M` but at a
  different location. The helix's geometry is intact; the model's internal
  pointer just landed at the wrong number.
- **Off-manifold drift.** Wrong activations live in a region that the manifold
  does not visit. The geometry has broken in a specific direction.

Distinguishing these is the question this paper answers. The interpretability
implication differs sharply between them. On-manifold errors suggest the
algorithm is correct but the inputs to it are imprecise; off-manifold drift
suggests the algorithm itself has structurally departed.

### 1.3 The two manifold-residual quantities we track

Given a candidate manifold `M ⊂ R^{d_m}` recovered from correct samples, we
define for any activation `h ∈ R^{d_m}`:

- **The on-manifold projection.** `π_M(h) = argmin_{p ∈ M} ‖h - p‖_2`.
  When `M` is a linear subspace with orthonormal basis `U ∈ R^{d_m × k}`, this
  reduces to the linear projection `π_M(h) = U U^T h`. When `M` is the helix
  curve itself, `π_M(h)` is the closest helix point, which we approximate by
  finding the integer parameter value
  `â(h) = argmin_{a ∈ ℤ ∩ [0, 99]} ‖h - h^ℓ_helix(a)‖_2`
  (the search is over integers in the operand range from Section 1.2, with
  `[0, 198]` substituted when projecting onto the answer helix; this discrete
  search is what makes `M_C` a 1-dimensional submanifold rather than a
  full `m`-dimensional subspace, with `m = 8` for integer data per
  paper_math.md Remark 2.2) and setting `π_M(h) = h^ℓ_helix(â(h))`.
- **The projection residual.** `r(h) = h - π_M(h)`.

The whole paper depends on these two quantities behaving statistically the way
we expect under correct- and wrong-population samples.

### 1.4 The generative model assumption (stated and defended)

For the theorems to hold, we assume three structural conditions: GM (the
generative model itself), BD (uniform boundedness of activations), and REG
(manifold regularity controlling the linearization error). We state each
precisely now and defend them in Section 5 (Limitations). The formal
versions live in [paper_math.md §3](paper_math.md), Assumptions 3.1, 3.4,
3.5; the plan version below is a reader-friendly restatement.

**Assumption GM (generative model).** For each population `p ∈ {c, w}`,
each activation `h ∈ H_p` is generated as

```
h = m_p(a, b) + ε_p
```

where `m_p` is a deterministic mapping from problem `(a, b)` to a point in
`R^{d_m}`, and `ε_p` is a zero-mean sub-Gaussian noise vector with covariance
`Σ_ε` and Orlicz `ψ_2`-norm at most `σ` (i.e., `‖⟨u, ε⟩‖_{ψ_2} ≤ σ ‖u‖_2`
for every unit `u ∈ R^{d_m}`). Furthermore:

- `m_c(a, b) ∈ M` for all `(a, b)` (correct samples lie on the manifold).
- `m_w(a, b) = m_c(a, b) + ξ(a, b)` for some perturbation `ξ(a, b) ∈ N_{m_c(a,b)} M`
  after the WLOG tangent–normal re-centering of paper_math.md Lemma 3.2.
- `{ξ(a, b)}` are i.i.d. with mean `μ_ξ ∈ R^{d_m}` and covariance `Σ_ξ`,
  and independent of `ε`.
- `ε_c` and `ε_w` share the same distribution `Σ_ε` (same-noise-across-
  populations; relaxed in paper_math.md Remark 4.9 / plan §5.1).

**Assumption BD (boundedness).** There exists `D_max < ∞` such that, with
probability at least `1 - δ` over the noise, `‖h‖_2 ≤ D_max` for every
activation in the analysis sample. For sub-Gaussian noise with parameter
`σ` plus a compact clean signal `m_p` uniformly bounded by `‖m_p‖_∞`,
the sub-Gaussian maximal inequality over `n` samples gives
`D_max = ‖m_p‖_∞ + σ(√d_m + √log(n/δ))` — a *sum of two distinct terms*,
not combined under one square root. The two terms have different origins:
`σ √d_m` is the typical Euclidean norm of an isotropic d-dimensional
sub-Gaussian vector, and `σ √log(n/δ)` is the maximum-over-n-samples
penalty. Combining them as `σ √(d_m + log(1/δ))` (as the previous draft
did) conflates the two regimes and is wrong when `n ≫ 1`. For
activations after layer normalization, `‖m_p‖_∞ = O(√d_m)` typically,
giving `D_max = O(√d_m + σ √log(n/δ))`. (Formal statement:
[paper_math.md Assumption 3.4](paper_math.md).)

**Assumption REG (manifold regularity).** There exist `τ_min > 0` and
`κ_max < ∞` such that the reach `τ(M) ≥ τ_min` uniformly and the operator
norm of the second fundamental form is bounded by `κ_max` on the data
support. We further require `σ · κ_max ≤ c_0` for a small absolute constant
`c_0 ∈ (0, 1]`, ensuring typical noise displacements are small relative to
the manifold's curvature scale so the linearization of paper_math.md
Lemma 4.3 is valid. We verify `σ · κ_max ≤ 0.1` empirically per (model,
layer) before computing `T_n`. (Formal statement:
[paper_math.md Assumption 3.5](paper_math.md).)

**Empirical diagnostics for REG and the curve-case drift (paper_math.md
Remark 3.6).** Both the reach lower bound `τ(M) ≥ τ_min` and the
curvature upper bound `‖II_p‖_op ≤ κ_max` are not directly observable
but can be estimated from correct-population activations. We
pre-register reporting the following diagnostics per (model, layer)
pair before computing the test statistic:

- *Reach lower bound* `τ̂`: Aamari–Levrard 2019 estimator
  `τ̂ ≥ min_{i ≠ j} ‖h_c^(i) - h_c^(j)‖² / (2 · d(h_c^(i) - h_c^(j), T_{p_j} M))`
  over a sample of pairs, where `T_{p_j} M` is estimated from local
  PCA at `p_j`.
- *Curvature upper bound* `κ̂_max`: maximum operator norm of
  `g''(t)` (the local Hessian of the parametric helix curve) at
  integer `t`, evaluated as `κ̂_max = max_t sup_{u ⊥ g'(t), ‖u‖=1}|⟨u, g''(t)⟩|`.
- *Effective normal-bundle scale* `σ̂_eff`: from the empirical
  residual covariance (paper_math.md Theorem 4.10 estimator).
- *Curve-case drift* `η̂`: sample base points `{p_k}` along the curve,
  estimate `P^N_{p_k}` from local PCA, compute
  `η̂ = max_k ‖P_V (P^N_{p_k} - P^N_{p_0}) P_V‖_op` for the chosen `V`
  and reference `p_0`.

The test is treated as conditional on the joint event that
`σ̂_eff · κ̂_max ≤ c_0` (small-noise regime) and `η̂ ≤ η_0` (drift
control for the curve case). Layers failing these diagnostics are
excluded.

**Curvature terminology.** "Curvature" throughout this plan means
*extrinsic* curvature, i.e., the operator norm of the second fundamental
form `‖II_p‖_op`, not sectional curvature. Sectional curvature is an
intrinsic Riemannian invariant of 2-planes and does not apply to the
1-dimensional helix curve. For a linear subspace, the second fundamental
form vanishes identically. (See paper_math.md §1.3.)

**Why these assumptions are reasonable.** Three reasons.

First, GM captures the simplest non-trivial alternative to "wrong activations
are correct activations at different positions on `M`." If `μ_ξ = 0`, the
wrong population's mean lies on `M` but with extra normal-direction variance;
this is a degenerate case in which we expect the test to read variance, not
structure. We address this in Theorem 1's null statement.

Second, the assumption that `ξ ⊥ T_p M` is without loss of generality up to
first order: any tangent component is absorbed into a different value of
`m_c(a, b)`, so writing `ξ` as the orthogonal-to-tangent component just
labels where on `M` we center. Concretely, let `P_T, P_N` denote the
orthogonal projections onto `T_{m_c(a,b)} M` and its complement, and
decompose `ξ = P_T ξ + P_N ξ`. Redefining

```
m̃_c(a, b) := m_c(a, b) + P_T ξ(a, b),     ξ̃(a, b) := P_N ξ(a, b)
```

preserves the model `h_w = m̃_c + ξ̃` while making `ξ̃ ⊥ T_{m̃_c} M` to
first order; the second-order correction is curvature-controlled at the
quadratic rate `‖ξ̃ - P_N ξ‖_2 ≤ (1/2) κ_max · ‖P_T ξ‖_2²` (paper_math.md
Lemma 3.2 — note the *squared* tangent-component norm, not linear), and
is absorbed into the linearization remainder `R_1 = O(σ² κ_max²)` of
paper_math.md Lemma 4.3. The exponential map `exp_p: T_p M → M` makes
this WLOG re-centering rigorous: for a 1D arc-length-parameterized curve,
`exp_{g(t)}(c · g'(t)/‖g'(t)‖) = g(t + c)`. Throughout the plan we work
with this re-centered version and drop the tilde, treating `ξ ⊥ T M` as
the working assumption. This is the same geometric noise decomposition
used in Bickel-Lindner 2008 for non-Euclidean covariance models.

Third, BD and REG are routinely satisfied for residual-stream activations
of layer-normalized transformers: `D_max = O(√d_m)` and the helix curve
has bounded extrinsic curvature on its compact integer support. We verify
both empirically per (model, layer) pair as a sanity check before
computing `T_n`.

The strongest assumption is GM(iv): within-population noise is the same
across correct and wrong populations. We can relax this and still prove a
slightly weaker version of Theorem 1; the trace terms no longer cancel and
pick up a `tr(P^N (Σ_ε^w - Σ_ε^c) P^N)` correction bounded by
`k · ‖Σ_ε^w - Σ_ε^c‖_op`. We discuss this in Section 5 and in
[paper_math.md Remark 4.9](paper_math.md).

### 1.5 Two manifold objects: the helix curve and the helix span

KT's helix is *both* a 1-dimensional curve (the image of the parameterization
`a ↦ helix(a)` for `a ∈ ℤ`) and the embedding of that curve into an
ambient subspace of dimension `m ∈ {8, 9}` (one linear axis plus the
non-degenerate cosine/sine pairs; `m = 9` for continuous inputs, `m = 8`
for integer-only inputs after the `sin(πa)`-degeneracy reduction —
paper_math.md Remark 2.2). These are different geometric objects and they
correspond to different failure modes.

We define both formally and use both throughout.

- **Helix curve `M_C`.** The 1-dimensional submanifold given by
  `M_C = {helix(a) : a ∈ ℤ} ⊂ R^{d_m}`. A point `h` is "on the curve" if
  `h = helix(a)` for some integer `a`. The closest-point projection
  `π_{M_C}(h) = argmin_{a ∈ ℤ} ‖h - helix(a)‖²` finds the nearest integer's
  helix point.
- **Helix span `M_S`.** The linear subspace
  `M_S = span{u_lin, cos(2π·/T_j) · u_cos^{T_j}, sin(2π·/T_j) · u_sin^{T_j} : T_j ∈ {2, 5, 10, 100}}
  ⊂ R^{d_m}`. The *nominal* dimension of this span is 9 (one linear axis
  plus four cosine/sine pairs); the *identifiable* dimension on integer
  inputs is `m = 8` because the `sin(2π·a/2) = sin(πa) = 0` column
  vanishes identically on integer `a` (paper_math.md Remark 2.2; also
  KT Figure 12). We use `m = 9` for the continuous-input ideal model
  and `m = 8` for the integer-only empirical setting, with `K = 8`
  basis functions in the OLS design matrix throughout the empirical
  pipeline. A point `h` is "in the span" if it lies in this `m`-dim
  subspace. The orthogonal projection `π_{M_S}(h) = U_S U_S^T h` for
  orthonormal `U_S ∈ R^{d_m × m}` is straightforward linear algebra.
  Throughout the paper, after the mean-centering of Remark 2.4
  (paper_math.md), the affine `M_S` becomes the linear subspace
  `S = span(...)` through the origin.

The relationship: `M_C ⊂ M_S`. Every point on the curve is in the span,
but most points in the span are not on the curve (they correspond to
non-integer `a` values, or to integer values plus span-internal noise).

**Three failure modes.** This distinction lets us decompose the residual
into three orthogonal components and identify three structurally different
failure modes.

| Mode | `T_curve` | `T_span` | Interpretation |
|---|---|---|---|
| **On-curve, in-span** | low | low | Wrong activation lies on the helix at a different integer position. Position error, not geometric error. |
| **Off-curve, in-span** | high | low | Wrong activation lies in the helix's `m`-dim ambient subspace but not at any integer position. Interpolation/drift along the span. |
| **Off-span** | high | high | Wrong activation has departed the `m`-dim ambient subspace entirely. Geometric structure has broken. |

The interpretive content of each mode is qualitatively distinct:

- *On-curve* errors say the model's number representation is intact and
  errors are computational (the model just landed at the wrong integer).
- *Off-curve in-span* errors say the model has constructed a *non-integer*
  point in the helix subspace — possibly a partial-computation result that
  the readout maps to the wrong token.
- *Off-span* errors say the model has activated mechanisms outside the
  helix entirely, suggesting a Llama-specific pathway not captured by the
  9-parameter helix (consistent with KT's Figure 23).

**Two test statistics.** We define two test statistics, one for each
manifold.

```
T_curve  =  (1/n_w) Σ_j ‖h_w^{(j)} - π_{M_C}(h_w^{(j)})‖²
            -  (1/n_c) Σ_i ‖h_c^{(i)} - π_{M_C}(h_c^{(i)})‖²

T_span   =  (1/n_w) Σ_j ‖h_w^{(j)} - π_{M_S}(h_w^{(j)})‖²
            -  (1/n_c) Σ_i ‖h_c^{(i)} - π_{M_S}(h_c^{(i)})‖²
```

By orthogonal decomposition, `T_curve = T_span + T_within-span` where
`T_within-span` measures the residual within `M_S` after removing the
curve. So the three failure modes correspond to:

- On-curve, in-span: `T_curve ≈ 0`, `T_span ≈ 0`.
- Off-curve, in-span: `T_curve > 0`, `T_span ≈ 0`.
- Off-span: `T_curve > 0`, `T_span > 0`.

We report both and interpret jointly.

For the rest of this paper, when we write `T_n` without qualification we
mean `T_span` (the more conservative test, since `T_curve` is harder to
estimate well due to the discrete projection onto integer points).
Theorem 1 below is stated for the linear-subspace projection (T_span);
the analogous result for `T_curve` is in Appendix B.

---

## 2. The mathematical contribution

### 2.1 Theorem 1: validity of the off-manifold test

The formal statement of this theorem and its full proof outline live in
[paper_math.md §4](paper_math.md), Theorem 4.2. The plan version below is
a reader-friendly paraphrase of the four-part statement.

**Statement.** Let `H_c = {h_c^{(i)}}_{i=1}^{n_c}` and `H_w = {h_w^{(j)}}_{j=1}^{n_w}`
be i.i.d. samples from correct and wrong populations under Assumptions
GM, BD, REG (§1.4) with isotropic Gaussian noise `Σ_ε = σ² I_{d_m}`.
Define the test statistic

```
T_n = (1 / n_w) · Σ_{j=1}^{n_w} ‖r(h_w^{(j)})‖²  -  (1 / n_c) · Σ_{i=1}^{n_c} ‖r(h_c^{(i)})‖²
```

where the projection `π_M` (and therefore the residual `r`) is computed using
the *true* manifold `M`. Let `k = d_m - dim(M)` be the dimension of the
residual (normal-bundle) subspace and `n = min(n_c, n_w)`. Then:

**(a) Population mean (alternative).** Under `H_1`: `μ_ξ ≠ 0` or `Σ_ξ ≠ 0`,

```
E[T_n]  =  ‖μ_ξ‖²  +  tr(Σ_ξ)  +  R_1(σ, κ_max),
```

where `R_1(σ, κ_max) = O(σ² κ_max²)` is a linearization-error term that
vanishes identically when `M` is linear (e.g., the helix span `M_S` after
mean-centering). Importantly, the mean of `T_n` depends only on the
perturbation `ξ`, not on the noise `ε`: under Assumption GM(iii) the two
populations share `Σ_ε`, so the noise traces
`tr(P^N Σ_ε P^N)` cancel exactly across the difference
`Ȳ_w - Ȳ_c`. (Remark 4.9 of paper_math.md handles `Σ_ε^c ≠ Σ_ε^w`
where the residual term `tr(P^N (Σ_ε^w - Σ_ε^c) P^N)` is bounded by
`k · ‖Σ_ε^w - Σ_ε^c‖_op`.)

**(b) Null behavior.** Under `H_0`: `μ_ξ = 0` and `Σ_ξ = 0` —
equivalently, `ξ(a, b) ≡ 0`, so the wrong-population signal coincides
with the correct-population signal in distribution.
`E[T_n] = R_1(σ, κ_max)` — identically zero when `M` is linear, and
`O(σ² κ_max²)` in general. Under Assumption REG with `σ · κ_max ≤ c_0`
this is a small bias. The null is a statement about the perturbation
`ξ` being zero, *not* about the noise `ε`: the ε-contribution to
`E[T_n]` cancels regardless of whether `ξ` vanishes.

**(c) Non-asymptotic concentration.** For every `t > 0`,

```
P[ |T_n - E[T_n]| > t ]  ≤  4 · exp(-c · n · min(t² / (k σ⁴), t / σ²))
```

with absolute constant `c ≥ 1/8` (the Laurent–Massart sub-exponential
chi-square constant). The factor 4 (vs. 2 in the previous draft) is the
union bound across upper and lower tails for the two populations.

**(d) Sharp Laurent–Massart form.** For any `u > 0` and `n_c = n_w = n`,

```
P[ T_n - E[T_n]  ≥  2 σ² √(2 k u / n) + 2 σ² u / n ]  ≤  2 e^{-u},
```

with the same bound on the lower tail. This is the form we use when we
need the explicit `√(2 k u / n)` rate for sample-size calculations rather
than the Bernstein-style `min(·, ·)` bound. (Form (c) is convenient for
constants; form (d) is sharp in `k`.)

The `k σ⁴` factor in the variance is the chi-square-style noise variance:
for isotropic Gaussian `ε ∼ N(0, σ² I)`, the squared projection-residual
norm `‖P^N ε‖² ∼ σ² · χ²_k` has variance `2k σ⁴`, and Laurent–Massart
gives the sub-exponential tails.

**Equivalent dimension-free form.** It is sometimes more convenient to
work with the per-dimension-normalized statistic

```
T̄_n = T_n / k
```

which has mean `(‖μ_ξ‖² + tr(Σ_ξ)) / k + O(σ² κ_max² / k)` and
concentration

```
P[ |T̄_n - E[T̄_n]| > t ]  ≤  4 · exp(-c · n · min(t² · k / σ⁴, t · k / σ²))
```

We prefer the unnormalized form `T_n` for the headline test (its
expectation is the squared norm of the perturbation, which is
interpretable directly), but the normalized form `T̄_n` is what we use
when comparing across models with different `k` (which differs because
the manifold dimension differs slightly across model architectures).

**Anisotropic / effective-rank extension (Theorem 4.10).** For arbitrary
positive-semidefinite `Σ_ε` with `‖Σ_ε‖_op ≤ σ_op²`, define the
*effective normal-bundle rank* and *effective normal-bundle scale*

```
k_eff   :=  (tr(P^N Σ_ε))² / tr((P^N Σ_ε)²)
σ_eff²  :=  tr(P^N Σ_ε) / k_eff  =  tr((P^N Σ_ε)²) / tr(P^N Σ_ε)
```

For isotropic `Σ_ε = σ² I`, `k_eff = k` and `σ_eff² = σ²`. The
chi-squared variance `2 k σ⁴` is replaced by `2 k_eff σ_eff⁴` and the
mean by `tr(P^N Σ_ε P^N)`. The non-asymptotic concentration bound (via
Hanson–Wright, paper_math.md Theorem 4.10 (4.7)) reads

```
P[ T_n - E[T_n] ≥ 2 σ_eff² √(2 k_eff u / n) + 2 σ_op² u / n ]  ≤  2 e^{-u}
```

— note the linear-tail term carries `σ_op²`, not `σ_eff²` (the operator
norm controls the per-direction sub-exponential tail; the effective
scale controls the Gaussian-tail variance). For strongly anisotropic
noise `k_eff ≪ k` and the variance term tightens, but the linear-tail
term is bounded by the worst-case direction. A consistent estimator
`k̂_eff = (tr Σ̂_r)² / tr(Σ̂_r²)` from `Σ̂_r := (1/n_c) Σ r(h_c^(i)) r(h_c^(i))^T`
is used in practice.
([paper_math.md Theorem 4.10](paper_math.md).)

**Plain-language interpretation.** Under the null, the test reads zero
(up to a small `O(σ² κ_max²)` curvature bias). Under the alternative, it
reads the squared norm of the perturbation's mean plus the trace of the
perturbation's covariance. Either signature — a non-zero mean shift or
just extra variance in the normal bundle — produces a positive `T_n`.

### 2.2 Proof sketch of Theorem 1

The full proof outline is in [paper_math.md §4.3–§4.6](paper_math.md);
this section gives the four-step skeleton in plan-friendly notation.
The four steps are:

1. **Linearize `π_M`** around clean points (paper_math.md Lemma 4.3,
   the tubular-neighborhood projection).
2. **Take expectations** of the squared residual under GM (paper_math.md
   Lemma 4.5 / Corollary 4.6).
3. **Bound the sub-exponential parameter** of the centered squared
   residual via Laurent–Massart (paper_math.md Lemma 4.7), then
   propagate to the empirical mean (paper_math.md Lemma 4.8).
4. **Union bound** across the two populations and two tails to get the
   four-event statement of Theorem 1(c)–(d).

The plan retains the Bernstein-style proof below as a self-contained
walkthrough; the sharp Laurent–Massart constants
`(2 σ² √(2 k u / n) + 2 σ² u / n)` from Theorem 1(d) come from the same
proof at the cost of tracking the LM constants through.

**Step 1: decompose the residual.**

For correct activations, by Assumption GM, `h_c = m_c(a) + ε_c` with
`m_c(a) ∈ M`. So

```
π_M(h_c) = π_M(m_c(a) + ε_c).
```

For `ε_c` small relative to the curvature of `M`, we can linearize. Let
`p = m_c(a)`, let `T_p M` be the tangent space to `M` at `p`, and let
`P_p^T, P_p^N` be the orthogonal projections onto the tangent and normal
spaces respectively. To first order in `‖ε_c‖`,

```
π_M(h_c) ≈ p + P_p^T ε_c
r(h_c) = h_c - π_M(h_c) ≈ P_p^N ε_c
```

So `r(h_c) ≈ P_p^N ε_c`, the normal-bundle component of the noise.

For wrong activations, `h_w = m_c(a) + ξ + ε_w` with `ξ ⊥ T_p M`. The same
linearization gives

```
π_M(h_w) ≈ p + P_p^T ε_w
r(h_w) = h_w - π_M(h_w) ≈ ξ + P_p^N ε_w
```

So `r(h_w) ≈ ξ + P_p^N ε_w`.

**Step 2: take expectations.**

```
E[‖r(h_c)‖²]  =  E[‖P_p^N ε_c‖²]  =  tr(P_p^N Σ_ε P_p^N)
              =  (d_m - dim(M)) · σ²    [under isotropic Σ_ε = σ² I]
E[‖r(h_w)‖²]  =  E[‖ξ + P_p^N ε_w‖²]
              =  ‖μ_ξ‖² + tr(Σ_ξ) + tr(P_p^N Σ_ε P_p^N) + 2 · ⟨μ_ξ, E[P_p^N ε_w]⟩
              =  ‖μ_ξ‖² + tr(Σ_ξ) + (d_m - dim(M)) · σ²
```

(The cross term vanishes because `P_p^N` is linear, so
`E[P_p^N ε_w] = P_p^N E[ε_w] = 0`.) Subtracting,

```
E[T_n] = E[‖r(h_w)‖²] - E[‖r(h_c)‖²] = ‖μ_ξ‖² + tr(Σ_ξ).
```

This is the population mean of `T_n`. It is zero under the null and equals
`‖μ_ξ‖² + tr(Σ_ξ)` under the alternative.

**Step 3: concentration around the mean.**

`T_n` is a difference of two means of squared norms of projections of
sub-Gaussian random vectors. Each summand `X_i = ‖r(h_i)‖²` is
sub-exponential. The sub-exponential parameter depends on the dimension
of the residual subspace.

For isotropic Gaussian noise `ε ∼ N(0, σ² I_{d_m})`, the projection
residual `P_N ε` has squared norm `‖P_N ε‖² ∼ σ² · χ²_k` with
`k = d_m - dim(M)`. The chi-squared distribution `χ²_k` has mean `k`
and variance `2k`, so `‖P_N ε‖²` has mean `k σ²` and variance `2 k σ⁴`.
Its sub-exponential norm `‖X_i - E[X_i]‖_{ψ_1}` scales as `√(k) · σ²`
(from the chi-squared sub-exponential parameter; see Wainwright 2019,
Section 2.1.1).

By Bernstein's inequality (Vershynin 2018, Theorem 2.8.2), for sample mean
of sub-exponential variables with sub-exponential norm `K = √(k) · σ²`,

```
P[ |X̄ - μ| > t ]  ≤  2 · exp(-c · n · min(t² / K², t / K))
                   =  2 · exp(-c · n · min(t² / (k σ⁴), t / (√(k) · σ²)))
```

For `t · k > √(k) · σ²`, the linear-tail term `t / K` dominates, giving
`exp(-c·n·t / (√(k)σ²))`. We use the looser but cleaner form with
`σ²` (rather than `√(k)·σ²`) in the denominator of the linear-tail
term, which is conservative when `k` is large; the variance term
`t² / (k σ⁴)` is the one that matters for typical detection regimes.

Applying this to both `(1/n_c) Σ ‖r(h_c)‖²` and `(1/n_w) Σ ‖r(h_w)‖²`,
then combining via union bound, gives the stated concentration with the
explicit `k` factor.

The constant `c > 0` is universal (does not depend on `d_m` or `σ`). Under
non-isotropic noise the formula generalizes to: `tr(P_p^N Σ_ε P_p^N)` replaces
`(d_m - dim(M)) σ²` in the mean, and `tr((P_p^N Σ_ε P_p^N)²)` replaces
`(d_m - dim(M)) σ⁴` in the variance. The chi-squared variance of `‖P_N ε‖²`
under arbitrary `Σ_ε` is `2 tr((P_N Σ_ε)²)` (Magnus-Neudecker matrix
calculus, Theorem 11.21), which reduces to `2 k σ⁴` in the isotropic case.
The effective dimension `k_eff = (tr(P_N Σ_ε))² / tr((P_N Σ_ε)²)` then plays
the role of `k` in the concentration bound; we use `k_eff` in Section 5.1's
anisotropic discussion. □

**Full detail of Step 1: linearization of π_M.**

The linearization `π_M(p + δ) ≈ p + P_p^T δ` for small `δ` is the standard
local approximation in differential geometry. The error in this linearization
is `O(‖δ‖² · κ_max)` where `κ_max` is the maximum sectional curvature of `M`
at `p`. Specifically (see Federer 1959 on tubular neighborhoods, or Niyogi-
Smale-Weinberger 2008 for a modern statement):

```
‖π_M(p + δ) - (p + P_p^T δ)‖  ≤  (κ_max / 2) · ‖P_p^N δ‖²
```

For our setting with `‖δ‖ = O(σ)` and curvature `κ_max` controlled (we
verify this empirically per layer per model), the linearization error
contributes a *second-order additive* term `O(σ² · κ_max)` to the residual.
The leading first-order residual is `P_p^N ε = O(σ)`, so the relative error
is `O(σ · κ_max)` — small whenever `σ · κ_max ≪ 1`, which we verify
empirically per (model, layer) pair before computing T_n. We adopt as
**Assumption REG (regularity)**: there exists `κ̄` such that
`κ_max(p) ≤ κ̄` for all `p` in the data support, with `σ · κ̄ ≤ 0.1`
(an empirically checkable threshold). Under Assumption REG the
linearization is uniform in `p` and Theorem 1's rate is unchanged.

**Full detail of Step 2: the trace formula.**

The expectation `E[‖P_p^N ε‖²]` decomposes as

```
E[‖P_p^N ε‖²]  =  E[ε^T P_p^N ε]  =  tr(P_p^N · E[ε ε^T])  =  tr(P_p^N Σ_ε P_p^N)
```

For isotropic `Σ_ε = σ² I`, `tr(P_p^N · σ² I · P_p^N) = σ² · tr(P_p^N) = σ² · (d_m - dim(M))`.

For anisotropic noise with `Σ_ε ≠ σ² I`, the trace becomes
`tr(P_p^N Σ_ε P_p^N)`, which depends on how the noise's principal axes align
with the manifold's normal bundle. We bound this in the worst case by
`(d_m - dim(M)) · ‖Σ_ε‖_op` and use this anisotropic version in Section 5.1.

**Full detail of Step 3: the concentration argument.**

The summands `X_i = ‖r(h_i)‖²` are sub-exponential. Specifically, for
isotropic Gaussian noise `ε ∼ N(0, σ² I)` and `k`-dimensional projection
`P_p^N`, `‖P_p^N ε‖² ∼ σ² · χ²_k`, and the centered chi-square is
sub-exponential with the *Laurent–Massart* parameters

```
(ν, α) = (2 σ² √k,  2 σ²)
```

(paper_math.md Lemma 4.7). The two parameters separate the Gaussian-tail
regime (rate `t²/(k σ⁴)`) from the linear-tail regime (rate `t/σ²`).
Importantly, `ν` scales as `σ² √k`, *not* as `σ²` alone — the previous
draft's "`K = O(σ²)`" obscured the `√k` dependence. Laurent–Massart
[laurent2000] Lemma 1 states for `Z ∼ χ²_k` and `u > 0`,

```
P[Z - k ≥ 2 √(k u) + 2 u]  ≤  e^{-u}
P[Z - k ≤ -2 √(k u)]        ≤  e^{-u}
```

Multiplied by `σ²` and propagated through the empirical mean
(paper_math.md Lemma 4.8), this gives, for any `u > 0` and per-population,

```
P[ X̄_n - μ ≥ 2 σ² √(k u / n) + 2 σ² u / n ]  ≤  e^{-u}
```

Applying to each population at level `u/2` and union-bounding over
upper/lower tails of two populations (4 events total) gives

```
P[ |T_n - E[T_n]| ≥ 2 σ² √(2 k u / n) + 2 σ² u / n ]  ≤  4 e^{-u}
```

This is the sharp Laurent–Massart form of Theorem 1(d). For the
Bernstein-style form (4.1) of Theorem 1(c), set
`u = c n min(t²/(k σ⁴), t/σ²)`; the constant `c ≥ 1/8` is the LM
sub-exponential chi-square constant. □

### 2.3 Theorem 2: localization power

This is the headline theorem and the central methodological contribution.
The formal three-part statement and proof outline live in
[paper_math.md §5](paper_math.md), Theorem 5.2: (a) achievability,
(b) minimax lower bound, (c) exact null distribution. The plan version
below paraphrases all three parts.

**Statement.** Under Assumptions GM, BD, REG with isotropic Gaussian
noise, suppose **`ξ(a, b) ∈ V`almost surely** (so `μ_ξ ∈ V` and `Σ_ξ`
has support in `V`) where `V ⊂ R^{d_m}` is a fixed `r`-dimensional
subspace, and *one of the following holds*:

- **(L) Linear-span case.** `M = M_S` is the linear (helix) span. The
  common-normal subspace condition `V ⊆ ∩_p N_p M = M_S^⊥` is automatic
  (since `N_p M_S` is constant in `p`).
- **(C) Curve case with bounded normal-frame drift.** `M = M_C` is the
  helix curve, `V ⊆ N_{p_0} M_C` at a reference base point `p_0`, and
  the *normal-frame drift constant*
  `η := sup_{p ∈ M} ‖P_V (P_p^N - P_{p_0}^N) P_V‖_op ≤ η_0` is small.
  In this case, the conclusions hold with an additional
  `O(η_0 D_max^2)` term in `E[T_n^V]` and `O(η_0 r σ^2)` in variance.

The strict assumption `V ⊆ ∩_p N_p M` used in earlier drafts is
unrealistic for a curved 1D manifold: the normal space rotates along
the curve and the intersection over all `p` collapses to a small set.
Case (C) replaces it by an explicit, empirically verifiable drift bound
(see [paper_math.md §5.2](paper_math.md), Theorem 5.2). Define

```
T_n^V = (1 / n_w) · Σ_{j=1}^{n_w} ‖P_V r(h_w^{(j)})‖²  -  (1 / n_c) · Σ_{i=1}^{n_c} ‖P_V r(h_c^{(i)})‖²
```

where `P_V = U_V U_V^T` is the orthogonal projection onto `V` for
orthonormal `U_V ∈ R^{d_m × r}`. Let `Δ := ‖μ_ξ‖_2`. Then:

**(a) Achievability (upper bound).** For the test that rejects `H_0` when
`T_n^V > c_α` with `c_α` calibrated to level `α`, the test achieves power
at least `1 - β` provided

```
n  ≥  C_1 · r σ⁴ / Δ⁴ · log(1 / β)
```

for an absolute constant `C_1` and `n = min(n_c, n_w)`. By contrast, the
unprojected test of Theorem 1 requires `n ≥ C · (d_m - dim(M)) σ⁴ / Δ⁴ · log(1/β)`
to achieve the same power.

**(b) Two-point Le Cam lower bound (proven).** Let `P_Δ` denote the class
of GM distributions with `‖μ_ξ‖_2 ≥ Δ` and `ξ ∈ V`. Any test `ψ` with
size at most `α` and uniform power `inf_{P ∈ P_Δ} P[ψ = 1] ≥ 1 - β`
requires

```
n  ≥  C_2^{LC} · σ² / Δ² · log(1 / (β (1 - α)))      (proven, two-point Le Cam)
```

This is the rate established in [paper_math.md §5.6](paper_math.md) via
Lemmas 5.5–5.6. It matches the upper bound in `σ² / Δ²` but does not
match in `r` or in the exponent of `Δ`.

**(b') Conjectured matching minimax rate.** We conjecture but do not
prove in this submission that the achievable rate (a) is minimax-optimal:

```
n  ≥  C_2 · r σ⁴ / Δ⁴ · log(1 / (β (1 - α)))         (conjectured)
```

This is the standard separation rate for testing `N(0, σ²I_r)` against
`{N(μ, σ²I_r) : ‖μ‖ = Δ}`, established by Ingster (1993, 2003) and
refined non-asymptotically by Baraud (2002) and
Collier–Comminges–Tsybakov (2017). The argument combines Le Cam with a
chi-squared mixture over a packing of `S^{r-1}(Δ)` — *not* the
Fano-over-orthogonal-directions argument an earlier draft sketched
(which gives only `r σ² / Δ² · log r`). Translating Ingster's argument
to our setting requires checking that the wrong-population conditional
distribution under GM matches the Gaussian sequence model after
Lemma 5.3's projection; we believe this is straightforward but have
not pushed it through. We treat (b') as a target for follow-up work
and present the BlackboxNLP submission with (b) only.

**(c) Exact null distribution.** Under `H_0` with isotropic Gaussian noise
and oracle `M`,

```
(n_c n_w / (n_c + n_w)) · T_n^V / σ²   →_d   ½ (χ²_r - r),
```

equivalently `T_n^V` is distributed exactly as a difference of two scaled
chi-squared random variables (a generalized chi-squared in the sense of
Davies 1980): `(1/n_p) Σ ‖P_V r(h_{i,p})‖² ∼ (σ²/n_p) χ²_{n_p · r}` for
each population, and `T_n^V = Ȳ_w - Ȳ_c`. The Wald form
`√(n_c n_w / (n_c + n_w)) · T_n^V / (σ² √(2r)) →_d N(0,1)` gives
closed-form `p`-values without permutation. Berry–Esseen with the modern
Esseen constant `0.4748` (Tyurin 2010) gives the standardized rate

```
sup_t | P[ (T_n^V - E[T_n^V]) / √(Var(T_n^V)) ≤ t ] - Φ(t) |  ≤  O(1 / √(n · r))
```

Note the rate *improves* with `r` (the chi-squared distribution is closer
to Gaussian when `r` is large, by internal CLT applied to
`χ²_r = Σ_i Z_i²`). For fixed `r`, the rate is the classical `O(n^{-1/2})`.
The previous draft incorrectly stated the rate as `O(√(r/n))`, which
inverts the scaling — paper_math.md Remark 4.11 explicitly catches
this. Wald-type CIs are valid when `n · r ≫ 1`, which is essentially
always.

**Plain-language interpretation.** If you can guess a low-dimensional
subspace `V` that contains the perturbation, projecting onto `V` first
reduces the sample-size requirement by a factor of `(d_m - dim(M)) / r`.
For `d_m = 4096`, `dim(M_S) = 8` (paper_math.md Remark 2.2's `T = 2`
identifiability fix), `r = 1`, the speedup is `4088×`. A perturbation
that requires ~4000 samples to detect by the unprojected test can be
detected by the localized test in just a few samples.

This is a substantial but not astronomical improvement. We deliberately
*do not* claim quadratic-in-`r` speedups; the variance of the chi-squared
test statistic scales linearly in dimension, so the sample-complexity
gain scales linearly in `(d_m - dim(M))/r`, not quadratically. The
two-point lower bound (b) gives only the `σ² / Δ²` floor; whether the
matching `r σ⁴ / Δ⁴` rate (b') holds is open (see above).

**Why the rate has `r` in it, not `d_m - dim(M)`.** The noise variance
scales with the dimension of the residual space. By projecting to `V`,
we are restricting attention to the directions where the signal lives,
so the test sees only the noise variance in `V`, which is `r · σ²`
rather than `(d_m - dim(M)) · σ²`.

**Why this matters for interpretability.** The subspace `V` in this
theorem is the practitioner's *hypothesis* about where the failure
lives. We pre-register candidate failure subspaces (carries, fragile
`T=2` components, higher-order Fourier components) and the test confirms
or rejects each one. The result localizes the failure mechanically.

**Adaptive `V` via Romano–Wolf.** If `V` is selected from a pre-specified
family `V = {V_1, ..., V_K}` rather than fixed, joint Type I error
control follows from Romano–Wolf step-down with studentized maxT
statistic `max_k T_n^{V_k} / σ̂²`. This preserves the achievability rate
up to a factor of `log K`. ([paper_math.md Remark 5.7](paper_math.md).)

### 2.4 Proof sketch of Theorem 2

The full proof outline is in [paper_math.md §5.3–§5.7](paper_math.md);
this section gives the achievability part in plan-friendly notation.
The five-step structure of the formal proof is:

1. **Project the residual** under GM: `P_V r(h_c) = P_V ε_c + O_P(σ² κ_max)`
   and `P_V r(h_w) = ξ + P_V ε_w + O_P(σ² κ_max)`
   (paper_math.md Lemma 5.3).
2. **Variance in projected space**: `Var(‖P_V r(h_c)‖²) = 2 r σ⁴` and
   `Var(‖P_V r(h_w)‖²) = 2 r σ⁴ + 4 ‖μ_ξ‖² σ² + O(‖Σ_ξ‖²)`
   (paper_math.md Lemma 5.4).
3. **Achievability**: solve for the sample size that makes the
   level-`α` test achieve power `1 - β` (steps 1–4 below).
4. **Two-point Le Cam lower bound** at rate `σ² / Δ²` (paper_math.md
   Lemmas 5.5–5.6 and the proof of Theorem 5.2(b)/(5.2a)). The matching
   `r σ⁴ / Δ⁴` rate is conjectural and would require an Ingster-style
   chi-squared mixture; see paper_math.md §11 item 4.
5. **Exact null distribution**: under `H_0` with isotropic Gaussian noise
   and oracle `M`, the standardized statistic converges to `½(χ²_r - r)`,
   with explicit Berry–Esseen rate `O(1/√(n r))` — note the rate
   *improves* with `r` (paper_math.md Remark 4.11), not `O(√(r/n))` as
   the previous draft incorrectly stated. (Proof of Theorem 5.2(c).)

The plan retains the achievability walkthrough below; the lower-bound
and null-distribution parts are referenced rather than re-derived here.

**Step 1: reduce to Theorem 1 in projected space.**

Define `r_V(h) = P_V r(h)`. Then `r_V(h_w) ≈ P_V (ξ + P_p^N ε_w)
            = P_V ξ + P_V P_p^N ε_w
            = ξ + P_V ε_w`

(using `ξ ∈ V` so `P_V ξ = ξ`, and `V ⊥ T_p M` so `P_V P_p^N = P_V`).

Similarly `r_V(h_c) ≈ P_V ε_c`.

**Step 2: take expectations of squared norms in V.**

Since `ξ ∈ V` a.s. by assumption, we have `‖ξ‖² = ‖μ_ξ‖² + tr(P_V Σ_ξ P_V)`
where `P_V = U_V U_V^T` is the orthogonal projector onto V. The notation
`tr(P_V Σ_ξ P_V)` reduces to `tr(Σ_ξ)` whenever `Σ_ξ` has support in V (the
generic case under our assumption). We retain the projected form for
generality:

```
E[‖r_V(h_w)‖²] = E[‖ξ + P_V ε_w‖²]
              = ‖μ_ξ‖² + tr(P_V Σ_ξ P_V) + r · σ²
E[‖r_V(h_c)‖²] = E[‖P_V ε_c‖²] = r · σ²
```

So `E[T_n^V] = ‖μ_ξ‖² + tr(P_V Σ_ξ P_V)`.

**Step 3: variance and concentration.**

The variance of each `‖P_V ε‖²` is `O(r · σ⁴)` rather than
`O((d_m - dim(M)) · σ⁴)`. Apply Bernstein's inequality to the projected
quantities. The variance term shrinks proportionally to `r`, which is what
gives the dramatic sample-complexity reduction.

**Step 4: power calculation.**

For the test to detect a signal of size `Δ²` against a noise variance of
`r σ⁴ / n`, we need `Δ² ≥ z_{1-β} · √(r σ⁴ / n)`, giving
`n ≥ z_{1-β}² · r σ⁴ / Δ⁴ = O(r σ⁴ / Δ⁴ · log(1/β))`. □

**Full detail of Step 1: projection of the residual under (L) and (C).**

We claim `P_V r(h_w) = P_V (ξ + ε_w) + e_w` with `e_w` controlled
according to the regime ((L) or (C)).

Recall `r(h_w) ≈ ξ + P_p^N ε_w + R_2` from Theorem 1's Step 1, where
`R_2 = O(κ_max ‖P^N δ‖²)` is the tubular linearization remainder. Then

```
P_V r(h_w)  =  P_V ξ + P_V P_p^N ε_w + P_V R_2
```

**Case (L) — linear span.** When `M = M_S` is linear, `P_p^N = I - U_S U_S^T`
is constant in `p`, so `V ⊆ M_S^⊥` gives `P_V P_p^N = P_V` exactly, and
the tubular remainder `R_2` vanishes (`κ_max = 0` for a linear M). Hence
`P_V r(h_w) = ξ + P_V ε_w` exactly to leading order in `σ²` (paper_math.md
Lemma 5.3, case (L)).

**Case (C) — curve with bounded normal-frame drift.** When `M = M_C` is
the helix curve, `V ⊆ N_{p_0} M` at the reference base point but
`V ⊄ N_p M` at general `p`. Decompose
`P_V P_p^N = P_V P_{p_0}^N + P_V (P_p^N - P_{p_0}^N) = P_V + P_V (P_p^N - P_{p_0}^N)`,
the second term having operator norm at most `η ≤ η_0` on `V`. Hence

```
‖P_V P_p^N ε - P_V ε‖  ≤  η_0 ‖ε‖    (drift error, O_P(η_0 σ))
‖P_V P_p^N ε_w (when ξ ≠ 0) - P_V ε_w‖  =  O_P(η_0 ‖ξ‖)   (additional drift)
```

so `‖e_c‖, ‖e_w‖ = O_P(σ² κ_max) + O_P(η_0 (σ + ‖ξ‖))` — the second
term is the normal-frame drift error specific to case (C). The
conclusions of Theorem 2 hold with an additional `O(η_0 D_max²)`
term in `E[T_n^V]` and `O(η_0 r σ²)` in variance, both of which we
report alongside the diagnostic `η̂ ≤ η_0`. ✓ The strict assumption
`V ⊆ ∩_p N_p M` used in earlier drafts holds in case (L) trivially
and in case (C) only with `η_0 = 0` (which fails for a curved 1D
manifold whose normal space rotates along the curve).

**Full detail of Step 2: the variance gain.**

Under the alternative, `‖P_V r(h_w)‖² = ‖ξ + P_V ε_w‖²`. The variance of
this quantity (under the noise distribution) is

```
Var(‖ξ + P_V ε‖²)  =  E[‖ξ + P_V ε‖⁴] - (E[‖ξ + P_V ε‖²])²
```

For Gaussian `ε ∼ N(0, σ² I)`, the projected `P_V ε ∼ N(0, σ² P_V)` is
Gaussian in `V`, and `‖P_V ε‖² ∼ σ² · χ²_r` (chi-squared with `r` degrees of
freedom).

The variance of `χ²_r` is `2r`, so `Var(‖P_V ε‖²) = 2r · σ⁴`.

For the alternative `Var(‖ξ + P_V ε‖²)`, the cross term `2 ⟨ξ, P_V ε⟩` has
variance `4 ‖ξ‖² σ²` (since `⟨ξ, P_V ε⟩ ∼ N(0, ‖ξ‖² σ²)`). Combined,
`Var(‖ξ + P_V ε‖²) = 2r σ⁴ + 4 ‖ξ‖² σ²`. The first term dominates when
`‖ξ‖² < r σ²`, the regime where detection is hard.

Compared to the unprojected test where the variance is `2(d_m - dim(M)) σ⁴ + 4 ‖ξ‖² σ²`,
projection reduces the noise variance by factor `r / (d_m - dim(M))`.

**Full detail of Step 3: combining via Bernstein.**

The same Bernstein argument as Theorem 1 applies, with `σ²` replaced by
`σ² · √(r/(d_m - dim(M)))` effectively in the variance. The concentration
exponent has `min(t²/(r σ⁴), t/(r σ²))`, leading to the stated bound on `n`.

**Full detail of Step 4: the explicit power calculation.**

To achieve power `1 - β` at level `α`, we need

```
P[ T_n^V > c_α | H_1 ]  ≥  1 - β
```

`c_α` is calibrated so `P[ T_n^V > c_α | H_0 ] = α`. Under `H_0`, the
distribution of `T_n^V` is centered at zero with standard deviation
`σ_0 = √(2 r σ⁴ / n + ...)`, so `c_α ≈ z_{1-α} σ_0`.

Under `H_1`, `T_n^V` is centered at `‖μ_ξ‖² + tr(Σ_ξ |_V)` with similar
standard deviation. Power is

```
P[ T_n^V > c_α | H_1 ]  =  Φ((‖μ_ξ‖² - z_{1-α} σ_0) / σ_0)
```

Setting this `≥ 1 - β` and solving for `n` gives the stated rate. The
specific constant `C` in `n ≥ C · r σ⁴ / Δ⁴ · log(1/β)` depends on
`α` and `β`; for `α = β = 0.05`, `C ≈ 17`. We absorb constants into
the universal `C` for simplicity. □

### 2.5 Theorem 3: finite-sample concentration of M̂

This is the foundation under Theorems 1 and 2. The previous theorems assume
the manifold `M` is known. In practice we estimate it from `H_c`. Theorem 3
bounds the estimation error and propagates it through.

We give five candidate estimators and prove the cleanest statement for the
parametric estimator. We also state the empirical evidence for the others
and explain their theoretical status.

**Statement (parametric estimator).** Let `B = [b_1(a) ... b_K(a)]` be a
known basis of `K` functions of the integer label `a`, and let `H_c` be `n_c`
correct activations with corresponding labels `a_1, ..., a_{n_c}`.
Concretely `B ∈ R^{n_c × K}` (one row per sample) and `H_c ∈ R^{n_c × d_m}`.
The OLS coefficient is `Ĉ := (B^T B)^{-1} B^T H_c ∈ R^{K × d_m}`. The
estimated manifold subspace is

```
M̂_param  =  col-span(Ĉ^T)  ⊂  R^{d_m}.
```

That is, the columns of `Ĉ^T ∈ R^{d_m × K}` (each a `d_m`-vector,
corresponding to one basis-function direction) span `M̂_param`. We
orthonormalize via QR for use in projections: `U_M̂ = QR(Ĉ^T)[0]`.

**Note on T=2 fragility (paper §1.2 footnote).** When the basis includes
`sin(2π·a/2) = sin(π·a)`, that column is identically zero on integer `a`,
making `B` rank-deficient. The toy validation (§0 above) discovered this
empirically. We drop the `sin(π·a)` column throughout, giving `K = 8` for
the four-period basis `{linear, cos(πa), cos/sin pairs for T ∈ {5,10,100}}`.
This matches KT's effective basis on integer inputs; KT's Figure 12
discusses the T=2 fragility explicitly.

Suppose Assumption GM holds with sub-Gaussian within-class noise of
parameter `σ`, and the parametric class is well-specified
(`E[H_c | B] = B (C*)^T` exactly). Let `λ_min^B := λ_min(E[bb^T])` and
`λ_max^B := λ_max(E[bb^T])` where `b = (b_1(a), ..., b_K(a))^T` and the
expectation is over the empirical distribution of integer labels. Let
`κ^B := λ_max^B / λ_min^B` denote the design condition number, and let
`σ_K(C*)` denote the smallest singular value of the population
coefficient matrix `C* ∈ R^{K × d_m}`. Then with probability at least
`1 - δ`,

```
sin θ_max(M̂_param, M)  ≤  C_3 · σ √(κ^B) / (√(λ_min^B) · σ_K(C*)) · √(d_m · log(d_m / δ) / n_c)
```

for an absolute constant `C_3` (paper_math.md Theorem 6.2, eq (6.1)).
When the helix design is well-conditioned (`κ^B = O(1)`, which holds
for the orthonormalized helix basis after centering), this simplifies to

```
sin θ_max  ≤  C_3' · σ / (√(λ_min^B) · σ_K(C*)) · √(d_m log(d_m / δ) / n_c)
```

The denominator carries `√(λ_min^B)` rather than `λ_min^B`: in the OLS
operator-norm bound, `(B^T B / n_c)^{-1}` contributes `1/λ_min^B`, but
`‖B^T E‖_op ≲ σ √(n_c · λ_max^B · (d+K))` contributes a `√(λ_max^B)`,
and the two combine to `√(λ_max^B)/λ_min^B = √(κ^B)/√(λ_min^B)`. The
previous draft had `1/λ_min^B` as a single factor in the denominator,
which is dimensionally incorrect — paper_math.md §6.2 prose corrects
this. The product `√(λ_min^B) · σ_K(C*)` is the "effective signal
strength" of the parametric fit. For KT's helix with periods
`T ∈ {2, 5, 10, 100}` (with `K = 8` after dropping the degenerate
`sin(πa)` column), both factors are bounded away from zero on the
integer range `[0, 99]`, so the bound is non-trivial.

**Proof.** The proof has two steps: (1) operator-norm bound on the OLS
coefficient via matrix concentration, (2) operator-norm Wedin to convert
to sin-theta. We use **operator-norm Wedin** (paper_math.md Lemma 6.4
via Stewart–Sun 1990 Theorem 3.6) rather than Frobenius Davis–Kahan /
Yu-Wang-Samworth as in the previous draft; this saves a `√K` factor in
the constant. Standard linear-regression theory (Vershynin 2018,
Section 4.7) gives the operator-norm bound on
`Ĉ = (B^T B)^{-1} B^T H_c`:

```
‖Ĉ - C*‖_op  ≤  C · σ · √(κ^B) / √(λ_min^B) · √((d_m + K log(d_m / δ)) / n_c)
```

(paper_math.md eq (6.2)). The combination
`√(κ^B)/√(λ_min^B) = √(λ_max^B)/λ_min^B` reflects the joint contribution
of `(B^T B/n_c)^{-1}` and `‖B^T E‖_op`. Wedin's perturbation theorem
(paper_math.md Lemma 6.4) applied to `A = (C*)^T` and `E = (Ĉ - C*)^T`
then gives

```
sin θ_max(col-span((Ĉ)^T), col-span((C*)^T))  ≤  2 ‖Ĉ - C*‖_op / σ_K(C*)
```

provided `‖Ĉ - C*‖_op ≤ σ_K(C*) / 2`. Combining and absorbing constants
yields the stated bound. □

**Note on the `√K` Wedin saving.** The previous draft used the
Frobenius form `‖·‖_F ≤ √K ‖·‖_op` in the OLS step and then
Frobenius Davis–Kahan, picking up a spurious `√K` factor. Operator-norm
Wedin avoids this. We flag in [paper_math.md §11 item 5](paper_math.md)
that we have not double-checked the OLS bound's `K`-dependence does not
re-introduce `√K`; pending Barnábás's verification.

**Full detail of the OLS operator-norm bound.** See
[paper_math.md Lemma 6.3](paper_math.md). Under sub-Gaussian noise with
parameter `σ` and design matrix with population Gram `Σ_B = E[bb^T]`
(`λ_min^B = λ_min(Σ_B) > 0`, `λ_max^B = λ_max(Σ_B) < ∞`), with
probability at least `1 - δ`,

```
‖Ĉ - C*‖_op  ≤  C · σ · √(λ_max^B) / λ_min^B · √((d_m + K log(d_m / δ)) / n_c)
             =  C · σ · √(κ^B) / √(λ_min^B) · √((d_m + K log(d_m / δ)) / n_c)
```

via Vershynin 2018 Section 4.7 (which gives
`‖B^T E‖_op ≤ C σ √(n_c · λ_max^B · (d_m + K log(d_m/δ)))`) plus matrix
Bernstein (Tropp 2015) to invert `B^T B / n_c` (whose minimum eigenvalue
is `λ_min^B (1 - O(√(K log K / n_c)))` for `n_c ≫ K log K`). Note the
distinction from the previous draft: the previous form
`√((d_m + K log(d_m/δ)) / (n_c · λ_min^B))` is missing the `√(λ_max^B)`
factor that comes from `‖B^T E‖_op`.

**Full detail of the Wedin composition.** See
[paper_math.md Lemma 6.4](paper_math.md). For full-column-rank
`A, A + E ∈ R^{d × K}` with `‖E‖_op ≤ σ_K(A) / 2`,

```
sin θ_max(col-span(A), col-span(A + E))  ≤  2 ‖E‖_op / σ_K(A).
```

Applying this with `A = (C*)^T` and `E = (Ĉ - C*)^T` (so column spaces
become `M_true` and `M̂_param`) and substituting the OLS bound gives the
Theorem 3 statement. The product `λ_min^B · σ_K(C*)` is the "effective
signal strength" of the parametric fit; for KT's helix on the integer
range `[0, 99]` both factors are bounded away from zero.

#### 2.5.1 Misspecification accounting (paper_math.md §6.5)

Suppose the true manifold `M` is approximated by the parametric class
but not exactly contained in it: define the *misspecification radius*

```
b²(M)  :=  inf_{C}  E[ ‖m_c(a, b) - B(a) C^T‖_2² ]  >  0.
```

Then [paper_math.md Theorem 6.5](paper_math.md) gives, with probability
at least `1 - δ`,

```
sin θ_max(M̂_param, M)  ≤  b(M) / σ_K(C*)              [bias term]
                        +  C_3 σ / (λ_min^B σ_K(C*)) · √(d_m log(d_m / δ) / n_c)  [variance term]
```

The bias term does *not* vanish in `n_c`. For Llama 3.1 8B, where KT
Figure 23 shows weaker last-token helix fit, we expect `b(M) > 0` to be
non-trivial; this is exactly the regime where Section 3.6's
Diffusion-Maps-fallback rule kicks in. The bias term is what makes the
parametric estimator *not* converge to `M` even with infinite data when
the parametric class is wrong; reporting it explicitly lets the
Llama-specific discussion make this connection cleanly.

**Composition with Theorem 1.** Section 2.7 below states the composed
finite-sample bound; the full proof outline is
[paper_math.md Theorem 6.6](paper_math.md). The key intermediate fact is
that the operator-norm difference between projections onto two subspaces
with principal angle `θ` is `‖π_M - π_{M̂}‖_op = sin θ` (Stewart–Sun 1990
Theorem 3.6). For any `h` with `‖h‖ ≤ D_max`,

```
| ‖r_{M̂}(h)‖² - ‖r_M(h)‖² |  ≤  C_4 · D_max² · sin θ_max(M̂, M).
```

Averaged across populations and combined with Theorem 1(d) via union
bound at `δ/2` each, this yields the Section 2.7 statement. Since
`sin θ_max(M̂, M)` is bounded by Theorem 3 at rate `1/√n_c`, the composed
rate stays at `1/√n` in dominant terms.

### 2.6 Theorem 3 in five flavors: empirical comparison of estimators

We tested five manifold estimators on a synthetic helix in `R^{50}` with
known truth (full code in `/mnt/user-data/outputs/five_methods_comparison.py`).
The empirical results back the theoretical claims. We summarize:

#### Method 1: Parametric helix fit (theorem proven above)

Construction: `B = [a, cos(2π a / T), sin(2π a / T) for T ∈ {2, 5, 10, 100}]`,
9 columns. OLS to recover `Ĉ` of shape `(9, d_m)`. `M̂_param` is the column
span of `Ĉ^T`.

Empirical at `n = 4000`: `θ_max = 0.64°`. Scales as `1/√n`.

Theory: clean concentration bound (proved above).

Reviewer-relevant: matches KT's own helix fitting protocol exactly.

#### Method 2: PCA on class means

Construction: bin `a` into `K_bins = 20` equal-width bins, compute the mean
activation `μ_k` for each bin, do SVD of the matrix
`[μ_1 - μ̄, ..., μ_{K_bins} - μ̄]`. `M̂_pca-means` is the column span of the
top-`k` left singular vectors.

Empirical at `n = 4000`: `θ_max = 6.31°`. **Saturates** at this level even as
`n → ∞`.

Theory: bias-variance tradeoff. The bias from binning is order `1/K_bins`,
not vanishing in `n`. The variance shrinks as `1/(n_c / K_bins)`. The total
error has an irreducible bias floor.

Why this matters for the paper: PCA on class means is the obvious
"data-driven baseline." We report it but flag its bias-limited behavior. It
will give us a reasonable but inferior estimate; in our framing, it serves as
a cautionary case for researchers tempted to use it without thinking about
binning bias.

#### Method 3: Local PCA (Singer-Wu 2012)

Construction: at each point `h_i`, find `k_neighbors` nearest neighbors in
`H_c`, do local SVD on the centered neighborhood, extract top-`k` directions.
Average the projection matrices `P_T(h_i) P_T(h_i)^T` across all points,
eigendecompose, take top-`k` eigenvectors.

Empirical at `n = 1000`: `θ_max = 84°`. **Catastrophically fails.**

Theory: Singer-Wu prove convergence under a small-curvature assumption that
the sectional curvature of `M` times the kernel bandwidth is small. Our helix
has high curvature (radius/pitch ≈ 10), so the assumption is violated. The
neighborhoods become tight enough that local PCA recovers the *intrinsic 1D
tangent direction* rather than the *3D ambient hull*. Forcing
intrinsic_dim = 3 then gives 1 signal direction plus 2 noise directions.

Reviewer-proof point: include this as an explicit negative result in the
paper. Tell the field which off-the-shelf manifold tools do not work for
mech-interp.

#### Method 4: Diffusion Maps (Coifman-Lafon 2006) plus regression lift

Construction: build affinity matrix `K_{ij} = exp(-‖h_i - h_j‖² / (2 σ_DM²))`,
row-normalize to `P`, eigendecompose. Take top non-trivial eigenvectors as
the embedding `Φ ∈ R^{n × k}`. Lift back to input space via regression:
solve `H ≈ Φ A` for `A ∈ R^{k × d_m}`. `M̂_dm` is the column span of `A^T`.

Empirical at `n = 4000`: `θ_max = 1.27°`. Scales cleanly as `1/√n`.

Theory: Coifman-Lafon prove operator-norm convergence of `(P - I) / σ_DM²`
to the Laplace-Beltrami operator on `M`, with rate
`O(σ_DM² + n^{-1/2} σ_DM^{-d/2})`. The lift step (regression) has standard
finite-sample theory. Combining the two steps does not give a clean published
rate for subspace recovery, but empirically it tracks `1/√n`.

For the paper: Diffusion Maps is the *Riemannian* alternative we present
alongside the parametric fit. It assumes only that activations come from a
manifold, not its parametric form. It serves as cross-validation: if both
parametric and Diffusion Maps recover the same `M̂`, the manifold-sufficiency
claim is robust to method choice.

#### Method 5: Kernel PCA (RBF) plus regression lift

Construction: build double-centered RBF kernel matrix
`K^c_{ij} = exp(-γ ‖h_i - h_j‖²) - centered`, eigendecompose. Top eigenvectors
form embedding scaled by `√λ`. Lift via regression as in Method 4.

Empirical at `n = 4000`: `θ_max = 1.27°`. Identical to Diffusion Maps in our
setting (both are kernel methods with median-bandwidth heuristic; they agree
exactly when the bandwidths match).

Theory: kernel PCA consistency (Schölkopf-Smola-Müller 1998); regression-lift
recovery again has no clean textbook bound but empirically scales `1/√n`.

For the paper: include alongside Diffusion Maps as a second
non-parametric/kernel method. Show it agrees with Method 4 to validate the
kernel approach is not bandwidth-specific.

#### LDA (mentioned, then dismissed)

We also tested LDA via `eig(S_W^{-1} S_B)`. Empirical at `n = 500`:
`θ_max = 22°`. The reason: LDA optimizes classification, not manifold
recovery. When within-class variance is anisotropic (which is true for any
non-trivial manifold structure), `S_W^{-1}` rotates the optimal classification
subspace away from the true manifold. This is correct LDA behavior, not a
bug. We discuss in a remark and do not report LDA as a primary estimator.

#### Summary table for the paper

| Method | sin θ at n=4000 | Scaling | Theory | Recommendation |
|---|---|---|---|---|
| 1. Parametric helix | 0.011 (0.64°) | 1/√n | Clean (op-norm Wedin) | **Primary**, matches KT |
| 2. PCA on class means | 0.110 (6.31°) | Saturates | Bias-limited | **Cautionary baseline** |
| 3. Local PCA | breaks | Diverges | Curvature limit | **Negative result** |
| 4. Diffusion Maps + reg | 0.022 (1.27°) | 1/√n | Operator conv. + reg | **Secondary, Riemannian** |
| 5. Kernel PCA + reg | 0.022 (1.27°) | 1/√n | Consistency + reg | **Cross-validation** |

The paper presents Method 1 as the primary basis for Theorem 3's clean
finite-sample bound. Methods 4 and 5 serve as Riemannian validation
methods. Methods 2 and 3 are reported as honest limitations of common
alternatives.

### 2.7 Composition: full theorem 1 + 3 finite-sample bound

This combines Theorems 1 and 3. The formal statement and proof outline
are [paper_math.md Theorem 6.6](paper_math.md). The plan version below
matches that statement.

Combining Theorem 1 with Theorem 3, the test statistic computed using
parametric `M̂_param` satisfies, with probability at least `1 - 2δ`,

```
|T_n^{M̂_param} - (‖μ_ξ‖² + tr(Σ_ξ))|  ≤
       (Theorem 1 noise term)  +  (Theorem 3 manifold-error term)
       =  2 σ² · √(2 k log(2/δ) / n) + 2 σ² log(2/δ) / n
       +  C_4 · D_max² · sin θ_max(M̂, M)
```

(where the first term is the sharp Laurent–Massart form from Theorem 1(d)
and the second is the Wedin operator-norm bound from
paper_math.md Theorem 6.6's proof). Substituting Theorem 3's bound on
`sin θ_max(M̂, M)` (with the corrected `√(κ^B)/√(λ_min^B)` factor) gives
the explicit form

```
|T_n^{M̂} - E[T_n]|  ≤  2 σ² √(2 k log(2/δ) / n) + 2 σ² log(2/δ) / n
                       +  C_4 · D_max² · σ √(κ^B) / (√(λ_min^B) σ_K(C*)) · √(d_m log(d_m / δ) / n_c)
```

where:
- `k = d_m - dim(M̂)` is the residual-bundle dimension (so the chi-squared
  variance scales as `k σ⁴`, giving the `√k` factor in Theorem 1's term —
  this was missing in the previous draft).
- `D_max := max_h ‖h‖` is a uniform bound on the activation magnitude. We
  adopt as **Assumption BD (boundedness)**: there exists `D_max < ∞` such
  that `‖h‖ ≤ D_max` for all `h ∈ H_c ∪ H_w`, with high probability under the
  noise distribution. For sub-Gaussian noise plus a bounded clean signal
  (`m_c(a) ∈ M` with `M` compact), `D_max = O(‖m_c‖_{∞} + σ √(d_m + log(1/δ)))`
  with probability `1 - δ`. Activation post-LayerNorm typically gives
  `D_max = O(√d_m)`, which is what propagates as `D_max²` in the
  Theorem 3 manifold-error term.

Both terms vanish at rate `1/√n`. The total error of the test, in absolute
units, scales as `σ² √(k/n) + σ D_max² √(d_m log d_m / n_c)`. The dominant
term depends on the relative size of `σ` and `σ²`; for `σ < 1` (typical for
normalized activations) and `D_max = O(√d_m)`, the second term dominates.

**Key practical implication:** to detect a perturbation of size `‖μ_ξ‖² ≈ 1`,
we need `n_c ≥ C σ² d_m log d_m`. For Llama 3.1 8B with `d_m = 4096`, this
is `n_c ≥ C · σ² · 4096 · log 4096 ≈ 34000 σ²`. With `σ ≈ 0.1` (typical
post-LayerNorm), this is `n_c ≥ 340`. Easily achievable with the 10,000
ordered-pair dataset, subject to the exact tokenizer-retained count
reported empirically in Appendix H.

**Remark on tightening `D_max²` via centering (paper_math.md Remark 6.7).**
The `D_max²` factor in the manifold-error term is potentially loose: it
bounds the squared norm of the activation, including the center-of-mass
component which is annihilated by the projection difference
`π_M - π_{M̂}` when both pass through the same affine offset. After
mean-centering activations (paper_math.md Remark 2.4), the relevant
quantity is the *centered* width
`D_max^ctr := max_i ‖h_c^(i) - h̄_c‖`, which is typically much smaller
than the uncentered `‖h_c^(i)‖` at deeper layers (where activations have
large mean component due to residual-stream accumulation). Empirically
we expect `D_max^ctr / D_max ∈ [0.1, 0.3]`. We will report both the
uncentered Theorem 6.6 bound and the centered version with `D_max^ctr`
replacing `D_max`.

**Unstable-subspace sharpening (paper_math.md Remark 6.7, second
sharpening).** When `M̂` and `M` share their first `m_0` basis directions
(typical for the helix span where the linear and large-period
`T ∈ {50, 100}` components are recovered with high accuracy and the
bulk of the error is in high-frequency `T ∈ {2, 5}` components), the
relevant operator-norm difference `‖π_M - π_{M̂}‖_op` is restricted to
the unstable subspace, often substantially smaller than the global
`sin θ_max`. We report both the global and per-subspace versions.

---

## 3. The empirical plan

### 3.1 Three models, three protocols

We run the full pipeline on:

- **GPT-J 6B** (EleutherAI 2021, 28 layers, 4096 dim, BPE tokenizer with
  single-token integers up to 999). KT's primary positive case.
- **Pythia 6.9B** (Biderman et al. 2023, 32 layers, 4096 dim, GPT-NeoX
  tokenizer). KT's secondary positive case.
- **Llama 3.1 8B base** (Llama 3 community license, 32 layers, 4096 dim,
  Llama 3 BPE tokenizer with single-token integers in some ranges). KT's
  inconvenient case where the algorithm claim is weakest.

Per-model tokenizer handling and prompt templates **(matching KT exactly,
not paraphrased)**. KT use different prompts on Llama vs GPT-J/Pythia; we
preserve this. KT do not justify the asymmetry — we flag it as a
methodological inconsistency in §5.4 but match for direct comparability.

- **GPT-J 6B**: integers 0-999 are single tokens.
  Prompt: `"Output ONLY a number. {a}+{b}="` (KT Table 2, Appendix).
  KT report 80.5% accuracy on two-digit addition.
- **Pythia 6.9B**: integers 0-999 are single tokens via GPT-NeoX tokenizer.
  Prompt: `"Output ONLY a number. {a}+{b}="` (same as GPT-J).
  KT report 77.2% accuracy.
- **Llama 3.1 8B base**: integers 0-999 are single tokens via Llama 3 BPE.
  Prompt: `"The following is a correct addition problem.\n{a}+{b}="`
  (KT Table 2). KT report 98% accuracy.

For all three, verify single-token status of every operand and every
candidate sum (`a+b ∈ {0,...,198}`) by tokenizing and confirming length 1.
Restrict primary analysis to (a,b) pairs where both inputs and the answer
tokenize as single tokens.

**Why per-model wrong-population sizes matter.** Given KT's accuracy numbers,
expected wrong-sample counts on the 10,000-pair primary dataset are
approximately:

| Model | KT accuracy | Expected n_w |
|---|---|---|
| GPT-J 6B | 80.5% | ~1950 |
| Pythia 6.9B | 77.2% | ~2280 |
| Llama 3.1 8B | 98% | ~200 |

Llama's tiny `n_w` is a real concern for power. The Section 2.7 composition
bound requires `n_w ≥ ~340` for detection at typical effect sizes, and
Llama at 200 falls below this. Mitigations:
1. The localized statistic `T_n^V` (Theorem 2) needs only `r σ⁴ / Δ⁴ · log(1/β)`
   samples, which for `r=1, σ=0.1, Δ=1` is ~10 samples. So even 200 wrong
   samples on Llama are sufficient for the localized test.
2. We can extend Llama's wrong sample by including the multi-token-answer
   robustness set as a sensitivity analysis (Appendix G).

For all three, we restrict empirical analysis to sums where both inputs and
the answer tokenize as single tokens. For two-digit addition with operands
in `[0, 99]`, sums range over `[0, 198]`. We pre-tokenize all candidate sums
and exclude problems where the answer is multi-token in any of the three
models, to keep the per-model populations directly comparable. (Multi-token
sums are deferred to an appendix robustness check.)

### 3.2 Data generation, pre-registered (KT-faithful extraction protocol)

Lock the following before any experiment runs. This protocol mirrors KT's
Section 5 procedure precisely; deviations are flagged.

- **Operand range.** `(a, b) ∈ {0, ..., 99}²`. We use **all 10,000 ordered
  pairs** as the primary dataset. (Note: 5050 = (100·101)/2 is the
  *unordered*-pair count and is *not* a tokenizer-derived quantity. We
  reserve unordered analysis as a symmetry-controlled robustness check
  in the appendix.)
- **Tokenization audit, run BEFORE primary analysis.** For each model, we
  pre-tokenize:
  - All operands `a, b ∈ {0, ..., 99}`.
  - All possible answers `a + b ∈ {0, ..., 198}`.
  - All combinations of these in the prompt template.
  We report the *exact* retained set per model, with no estimates. The
  per-model retained count `n_m` and the three-way intersection
  `n_∩ = |{problems retained by all three models}|` are first-line
  numbers in the experimental section.
- **Why 10,000 not 5050.** Two-digit addition with operands `[0, 99]` and
  ordered pairs gives 10,000 distinct prompts. Tokenizer filtering reduces
  this; we report the actual reduction empirically. Symmetry deduplication
  (`a + b = b + a`) is a separate analysis and is not the same as
  tokenizer filtering.
- **Prompt template.** Per-model prompts as specified in §3.1 (different for
  Llama; matches KT Table 2 exactly). Run at greedy temperature = 0. Record
  the model's single-token next-token prediction.
- **Correctness label.** The model is "correct" on `(a, b)` if its top-1
  next-token prediction equals the integer `a + b`. Otherwise "wrong."
- **Per-model expected accuracy** (corrected to KT's actual published
  numbers, not the previous draft's estimates).
  KT report 80.5% accuracy on GPT-J 6B, 77.2% on Pythia 6.9B, 98% on
  Llama 3.1 8B. Expected wrong-sample counts on the 10,000-pair dataset:
  ~1950 GPT-J, ~2280 Pythia, ~200 Llama. The composition bound's
  `n_w ≥ ~340` is satisfied for GPT-J and Pythia comfortably; Llama
  is below it for the unprojected `T_n` but above it for the localized
  `T_n^V` (which needs ~10 samples for r=1, Δ=1). See §3.1 above.
- **Layer set.** Extract residual-stream activations at the `=` token at all
  layers `ℓ = 0, 1, ..., L_m`. We will later select the analysis layer per
  model based on where the helix-fit quality (KT-style metric) peaks for
  the correct population. Pre-registered candidates: layers
  `{L_m / 4, L_m / 2, 3 L_m / 4, L_m - 1}` for each model. Final layer
  chosen from the four based on a hold-out subset.
- **Replicate count.** Single forward pass per problem (deterministic given
  greedy decoding). No within-problem replication needed.

### 3.3 Difficulty stratification: the central methodological safeguard

The formal exact-conditional-validity statement for this section's
matched permutation test is [paper_math.md Theorem 8.3](paper_math.md),
proved via the Lehmann–Romano permutation-group argument
(Lehmann–Romano 2005 Theorem 15.2.1) applied to the within-bin
permutation group `∏_B 𝔖(φ⁻¹(B))`. The plan version below describes
the implementation; the formal proof of Type I control is in the math
file.

This section addresses the most important threat to the headline result:
that wrong samples are systematically *harder* than correct samples on
features unrelated to manifold geometry, so any positive `T_n` could
trivially reflect input-difficulty distribution rather than failure
geometry.

**The problem.** Wrong-population samples for two-digit addition are not
i.i.d. from the same distribution as correct-population samples. They
concentrate on:

- Larger sums (`a + b ≥ 100` with carries).
- Larger operand magnitudes individually.
- Specific carry patterns (e.g., `9 + 9 = 18` triggers a carry; `4 + 5 = 9`
  does not).
- Tokenization-sensitive cases (some sums are multi-token in some models).
- Edge cases near 99 / 100 / 198.

If we ran a naive permutation test against an unconditional null, we
would reject `H_0` because the *inputs* differ between populations, not
because of any geometric structure in the activations.

**The redefined null.** The hypothesis we are actually testing is *not*

> H_0^naive: correct and wrong activations are drawn from the same distribution.

It is, formally (paper_math.md Definition 8.2):

> H_0^conditional: for every bin `B ∈ B`, the joint within-bin
> distribution `{(h_i, y_i)}_{i ∈ φ⁻¹(B)}` is invariant under permutation
> of indices. That is, conditional on `φ`, correct/wrong labels within
> each bin are exchangeable.

In words: conditional on the problem's arithmetic features (sum
magnitude, carry status, operand magnitudes, tokenization properties),
the wrong activations have the same residual distribution as correct
activations within each bin. This is a stronger and more meaningful null
than the unconditional version. Rejecting it means the geometric
difference is *not* attributable to input difficulty.

**Implementation: matched-pair permutation, with COARSE bins.**

A naive feature vector that includes operands exactly,
`φ_naive(a, b) = (a, b, a+b, carry, tokenizer_class)`, fails operationally:
under deterministic greedy decoding, each `(a, b)` produces one outcome
(correct or wrong), so each bin contains a single sample. Within-bin
permutation is impossible — there is nothing to shuffle. We caught this
in design review and rejected the naive scheme.

The correct scheme uses *coarse* bins that group many problems together,
ensuring each bin contains both correct and wrong samples in non-trivial
counts. We define:

```
φ(a, b) = (sum_bin, carry_pattern, a_decile, b_decile, answer_token_class)
```

where:

- `sum_bin ∈ {[0, 49], [50, 99], [100, 149], [150, 198]}` — 4 bins.
- `carry_pattern ∈ {none, ones-only, tens-only, both}` — 4 bins (ones
  carry triggers if `(a mod 10) + (b mod 10) ≥ 10`, tens carry if the
  sum spans 100).
- `a_decile, b_decile ∈ {0-9, 10-19, ..., 90-99}` — 10 bins each.
- `answer_token_class ∈ {single-token answer, multi-token answer}`,
  per-model.

This gives a target of `4 × 4 × 10 × 10 × 2 = 3200` cells in principle,
but most are empty (e.g., `sum_bin = [150, 198]` and `a_decile = [0-9]`
is impossible). Empirically, the non-empty cells number about 200–300.
With 10,000 ordered pairs and ~5–15% wrong rate (per KT), expected
samples per non-empty cell are ~30–50 correct and ~3–5 wrong. We pre-
register a *minimum 2 wrong samples per bin* threshold — bins below this
are dropped from the matched analysis.

**Data convention for matched permutation (paper_math.md §8 data
convention).** The matched permutation test runs on the *pooled* data
`D_n = {(a_i, b_i, h_i, y_i)}_{i=1}^n` where `i` indexes the union of
correct and wrong populations and `y_i ∈ {0, 1}` is the correctness
label. We do *not* permute within `H_c` alone (which would give
single-label bins where permutation is meaningless); we permute the
*labels* `y_i` within bins of pooled `(h_i, y_i)` pairs. The
exchangeability condition
`H_0^cond: h ⊥ y | φ(a, b)` is then well-defined, and the permutation
group `G = ∏_B 𝔖(φ⁻¹(B))` acts on the data by permuting indices
within each bin. Lehmann–Romano 2005 Theorem 15.2.1 (the randomization
hypothesis for general invariance groups) then gives
`P[T_n^matched > c_α] ≤ α`.

For the matched analysis:

1. Bin all problems by `φ`. Within each retained bin, count correct
   (`n_c^bin`) and wrong (`n_w^bin`) samples; require `n_w^bin ≥ 2` and
   `n_c^bin ≥ 2`.
2. Run a bin-restricted permutation test: shuffle correct/wrong labels
   only among samples within that bin (i.e., permute `y_i` while
   holding `h_i` and `φ(a_i, b_i)` fixed within bin), recompute
   `T_n^bin`, build a bin-conditional null distribution from 1000
   shuffles.
3. The matched test statistic `T_n^matched` is the inverse-variance
   weighted average of per-bin test statistics, with weights
   proportional to `min(n_c^bin, n_w^bin)`.
4. Reject the conditional null if the observed `T_n^matched` exceeds
   the 95th percentile of its empirical null distribution under
   bin-restricted shuffling.

If too few cells satisfy the `n_w^bin ≥ 2` threshold to give a
well-powered test (pre-registered: at least 50 retained cells), we fall
back to the next-coarser scheme: drop `a_decile` and `b_decile`, keep
only `(sum_bin, carry_pattern, answer_token_class)` for ~32 cells. We
pre-register both schemes so the choice is not made post-hoc.

We report both:
- Unconditional `T_n` from Section 3.7 below (informative if also
  significant; subject to the difficulty confound critique).
- Conditional `T_n^matched` from this section (the headline test).

A positive result requires `T_n^matched > 0` significantly. If
unconditional `T_n` is significant but `T_n^matched` is not, we report
this as "wrong activations are off-manifold *because they correspond to
harder problems*, not because of intrinsic geometric structure."

**Regression-based residualization (secondary analysis).** As an additional
robustness check, regress `‖r(h)‖²` on `φ` (treating each component as a
categorical fixed effect) and test whether the correct/wrong label
predicts residual norm beyond what `φ` predicts. If the partial effect
of correctness vanishes after controlling for `φ`, the geometric story
is weakened. Standard linear regression with HC3 heteroskedasticity-
robust standard errors. We use `φ` from above (the coarse scheme), not
the operand-exact scheme.

**Propensity-score reweighting (tertiary analysis).** As a third check,
fit a logistic regression of correctness on `φ`, compute propensity
weights, and test the reweighted `T_n` against zero using IPW standard
errors. Sensitivity to extreme propensities is bounded by trimming at
the 1st and 99th percentiles. Three different controls (matched
permutation, regression residualization, propensity reweighting) that
can disagree if the confound model is wrong; we pre-register all three
and require the headline result to hold across at least two.

### 3.4 Cross-fitting protocol: separating fit, calibration, and test

The second methodological safeguard addresses overfitting bias. The
formal asymptotic-normality and influence-function statement for the
cross-fitted estimator is [paper_math.md §7](paper_math.md), Theorem 7.2
and Corollary 7.3. Cross-fitting is *Neyman-orthogonal* in the sense of
Chernozhukov et al. 2018: the influence function

```
φ_p(h) = ‖r_M(h)‖² - E[‖r_M(h_p)‖²]   (for p ∈ {c, w})
```

is orthogonal at first order to manifold-estimation errors, so the
cross-fitted statistic is `√n`-consistent without rate loss from the
plug-in `M̂`.

**Explicit Neyman-orthogonality verification (paper_math.md §7.3).**
Parameterize the moment function in the linear-span case as

```
ψ(h_c, h_w; θ, Π) := (‖h_w - Π h_w‖² - ‖h_c - Π h_c‖²) - θ
```

with `Π = U U^T` for orthonormal `U ∈ R^{d × m}`, and let `Δ_U` be a
Stiefel-tangent direction (`U^T Δ_U + Δ_U^T U = 0`). Then

```
∂_t E[ψ(h_c, h_w; θ_0, Π_M + t · (U Δ_U^T + Δ_U U^T))]|_{t=0}
   = -2 E[ h^T (U Δ_U^T + Δ_U U^T)(I - UU^T) h ]
```

Under Assumption GM with the centered model `E[h_p] ∈ M_S = col(U)` for
both populations, `(I - UU^T) E[h_p] = 0` to leading order under both
populations, so the cross-term vanishes from population means. The
remaining variance contribution is the same across populations under
GM(iii) and cancels in the difference `E[ψ]`. Hence the Gateaux
derivative vanishes, establishing Neyman-orthogonality. Under the
alternative `μ_ξ ≠ 0`, the orthogonality holds at first order in
`‖Û - U‖_op`, with residual bias of order
`‖Û - U‖_op · ‖μ_ξ‖ = o_P(‖μ_ξ‖²)` whenever Theorem 6.2 gives
`sin θ_max = o_P(Δ)`.

The asymptotic variance is

```
V_∞ = Var(φ_w(h_w)) / (n_w / n) + Var(φ_c(h_c)) / (n_c / n)
    = 4 k σ⁴ + 8 ‖μ_ξ‖² σ²       (for n_c = n_w = n/2),
```

giving a closed-form Wald CI `T_n^{cross} ± z_{α/2} √(V_∞ / n)` once `V_∞`
is estimated from the cross-fitted residuals (paper_math.md Corollary 7.3).

**The problem.** If we fit `M̂` on correct samples `H_c` and then evaluate
the correct residual baseline on the same `H_c`, the residual norm is
artificially small (M̂ has been chosen to minimize residuals on H_c by
construction). The correct-population residual baseline is biased
downward, the wrong-population residual is unaffected, and `T_n` is
biased upward. The test is then over-rejecting.

**The fix: K-fold cross-fitting.** We partition the correct sample
`H_c = {h_c^{(i)}}` into K folds (K = 5). For each fold k:

1. **Fit fold.** Use folds `{1, ..., K} \ {k}` to fit `M̂^{(-k)}`
   (the manifold estimated without fold k).
2. **Calibrate fold.** Compute residuals
   `r^{(-k)}(h) = h - π_{M̂^{(-k)}}(h)` for all correct samples in fold k
   and for all wrong samples.
3. **Aggregate.** The cross-fitted residual statistic is

```
T_n^cross  =  (1/n_w) Σ_j ‖r̄(h_w^{(j)})‖²  -  (1/K) Σ_k (1/|fold k|) Σ_{i ∈ fold k} ‖r^{(-k)}(h_c^{(i)})‖²
```

where `r̄(h_w)` is the average residual across folds `r̄(h_w) = (1/K) Σ_k r^{(-k)}(h_w)`,
since wrong samples can be evaluated against any of the K manifold
estimates.

This is exactly the cross-fitting protocol used in semi-parametric
inference (Chernozhukov et al. 2018). It removes the overfitting bias
from the correct-population baseline while preserving the test's power
under the alternative.

**Cross-fitting also extends to V identification.** The localization
subspaces V_k (carry, T=2, etc.) are identified from correct samples.
If we use the same correct samples to fit M̂, identify V_k, and calibrate
the correct-residual baseline, we are triple-dipping. The full
4-way protocol is:

- **Split A (50% of correct):** fit M̂.
- **Split B (25% of correct):** identify V_k via supervised LDA on carry
  labels, T=2 Fourier component extraction, etc.
- **Split C (25% of correct):** calibrate the correct-residual baseline.
- **Split D (all wrong samples):** evaluate `T_n` and `T_n^V` against the
  calibrated baseline.

We use this 4-way split as the primary protocol. K-fold cross-fitting
(Section 3.4 above) is the secondary protocol that recovers efficiency
when sample sizes are tight; it is reported in the appendix with results
shown to agree with the 4-way split.

**Why this matters.** Without cross-fitting, a reviewer can correctly
say: "your correct residuals are artificially small because you fit the
manifold on them; the test rejects trivially." With cross-fitting, the
correct-residual baseline is unbiased, and a positive `T_n` is meaningful.

### 3.5 Manifold construction at the chosen layer

The activation at the `=` token encodes more than just the answer
integer: it carries operand-specific information, intermediate
computation state, and prompt context. A single helix fit to one integer
label is one possible manifold; a richer manifold over the joint
`(a, b, a+b, carry)` may be more appropriate. We pre-register a
three-manifold comparison and use the simplest version that fits well.

**Three candidate parameterizations.**

| Manifold | Parameterization | Dimension | Purpose |
|---|---|---|---|
| **Answer helix** `M_S^answer` | `s = a + b`, fit basis as KT does for last-token helix | 9 | Tests whether activation lies near answer representation |
| **Operand-union helix** `M_S^union` | Separate helix fits for `a`, `b`, and `s = a + b`; take union | up to 27 (often less due to overlap) | Tests whether all number features (operands and answer) are individually intact |
| **Joint arithmetic manifold** `M_S^joint` | Linear combinations of basis features in `(a, b, s, carry)`: `[a, b, s, carry, cos/sin for each T_j on each]` | up to 30+ | Tests whether full computation state is geometrically intact |

**Pre-registered comparison protocol.** For each model, fit all three
manifolds on the correct population. Report the R² of each fit.
Selection of the primary manifold for the headline test:

1. If the answer helix `M_S^answer` achieves R² ≥ 0.9, use it as primary.
   This matches KT's last-token analysis directly and gives the cleanest
   story.
2. If `M_S^answer` achieves R² < 0.9 but `M_S^union` achieves ≥ 0.9, use
   the operand-union manifold. (Likely for Llama 3.1 8B per KT Figure 23.)
3. If neither achieves R² ≥ 0.9, use the joint arithmetic manifold
   `M_S^joint`. The interpretation is then that no simple helix
   captures the model's arithmetic representation; the joint manifold
   is the practical alternative.

We report all three across all three models in the appendix as a
robustness check. The headline test uses the primary selection per the
above protocol.

**Why this addresses the joint-manifold critique.** A reviewer asking
"is the correct-population manifold really a 1D answer helix, or a
higher-dimensional joint manifold?" gets a direct answer: we test all
three. The selection rule is pre-registered. Disagreement between the
three (e.g., test fires under `M_S^union` but not `M_S^answer`) is
itself an interpretable result — it tells us *which* component of the
arithmetic representation breaks for wrong cases.

**Five manifold-recovery methods within each parameterization.** For
each of the three manifolds, we recover `M̂` using the five methods of
Section 2.6:

1. **Parametric:** fit the basis `B` for that manifold (with appropriate
   variables) to the correct activations via OLS. This is the primary
   method per Section 4.4.
2. **PCA on class means:** bin the relevant grouping variable
   (`a` for answer helix, `(a, b)` for joint) and SVD the class means.
3. **Local PCA:** k-NN tangent estimation.
4. **Diffusion Maps:** RBF kernel with median-bandwidth.
5. **Kernel PCA:** double-centered RBF kernel.

Evaluate by computing pairwise principal angles between all five `M̂`s
*within each parameterization*. Hypothesis: parametric, Diffusion Maps,
and Kernel PCA agree to within 5° within each manifold; Local PCA fails
on the joint manifold most severely (highest curvature).

### 3.6 Layer selection protocol (KT-faithful)

The layer at which the helix becomes the "right" representation differs
across the model and across what is being represented. KT's Figures 4 and 5
walk through the per-layer dynamics for GPT-J in detail; we use those as
ground truth for our layer-selection protocol.

**KT's per-layer findings (GPT-J 28 layers):**

- **Layer 0**: most basic numerical processing happens here (Nikankin et al.
  2024). The output of layer 0 is when the model has finished forming "the
  representation of this number" before any task-specific processing.
  KT's Figure 4 shows a sharp jump from layer 0 input to layer 1 input
  in the operand-helix patching effect: by the end of layer 0 the helix is
  already established for individual operands a and b.
- **Layers 1-13** (early): operand-helix patching `helix(a)` matches full
  layer patching closely. Helix(a, b) (joint-operand) does better than
  helix(a+b) — the model has the operands but hasn't computed the answer
  yet. At these layers the answer-helix patches are well below the operand
  patches.
- **Layer 14**: the answer-helix `helix(a+b)` curve crosses operand-helix
  in KT's Figure 5. This is where the model starts representing the answer.
- **Layer 17**: helix(a+b) — just 9 parameters — beats 27-dim PCA. The
  answer-helix is now a more compact and faithful description than a
  generic data-driven 27-dim subspace.
- **Layers 19-22**: helix(a+b) matches the full-layer patch — the answer
  helix captures essentially all the information at the equals-sign token.
- **Layers 24-26**: the "a+b heads" KT identify (5 attention heads in this
  range) perform the final write of the answer to the unembedding.

**KT's per-layer findings (Pythia 6.9B, 32 layers; Llama 3.1 8B, 32 layers):**

- Pythia replicates GPT-J qualitatively: similar curve crossings in the
  layer 14-22 range proportionally adjusted for the 32-layer architecture.
- Llama 3.1 8B is the inconvenient case. KT Figure 23 shows the last-token
  helix fit on Llama is markedly worse than on GPT-J or Pythia. The
  algorithm-claim transfers cleanly to GPT-J and Pythia but only weakly
  to Llama. KT do not investigate further — this gap is precisely what our
  paper attempts to localize via wrong-population analysis.

**Our layer-selection protocol.** We pre-register the following per-model
candidate layers and a held-out selection rule:

1. **Operand-helix layer ℓ_op**: layer where `M̂_param` fitted to operand
   `a` activations achieves highest R² on a held-out fold. Pre-registered
   candidates per KT's findings:
   - GPT-J: `ℓ_op ∈ {1, 2, 3}` (early; KT Figure 4 peaks here).
   - Pythia: `ℓ_op ∈ {1, 2, 3}` (analogous early layer).
   - Llama: `ℓ_op ∈ {1, 2, 3}` (analogous; KT confirms this transfers).

2. **Answer-helix layer ℓ_ans**: layer where `M̂_param` fitted to the
   *last-token* (equals-sign) activation against the answer label `s = a + b`
   achieves highest R². Pre-registered candidates:
   - GPT-J: `ℓ_ans ∈ {17, 19, 21, 22}` (KT Figure 5 plateau).
   - Pythia: `ℓ_ans ∈ {19, 22, 25, 28}` (analogous, scaled to 32 layers).
   - Llama: `ℓ_ans ∈ {19, 22, 25, 28}` (KT Figure 23 says the last-token
     fit is weakest here; we test all four and accept whichever is best).

3. **Selection on a held-out fold.** Reserve 20% of correct samples per
   model as a layer-selection fold. Fit `M̂_param` on the other 80% at each
   candidate layer. Compute R² of the fit on the held-out 20%. Choose the
   layer with highest held-out R².

4. **Pre-registered fallback.** If the best held-out R² for `ℓ_ans` is below
   90% (the §5.3 threshold), we declare the parametric fit misspecified at
   that layer and switch the primary M̂ to Diffusion Maps (Section 4.5
   protocol). This is most likely on Llama 3.1 8B per KT Figure 23.

5. **Robustness reporting.** We additionally report `T_n` and ACEs at each
   candidate layer in Appendix J, even after primary selection — so a
   reviewer can see whether the conclusions are layer-specific or layer-robust.

**What we extract per layer.** For each of the candidate layers above,
extract the residual-stream activation at the equals-sign token for every
problem (correct AND wrong) using HuggingFace forward hooks on
`transformer.h.{ℓ}` (GPT-J), `gpt_neox.layers.{ℓ}` (Pythia), or
`model.layers.{ℓ}` (Llama). Cache to Babel scratch at
`/data/user_data/$USER/blackbox/activations/{model}/layer_{ℓ}.npz`
per [babel_execution_plan.md §2](babel_execution_plan.md). SLURM
templates and resume logic for the extraction phase live in
[babel_execution_plan.md §6](babel_execution_plan.md).

**Pre-registration constraint.** The layer-selection step uses only correct
samples on a held-out 20% fold. The wrong-population analysis (Sections 3.7
through 3.9) operates on the remaining 80% of correct samples plus all wrong
samples, with K-fold cross-fitting (§3.4) for the M̂ baseline. This 4-way
data split eliminates double-dipping risk.

### 3.7 Test statistic computation

For each `(model, layer, manifold-method)` triple, we compute three
versions of the test statistic, in order of methodological rigor:

**Version 1: Unconditional, single-fit (reported as "naive baseline" in
appendix only).** Compute residuals `r(h) = h - π_{M̂}(h)` for all correct
and wrong activations using a single M̂ fit on all correct samples.
Compute `T_n` per Theorem 1. Bootstrap 95% CI. Unconditional permutation.
This version is biased upward (correct residuals are artificially small)
and includes difficulty confounding. It is NOT the headline result.

**Version 2: Cross-fitted (reported in body).** Apply the K-fold cross-fitting
protocol of Section 3.4. The correct-residual baseline is unbiased.
Bootstrap CIs on the cross-fitted statistic. This corrects the overfitting
issue but does *not* correct the difficulty confound.

**Version 3: Cross-fitted + matched permutation (the headline test).**
Apply both Section 3.3 stratification and Section 3.4 cross-fitting. The
test rejects only if both:
- The cross-fitted `T_n^cross > 0` significantly under bootstrap.
- The matched permutation null (within `φ`-bins) places the observed
  cross-fitted statistic above its 95th percentile.

The Version 3 result is what we report as the primary finding.

We compute all three versions and report them in a single table so
readers can see how much of the apparent effect is attributable to
overfitting bias (V1 vs V2) and how much to difficulty confounding
(V2 vs V3). Pre-registered prediction: V3 effect should be at least 50%
of V1 magnitude on Llama 3.1 8B (where wrong-population behavior is
expected to differ most from correct), and the V3 confidence interval
should exclude zero.

If V3 is null while V1 is large, we report this as: "the unconditional
geometric difference between correct and wrong populations is largely
explained by difficulty distribution and overfitting bias rather than
intrinsic geometric divergence."

### 3.8 Localization via pre-registered subspaces (Theorem 2)

Pre-register candidate failure subspaces, computed from correct samples only.

- **V1 (carry subspace):** the LDA direction(s) separating `a + b ≥ 100`
  from `a + b < 100` (carry vs no-carry), within the correct population.
  Dimension `r_1 = 1` (binary discriminator).
- **V2 (fragile T=2 subspace):** the cos/sin pair for period `T = 2` from
  KT's helix basis. Dimension `r_2 = 2`.
- **V3 (higher-order Fourier):** cos/sin pairs for periods
  `T ∈ {3, 4, 6, 7, 8}` (periods KT did not include). Dimension `r_3 = 10`.
- **V4 (random-direction baseline):** an `r = 1` random unit vector chosen
  orthogonally to `M̂`. Repeated 100 times to estimate baseline distribution.
- **V5 (orthogonal complement of M̂, full):** dimension
  `r_5 = d_m - dim(M̂) ≈ 4087`. The "no localization" baseline.

For each `V_k`, compute `T_n^{V_k}` and its bootstrap CI. Apply
Benjamini-Hochberg FDR correction at `q = 0.05` across the V's.

The interesting finding is *which* `V` carries the wrong-population
perturbation, and how the answer differs across models. KT's Figure 12
suggests `T = 2` is fragile, so we expect V2 to be informative. KT's Section
5.5 admits the composition step is unlocalized; the carry subspace V1 might
be where it lives.

### 3.9 Causal analysis (body, not appendix)

Theorems 1, 2, and 3 establish a *correlational* claim: wrong-population
activations have a measurable off-manifold component, and that component
localizes to a specific subspace. To convert this into a *mechanistic*
claim, we need targeted interventions showing the localized subspace is
functionally responsible for the failure. This section spells out the
full causal pipeline. It is not an afterthought; it is what separates
this paper from a goodness-of-fit paper.

We follow the activation-patching protocol of Vig et al. 2020 and the
formalization of Geiger et al. 2021, with effect-size measurement à la
Wang-Variengien 2022 (IOI paper) and Conmy et al. 2023 (Automatic Circuit
Discovery). Patching is run with HuggingFace forward hooks on each model.

#### 3.9.1 Notation: the patching operation (KT-faithful)

Define the patching operation as follows. Let `h ∈ R^{d_m}` be a residual-
stream activation at the analysis layer `ℓ_m` for model `m`, captured at the
equals-sign token. Let `V ⊂ R^{d_m}` be a subspace with orthonormal basis
`U_V ∈ R^{d_m × r}`. Let `α ∈ R` be a scalar strength. Let `δ ∈ R^r` be a
target perturbation in `V`-coordinates. Then:

```
Patch(h, V, α, δ)  =  h  +  α · U_V · δ  -  α · (U_V U_V^T) · h
                   =  h  +  α · (U_V · δ - P_V h)
```

**KT-faithful patching protocol.** This matches KT's procedure (their Section
5, walked through their five-step example):
1. Pick a correct prompt (clean) and a wrong prompt (corrupted).
2. Run the model on the clean prompt; save residual stream `h^ℓ_=` at every
   layer at the equals-sign token.
3. Run on the corrupted prompt; save residuals.
4. Pick a layer `ℓ_m` (the analysis layer from §3.6). Re-run the corrupted
   prompt, but at layer `ℓ_m`, when the model is processing the equals-sign
   token, **overwrite the residual stream with the result of `Patch(h_corrupted, V, α, δ)`**.
   For removal patches (necessity), `δ = 0` and `α = 1` zeros out the
   V-component. For injection patches (sufficiency), set `α = 1` and `δ` to
   the target value.
5. Look at the logits for `s = a + b` after the patched run. Compare to the
   un-patched corrupted run. The change is the causal effect.

KT use the analogous trick of patching the *helix fit* `helix(s)` rather than
the actual residual `h^ℓ_=` (their Figure 5 shows this works as well as
patching the full residual at the right layers). Our test patches arbitrary
subspaces, but the helix-fit patching is a special case (V = M̂, δ chosen so
the result equals `helix(s)`). The toy validation (§0) confirms our patching
operator behaves correctly in both modes (toy E8 ACEs).

The first term `α · U_V · δ` injects a component `δ` along `V`. The second
term `-α · P_V h` zeros out the original `V`-component proportionally. Together,
at `α = 1`, this *replaces* `h`'s `V`-component with `δ`. At `α = 0`, no
change. At intermediate `α`, partial replacement.

Two specializations matter for the experiments below.

- **Removal patch** (`δ = 0`, `α ∈ [0, 1]`): zeros out `h`'s component
  along `V`. Measures necessity.
- **Injection patch** (`α = 1`, `δ` set to a target value): replaces `h`'s
  `V`-component with `δ`. Measures sufficiency.

#### 3.9.2 Behavioral metric: logit difference

Following Wang-Variengien 2022 and Conmy et al. 2023 (paper_math.md
Definition 9.2), define `LD` as a *downstream forward-pass functional*:
for `h ∈ R^{d_m}` representing a candidate residual-stream activation
at the analysis layer `ℓ_m` on a problem `(a, b)` with answer token
`τ(s)`, let `forward_{ℓ_m → L}(h; a, b)` denote the model's downstream
forward pass — the function that takes `h` at layer `ℓ_m` in place of
the model's natural activation `h(a, b)` and propagates through layers
`ℓ_m + 1, …, L` and the unembedding map, producing logits at the
answer position. Then

```
LD(h; a, b)  =  logit^final_{τ(s)}(forward_{ℓ_m → L}(h; a, b))
              -  max_{t ≠ τ(s)} logit^final_t(forward_{ℓ_m → L}(h; a, b))
```

Critically, `LD(h; a, b)` is well-defined for *any* `h ∈ R^{d_m}`, not
only for the natural activation `h(a, b)`. We write `LD(h)` when the
dependence on `(a, b)` is clear from context. This is the standard
activation-patching formalism (Wang–Variengien 2022; Conmy et al. 2023).

Higher `LD` means more correct. `LD > 0` means the model would predict
`a + b` (the correct answer). `LD < 0` means it would predict something
else.

For wrong activations `h_w`, the original `LD(h_w) < 0`. For correct
activations `h_c`, the original `LD(h_c) > 0`. Patching changes `LD`,
and the change is the causal effect.

#### 3.9.3 Three intervention families (the full causal design)

The causal claim has three components, each with its own intervention.

**Necessity (Intervention N).** Project wrong activations onto `M̂` (zero
out their off-manifold residual). If the off-manifold residual is causally
responsible for the failure, behavioral metric should improve. Concretely:

```
Intervention N:  h_w  ↦  Patch(h_w, V_perp, α=1, δ=0)
                       =  π_{M̂}(h_w)  (project onto manifold)
```

where `V_perp = M̂^⊥` is the full orthogonal complement of `M̂`.

**Localized necessity (Intervention N-V).** Same as N but restricted to
the localized subspace `V` from Theorem 2's analysis. Tests whether the
*localized* part of the residual is the causal driver, not just the full
off-manifold component.

```
Intervention N-V:  h_w  ↦  Patch(h_w, V, α=1, δ=0)
```

**Sufficiency (Intervention S).** Inject the wrong-population's mean
perturbation `μ_ξ` into a correct activation. If `μ_ξ` is the causal
direction of failure, this should *break* the correct activation's
behavior (drive `LD` from positive to negative).

First, we estimate `μ_ξ` from the wrong population:

```
μ̂_ξ  =  P_V · [(1/n_w) · Σ r(h_w)  -  (1/n_c) · Σ r(h_c)]
```

This is the localized residual mean difference. Then:

```
Intervention S:  h_c  ↦  Patch(h_c, V, α=1, δ=μ̂_ξ)
```

**Specificity (Intervention R).** Random-subspace baseline. Replace `V`
with a random subspace `V_random` of matched dimension `r`, draw a
random `δ_random` of matched norm, and apply the same patches. If the
behavioral effect of N-V and S is real and not generic, the random
patches should produce smaller effects than the targeted patches.

#### 3.9.4 Effect-size measurement and statistical testing

For each intervention `I ∈ {N, N-V, S, R}`, we compute the average
causal effect on the relevant population:

```
ACE_N    =  E_{h_w} [ LD(Patch(h_w, M̂^⊥, 1, 0)) - LD(h_w) ]
ACE_NV   =  E_{h_w} [ LD(Patch(h_w, V, 1, 0)) - LD(h_w) ]
ACE_S    =  E_{h_c} [ LD(Patch(h_c, V, 1, μ̂_ξ)) - LD(h_c) ]
ACE_R    =  E_{h_c} [ LD(Patch(h_c, V_random, 1, δ_random)) - LD(h_c) ]
```

(Note `ACE_S` is signed: a successful sufficiency intervention drives
`LD` *down* on correct activations, so `ACE_S < 0` is the desired outcome.
We report `-ACE_S` for visual symmetry with `ACE_NV > 0`.)

Bootstrap 95% CIs per condition over 1000 resamples. Permutation
calibration of significance: shuffle labels for `ACE_NV` vs `ACE_R`,
recompute, get null distribution.

Pre-registered significance criteria (BH-FDR at q = 0.05 across the
four ACEs):

- `ACE_N` is significantly positive (necessity of off-manifold component).
- `ACE_NV` is significantly positive and at least 70% the magnitude of
  `ACE_N` (the localized component captures most of the necessary effect).
- `ACE_S` is significantly negative (sufficiency of injecting `μ̂_ξ`).
- `ACE_NV` significantly exceeds `ACE_R` (specificity to the localized
  subspace).

A positive causal result requires all four. A partial positive result
(say, necessity but not sufficiency) is reported as such with appropriate
hedging.

#### 3.9.5 The causal proposition

We can state a clean target proposition that the empirical pipeline
tests. The formal statement and proof outline are
[paper_math.md Proposition 9.3](paper_math.md). The plan version below
matches the math file.

**Proposition 4 (causal sufficiency, second-order, with Hessian
remainder).** Assume the model's logit-difference
`LD: R^{d_m} → R` (in the downstream-forward-pass sense of Definition 9.2,
paper_math.md) is twice continuously differentiable, with
operator-norm-bounded Hessian:

```
sup_{h ∈ B(h_c, ρ)}  ‖∇²_h LD(h)‖_op  ≤  L     for ρ > 2 (‖μ̂_ξ‖ + D_max^V),
```

where `D_max^V := sup_i ‖P_V h_c^(i)‖` is the maximum norm of the
V-projection of correct activations.

**The patch displacement is not just `U_V μ̂_ξ`.** The patch operation
`Patch(h_c, V, 1, μ̂_ξ) = h_c + U_V μ̂_ξ - P_V h_c` *replaces* the
V-component of `h_c` by `U_V μ̂_ξ`, so the displacement is

```
Δh_c := Patch(h_c, V, 1, μ̂_ξ) - h_c  =  U_V μ̂_ξ - P_V h_c
```

(paper_math.md Proposition 9.3). This corrects the previous draft's
displacement `Δh = U_V μ̂_ξ`, which omitted the `-P_V h_c` term and is
generally wrong unless `P_V h_c = 0` exactly.

If `V` localizes the wrong-population perturbation in the sense of
Theorem 2 (that is, `T_n^V > 0` significantly and `T_n^{V_random} ≈ 0`),
then injecting `μ̂_ξ ∈ V` into correct activations satisfies

```
|ACE_S - ACE_S^linear|  ≤  (L / 2) · E_{h_c}[ ‖Δh_c‖² ]
                        =  (L / 2) · ( ‖μ̂_ξ‖² + E‖P_V h_c‖² - 2 μ̂_ξ^T U_V^T E[P_V h_c] )
                                                                  [Hessian remainder]
```

where the first-order linear prediction is

```
ACE_S^linear  :=  E_{h_c}[ ⟨∇_h LD(h_c), Δh_c⟩ ]
              =  μ̂_ξ^T U_V^T E_{h_c}[∇_h LD(h_c)] - E_{h_c}[ h_c^T P_V ∇_h LD(h_c) ].
```

**Simplification under linear-span GM (paper_math.md).** In the
linear-span case (`M = M_S, V ⊥ M_S`), Assumption GM gives
`h_c = m_c + ε_c` with `m_c ∈ M_S`, so `P_V m_c = 0` and
`E[P_V h_c] = E[P_V ε_c] = 0` (zero-mean noise, GM(iii)). The cross
term in the displacement-norm vanishes in expectation:

```
E ‖Δh_c‖²  =  ‖μ̂_ξ‖² + E ‖P_V ε_c‖²  =  ‖μ̂_ξ‖² + r σ²    (isotropic noise)
```

The leading-order linear prediction simplifies to
`ACE_S^linear = μ̂_ξ^T U_V^T E[∇_h LD(h_c)]` when `∇_h LD(h_c)` is
uncorrelated with `ε_c` (which holds when `∇_h LD` is approximately
constant on the noise scale, i.e., when `L · σ √r ≪ ‖∇_h LD‖`).

**Magnitude lower bound: Cauchy–Schwarz is an *equality*, not a lower
bound.** Cauchy–Schwarz gives the *equality*

```
| ACE_S^linear |_leading  =  ‖μ̂_ξ‖ · ‖U_V^T E[∇_h LD(h_c)]‖ · |cos θ_{ξ, ∇}|
```

where `θ_{ξ, ∇}` is the angle between `μ̂_ξ` and
`U_V^T E[∇_h LD(h_c)]`. This is *not a useful lower bound* by itself —
Cauchy–Schwarz is sharp only at parallel vectors. To convert it into an
actual lower bound on the magnitude, we add an explicit alignment
assumption.

**Alignment Assumption (ALN, paper_math.md eq (9.3)):**

```
| ⟨μ̂_ξ, U_V^T E[∇_h LD(h_c)]⟩ |  ≥  c_0 · ‖μ̂_ξ‖ · ‖U_V^T E[∇_h LD(h_c)]‖
```

for a calibration constant `c_0 > 0`. Under (ALN), the magnitude lower
bound becomes

```
| ACE_S^linear |_leading  ≥  c_0 · ‖μ̂_ξ‖ · ‖U_V^T E[∇_h LD(h_c)]‖.
```

(ALN) is empirically testable: estimate both vectors and compute the
cosine. We pre-register reporting `cos θ̂_{ξ,∇}` alongside the test, with
threshold `c_0 = 0.3` as the alignment criterion below which the
magnitude prediction is treated as inconclusive. The previous draft
presented the Cauchy–Schwarz inequality as a lower bound without the
alignment assumption — that conflated equality with bound and is fixed
in this revision.

**Plain interpretation.** Sufficiency strength is the dot product of the
perturbation magnitude with the model's behavioral sensitivity in the
direction of the perturbation. The Hessian remainder bounds how far the
true causal effect can deviate from this first-order prediction.
The bound deviates from first-order by at most `(L/2) ‖μ̂_ξ‖²`. This
formalizes the standard mech-interp intuition that interventions move
behavior proportionally to "how much the model's output cares about
that direction."

**Proof sketch.** Apply Taylor's theorem with integral remainder to
`LD(h_c + Δh_c) - LD(h_c)` with `Δh_c = U_V μ̂_ξ - P_V h_c`:

```
LD(h_c + Δh_c) - LD(h_c)
   = ⟨∇LD(h_c), Δh_c⟩  +  ∫₀¹ (1-t) ⟨∇²LD(h_c + t Δh_c) Δh_c, Δh_c⟩ dt.
```

The squared displacement satisfies
`‖Δh_c‖² = ‖μ̂_ξ‖² + ‖P_V h_c‖² - 2 μ̂_ξ^T U_V^T P_V h_c`
(using `U_V^T P_V = U_V^T`, since `P_V = U_V U_V^T` is orthogonal
projection onto `V`). The integral remainder is bounded by
`(L/2) ‖Δh_c‖²` pointwise. Taking expectations over `h_c` yields the
Hessian-remainder statement. The magnitude relation follows from
Cauchy–Schwarz applied to `⟨μ̂_ξ, U_V^T E[∇LD]⟩` together with the
alignment assumption (ALN). □

**Smooth-max regularization for argmax flips.** `LD(h) = logit_s(h) -
max_{t ≠ s} logit_t(h)` is non-smooth at points where the argmax over
`t ≠ s` changes; the Hessian `L` is unbounded there. We replace the max
by log-sum-exp at temperature `τ`:

```
LD_τ(h) := logit_s(h) - τ · log Σ_{t ≠ s} exp(logit_t(h) / τ).
```

`LD_τ ∈ C^∞`, `LD_τ → LD` uniformly as `τ → 0⁺` on any compact set, and
the Hessian bound `L_τ` for `LD_τ` scales as `1/τ` near argmax flips.
We apply Proposition 4 to `LD_τ` for some `τ > 0` and report sensitivity
to `τ` in the appendix. The smooth-max question is one of the open
items to Barnábás (paper_math.md §11 item 8). The synthetic toy
(§0a) found a magnitude-vs-Taylor ratio of `5×` in one calibrated case,
suggesting the bound is loose but valid.

This proposition is empirically testable: estimate `‖μ̂_ξ‖`, compute
`U_V^T ∇LD(h_c)` via backprop, predict `ACE_S^linear`, compare with the
empirical `ACE_S` from Section 3.9.4. The Hessian bound `L` can be
estimated from a small batch of finite-difference evaluations.

#### 3.9.6 Per-model implementation notes (HuggingFace hook paths)

- **GPT-J 6B** (28 layers, LayerNorm). HuggingFace path:
  `transformer.h.{layer}`. We register a `forward_pre_hook` on the next
  block to *capture* the residual stream output of layer `ℓ_m`, modify it
  via `Patch(h, V, α, δ)`, and pass the modified tensor forward. KT's
  patching on GPT-J targets exactly this residual location.
- **Pythia 6.9B** (32 layers, LayerNorm-like, GPT-NeoX architecture with
  parallel attention/MLP). HuggingFace path: `gpt_neox.layers.{layer}`.
  Because attention and MLP are computed in parallel and *added* to the
  residual rather than chained, the residual stream is well-defined at
  the block input and block output. We patch at the **block input** (i.e.,
  the residual entering the parallel attention/MLP), matching KT's
  Pythia patching protocol.
- **Llama 3.1 8B** (32 layers, RMSNorm, sequential attention then MLP).
  HuggingFace path: `model.layers.{layer}`. The block structure is
  `h_{ℓ+1} = h_ℓ + Attn(RMSNorm(h_ℓ))`, then `h' = h_{ℓ+1} + MLP(RMSNorm(h_{ℓ+1}))`.
  We patch `h_{ℓ+1}` directly (block output before the next block's input).
  The RMSNorm inside the next block re-normalizes our perturbation; this
  scales perturbations by a layer-specific factor that we measure and
  report (small effect: RMSNorm is a per-sample rescale, so a fixed
  V-direction injection still acts in the same direction post-norm).
  KT successfully patched Llama; we follow.

For all three models, we use bf16 weights on Babel A100 80 GB; FP32
accumulation for hook arithmetic to avoid precision loss in the patch
operation. Activation cache is fp32 (npz), residuals only at the
equals-sign token; per-(model, layer) cache size is
`~5000 × 4096 × 4 bytes ≈ 80 MB`, total 1 GB across three models × four
candidate layers. Per [babel_execution_plan.md §2](babel_execution_plan.md),
the cache lives at
`/data/user_data/$USER/blackbox/activations/{model}/layer_{ℓ}.npz`.

#### 3.9.7 Compute budget for the causal experiments

For each model, with `n_c ≈ 4500` correct and `n_w ≈ 500` wrong samples
(approximate from KT's reported accuracies), and four interventions per
sample:

- Necessity (N) on wrong: 500 × 1 forward pass = 500.
- Localized necessity (N-V) on wrong: 500.
- Sufficiency (S) on correct: 4500 (one per correct sample, takes the
  whole correct set since causal effects need power).
- Specificity (R) on correct: 4500 with random subspaces.

Total: 10000 forward passes per model × 3 models = 30000 forward passes.
On A6000-class hardware with batch size 32, this is ~3 hours per model.
Negligible compared to data generation.

#### 3.9.8 What the causal evidence shows

If all four ACEs satisfy the pre-registered criteria, we have established:

- The off-manifold residual is *causally necessary* for the failure
  (intervention N).
- The localized subspace `V` carries most of that causal necessity
  (N-V vs N).
- The localized subspace is *causally sufficient* to produce the failure
  (intervention S).
- The localization is *specific* to `V` and not generic (R baseline).

This is the strongest causal evidence available without a full circuit
analysis (which is out of scope for an 8-page workshop paper). It moves
the paper from a goodness-of-fit test to a mechanistic claim that
identifies a specific subspace `V` whose contents drive failure.

The Llama-specific finding (per KT Figure 23, Llama's algorithm fit is
weakest, so we expect the causal effects to be most informative there)
is the headline empirical result.

### 3.10 Synthetic Theorem 2 power-curve validation

Generate 1000 synthetic helices in `R^{4096}` matching the empirical helix
parameters (operand 1's helix from Method 1 fit on GPT-J as the canonical
shape). Vary:

- Perturbation rank `r ∈ {1, 2, 5, 10, 50}`.
- Perturbation magnitude `Δ ∈ {0.1, 0.3, 1.0, 3.0}`.
- Sample size `n ∈ {100, 500, 2000}`.

For each cell, simulate 100 trials. Empirical rejection rate at `α = 0.05`
becomes the empirical power. Plot empirical power vs theoretical lower bound
from Theorem 2.

Hypothesis: empirical power tracks theoretical bound to within a factor of
2 (allowing for the universal constant `C` in Theorem 2).

---

## 4. The five-method comparison, presented in the paper

This is the centerpiece methodology section. We present all five M̂
estimators side by side and argue for the parametric primary based on three
criteria.

### 4.1 Criterion 1: theoretical guarantee

**Parametric (Method 1).** Operator-norm Wedin (Stewart-Sun 1990
Theorem 3.6, paper_math.md Lemma 6.4) applied to OLS regression.
Clean finite-sample bound:
`P[ sin θ_max ≤ C_3 σ √(κ^B) / (√(λ_min^B) · σ_K(C*)) · √(d_m log(d_m/δ)/n_c) ] ≥ 1 - δ`
(paper_math.md Theorem 6.2). Standard linear regression theory plus
matrix Bernstein for `(B^T B)^{-1}` (Tropp 2015). The previous draft
cited Yu-Wang-Samworth Davis-Kahan, which is the Frobenius form and
incurs an extra `√K` factor; operator-norm Wedin avoids this.

**Diffusion Maps (Method 4).** Operator-convergence theorem from
Coifman-Lafon for the embedding step; standard regression theory for the
lift. Composing the two does not give a textbook published bound, but the
ingredients are standard.

**Kernel PCA (Method 5).** Schölkopf-Smola-Müller consistency for the
embedding; regression for the lift. Same composition issue as Method 4.

**PCA on class means (Method 2).** Bias-limited. Davis-Kahan applies but
the bias from binning does not vanish in `n`.

**Local PCA (Method 3).** Singer-Wu 2012 give a rate, but only under a
small-curvature assumption that fails for sufficiently curved manifolds.
Our toy demonstrates the failure mode.

**Conclusion.** Method 1 has the cleanest published bound. Methods 4 and 5
have ingredients but no clean composition. Methods 2 and 3 fail.

### 4.2 Criterion 2: empirical performance under controlled noise

We will reproduce the toy experiment from
`/mnt/user-data/outputs/five_methods_comparison.py` in the paper appendix
and report the convergence table. The headline finding goes in the body
as Figure 2:

```
Method                      n=100   n=500   n=1000  n=4000   Behavior
---------------------------------------------------------------------------
1. Parametric helix         4.03    1.46    1.15    0.64     1/√n
2. PCA on class means       14.02   6.60    7.05    6.31     saturates
3. Local PCA                8.13    7.50    84.40   --       breaks
4. Diffusion Maps + reg     8.16    3.28    2.35    1.27     1/√n
5. Kernel PCA + reg         8.17    3.26    2.35    1.27     1/√n
```

Methods 1, 4, 5 all scale correctly. Method 2 saturates (bias). Method 3
breaks (curvature).

### 4.3 Criterion 3: agreement on real data

For each model and layer, compute the principal angles between the five
`M̂`s on real correct activations. If the methods agree (small angles
between them), the manifold-sufficiency claim is robust. If they disagree,
the disagreement itself is a finding.

Pre-registered hypothesis: methods 1, 4, 5 agree to within 5° on real
LLM data. Method 2 will be 5-10° off due to binning bias. Method 3 will
either agree on small `n` or break on large `n`.

### 4.4 Why parametric wins for the paper

Three reasons.

First, it has the cleanest theory (Section 4.1). A reviewer asking "what
is the rate for your manifold estimator" gets a one-line answer with a
textbook citation.

Second, it matches KT's own protocol. We are not introducing a novel
manifold-fitting method; we are using KT's exact basis and asking
the next question they did not ask.

Third, the empirical performance is best (Section 4.2). 0.64° at n=4000
is the smallest of any method.

Methods 4 and 5 serve as cross-validation. Methods 2 and 3 are reported as
honest characterizations of common alternatives.

### 4.5 Why we report the negative results

Three reasons.

First, reviewers will ask "did you try the standard manifold methods?"
We can answer: yes, we tried Local PCA and PCA on class means, here is
a synthetic counterexample showing why each fails. This pre-empts the
concern.

Second, the negative results are themselves a contribution to mech-interp.
Practitioners often reach for these methods without examining their
assumptions. We tell the field which methods to avoid in this setting.

Third, the comparison gives the paper methodological depth. A paper that
uses one method has weaker methodology than a paper that compares five.

---

## 5. Limitations, owned

### 5.1 Assumptions of Theorem 1 that may not hold in practice

**Isotropic noise.** The clean form of Theorem 1 assumes
`Σ_ε = σ² I_{d_m}`. Real activation noise is anisotropic. The
generalization replaces `2 k σ⁴` by `2 tr((P^N Σ_ε)²)` and uses the
*effective rank* `k_eff = tr(P^N Σ_ε)² / tr((P^N Σ_ε)²)` in place of
`k`; for strongly anisotropic noise `k_eff ≪ k` and the bounds are
sharper, not weaker. (See [paper_math.md Remark 4.10](paper_math.md).)
Theorems 2 and 3 have not yet been propagated through the anisotropic
generalization — that is open question 9 to Barnábás
(paper_math.md §11).

**Same noise distribution across populations.** We assume
`ε_c ∼_d ε_w`. If `Σ_ε^c ≠ Σ_ε^w`, the trace terms in `E[T_n]` no
longer cancel and we pick up `tr(P^N (Σ_ε^w - Σ_ε^c) P^N)`, bounded by
`k · ‖Σ_ε^w - Σ_ε^c‖_op` (paper_math.md Remark 4.9). This is a real
concern for very different populations, but for correct vs wrong on the
same task, the noise structure should be similar. We test this
assumption empirically by comparing residual covariance matrices between
correct and wrong populations (under the null of no perturbation, the
matrices should match).

**Perturbation orthogonal to tangent.** We assume `ξ ⊥ T_p M`. Tangent
components of `ξ` are absorbed into a different value of `m_c(a, b)`
via the Riemannian exponential map of `M` at `m_c(a, b)`, so this is a
labeling convention rather than a substantive assumption (the WLOG
re-centering of paper_math.md Lemma 3.2). The second-order error
`‖ξ̃ - P^N ξ‖_2 ≤ (1/2) κ_max ‖P^T ξ‖_2²` (note: *quadratic* in the
tangent component, not linear) is absorbed into the linearization
remainder `R_1 = O(σ² κ_max²)`. The exponential-map calculation in
Lemma 3.2 is open question 1 to Barnábás (paper_math.md §11);
confirmation that the rate is `(1/2) κ_max ‖P^T ξ‖²` (second-order in
displacement) is sought.

### 5.2 Assumptions of Theorem 2 that may not hold

**Pre-registered V.** Theorem 2's bound assumes `V` is fixed in advance,
not data-dependent. We pre-register V1-V5 in Section 3.8 to satisfy this.
Post-hoc subspace selection inflates Type I error; we use permutation
calibration of `c_α` which is robust to data-dependent `V` selection.

**`V ⊥ T_p M` at typical points.** If `V` overlaps `M`'s tangent, the
test is biased toward rejecting under both null and alternative. We
mitigate by pre-projecting `V` onto `M̂`'s orthogonal complement before
the test. This makes V-selection slightly indirect: the practitioner
specifies a direction of interest, we project it off the manifold first,
then run the test on the projected V.

### 5.3 Assumptions of Theorem 3 (parametric)

**The basis is correct.** Theorem 3 (well-specified version) assumes
the OLS regression is exact, that is, the manifold has the form
`E[H_c | B] = B (C*)^T`. If the true manifold is more complex than
KT's helix, the OLS estimate is biased; the misspecification radius
`b²(M) := inf_C E[‖m_c(a, b) - B(a) C^T‖²]` does not vanish in `n_c`.
The misspecification-aware bound (Section 2.5.1, paper_math.md
Theorem 6.5) gives `sin θ_max ≤ b(M)/σ_K(C*) + (variance term)`, so
the bias term is what makes the parametric `M̂` *not* converge to `M`
even with infinite data when the parametric class is wrong.

We address this empirically by computing the OLS `R²` and reporting it.
If `R²` is below the pre-registered 90% threshold (Section 3.6), we
report the parametric `M̂` as misspecified and switch to Diffusion Maps
as primary. This is a real possibility for Llama 3.1 8B, where KT
Figure 23 documents the weaker last-token helix fit.

**Independent samples.** We assume the `n_c` correct activations are
i.i.d. Distinct addition problems are mostly independent, but there is
some weak dependence through model parameters. We use the standard
i.i.d. approximation and report a sensitivity check (block bootstrap).

### 5.4 Limitations of the experimental scope

**Only two-digit addition.** We do not test on three-digit addition or
on non-addition tasks. KT studied only two-digit addition; we match.
Multi-digit and other arithmetic are deferred to future work.

**Only three models.** The KT models. Other models (e.g., GPT-2, smaller
Pythias, Mistral, instruction-tuned variants) are not tested. The theorems
are model-agnostic; the empirical scope is constrained.

**Causal analysis is single-layer.** Section 3.9's full causal pipeline
operates at the chosen analysis layer per model. We do not trace the
mechanism through subsequent layers (a full circuit analysis), nor do we
identify which attention heads or MLPs read from the localized subspace
`V`. This is mechanism *localization*, not full circuit *discovery*. A
full circuit analysis (à la Wang-Variengien 2022) is the natural
follow-up at ICLR 2027.

### 5.5 Threats to the headline interpretation

If `T_n > 0` we conclude wrong activations are off the manifold. Three
alternative explanations.

**Sampling bias.** If the wrong population is systematically different
from correct in a way that has nothing to do with manifold structure
(e.g., wrong population concentrates near specific operand magnitudes),
`T_n` could pick up that bias. We mitigate by stratifying the wrong
population to match the correct population's marginal distribution over
operand magnitudes, and reporting both stratified and unstratified `T_n`.

**Manifold misspecification.** If `M̂` is misspecified (Section 5.3),
even on-manifold wrong activations could appear off-manifold to `M̂`.
Mitigation: require Methods 1, 4, 5 to agree before declaring `T_n`
significant.

**Tokenizer artifacts.** Different tokenizations of "wrong" outputs (e.g.,
multi-token wrong answers) might land in different residual-stream regions
than single-token correct outputs. Mitigation: pre-filter to single-token
sums in all three models (Section 3.2).

---

## 6. Pre-registration document

The exact analyses we will run, locked before extracting any activations.

### 6.1 Datasets

- 10,000 unique two-digit addition problems (all ordered pairs
  `(a, b) ∈ {0, ..., 99}²`). Tokenizer-retained subset per model is
  reported empirically in Appendix H, with no estimates in the main
  text.
- Per-model correct/wrong split based on greedy decoding at temperature 0.

### 6.2 Statistical tests, locked

For each model:

1. Compute `M̂` via Methods 1-5.
2. Compute pairwise principal angles between `M̂`s. Report.
3. Choose primary `M̂` (Method 1 if R² > 90%, else Method 4).
4. Compute `T_n` and bootstrap CI for primary `M̂`. Report.
5. Compute `T_n^{V_k}` for k = 1, ..., 5. Apply BH FDR at q = 0.05. Report.
6. Identify top-V (the V_k with largest significant `T_n^V`).
7. Estimate `μ̂_ξ` from wrong-population residuals on top-V.
8. Run Intervention N: project all wrong activations onto `M̂` and measure
   ACE_N (mean LD shift on wrong samples).
9. Run Intervention N-V: zero out the top-V component of wrong activations
   and measure ACE_NV.
10. Run Intervention S: inject `μ̂_ξ` into all correct activations and
    measure ACE_S.
11. Run Intervention R: 100 random-subspace baseline patches matched to V
    in dimension. Compute ACE_R distribution.
12. Compute bootstrap 95% CIs for all four ACEs over 1000 resamples.
13. Apply BH FDR at q = 0.05 jointly across {ACE_N, ACE_NV, ACE_S, ACE_R}.
14. Compute the Proposition 4 prediction for ACE_S using `‖μ̂_ξ‖` and
    `U_V^T ∇_h LD(h_c)` (computed via PyTorch autograd), and compare with
    measured ACE_S.
15. Record all results in pre-registration spreadsheet.

### 6.3 Cross-model story

After per-model analyses, compare:

- Which `V_k` is most informative in each model.
- Whether Llama 3.1 8B shows distinct localization compared to GPT-J and
  Pythia.
- Whether `T_n` magnitudes differ across models.

Pre-registered prediction: Llama 3.1 8B will show stronger off-manifold
behavior on the wrong population, consistent with KT's Figure 23 showing
weaker last-token helix fit. The localization should reveal a model-specific
direction.

### 6.4 What constitutes a positive result

We will report a positive result if at least 2 of the 3 models show:

- Significant `T_n > 0` after permutation calibration.
- Significant `T_n^{V_k}` for at least one pre-registered V (k ∈ 1..3) with
  BH-corrected p < 0.05.
- All four causal ACEs (N, N-V, S, R) satisfy the pre-registered criteria
  from Section 3.9.4: ACE_N significantly positive, ACE_NV ≥ 0.7 × ACE_N,
  ACE_S significantly negative, ACE_NV significantly exceeds ACE_R.

Partial positive results are reported with explicit hedging:

- Correlational + localization but no causal: "geometric divergence is
  observable but does not appear to drive behavior in the layer we tested."
- Correlational + necessity but no sufficiency: "the off-manifold component
  is necessary for failure but injecting `μ̂_ξ` is not sufficient to break
  correct behavior; the failure mechanism may require additional context."
- All correlational + causal but Proposition 4 prediction fails: "causal
  effects exist but do not track the first-order Taylor prediction;
  higher-order or non-additive mechanisms are likely involved."

If no model shows a significant `T_n`, the headline becomes: "wrong
activations sit on the same manifold as correct activations across all
three KT models, suggesting the model's algorithm is geometrically intact
and errors are purely on-manifold position errors." This is also
publishable and informative.

### 6.5 What we will not do

- Cherry-pick layer or model. The layer-selection protocol is fixed in
  Section 3.6.
- Run additional models if the three KT models give unclear results. The
  scope is locked.
- Adjust the V_k subspaces after seeing results. They are locked in 3.8.

---

## 7. The claims chain (this is the spine the paper rests on)

In order, with what supports each:

1. **There is a manifold M.** Supported by KT's Figures 5 and 19 plus our
   empirical re-verification on all three models.
2. **We can estimate M from data with controlled error.** Supported by
   Theorem 3 (parametric) plus the five-method comparison empirical study.
3. **The off-manifold test T_n detects perturbations.** Supported by
   Theorem 1 (validity) plus synthetic experiments verifying the prediction.
4. **The localization test T_n^V identifies failure subspaces.** Supported
   by Theorem 2 (power) plus synthetic power-curve experiments.
5. **Real wrong-population activations on at least one of GPT-J, Pythia,
   Llama 3.1 8B exhibit non-trivial T_n.** Supported by Section 3.7
   experiments. (This is what we expect to find.)
6. **The localization recovers an interpretable failure direction.**
   Supported by Section 3.8 experiments.
7. **The localized subspace V is causally necessary for failure.**
   Supported by Section 3.9 Intervention N-V (removal patch on wrong
   samples). Pre-registered: ACE_NV ≥ 0.7 × ACE_N (localized component
   captures most of necessity).
8. **The localized subspace V is causally sufficient for failure.**
   Supported by Section 3.9 Intervention S (injection of `μ̂_ξ` into
   correct samples). Pre-registered: ACE_S < 0 significantly.
9. **The localization is specific to V, not a generic perturbation.**
   Supported by Section 3.9 Intervention R (matched random-subspace
   baseline). Pre-registered: ACE_NV > ACE_R significantly.
10. **Quantitative agreement with Proposition 4.** ACE_S magnitude tracks
    `‖μ̂_ξ‖ · ‖U_V^T ∇_h LD‖`, validating the first-order Taylor model.

Each claim is supported by both theory and empirics. Each claim is
falsifiable. The fallback structure:

- If step 5 fails (no off-manifold drift detected anywhere), the paper
  publishes with the message "the helix is sufficient on these models for
  these errors; failures are on-manifold position errors."
- If step 6 fails (localization is null but T_n significant), the paper
  publishes with the message "wrong activations are off-manifold but in
  diffuse directions, suggesting the failure is not subspace-localizable."
- If steps 7-9 fail (causal interventions null), the paper publishes
  with the message "geometric divergence is observable but not causal,
  suggesting an epiphenomenon downstream of the actual failure mechanism."
  The theorems still hold.
- If step 10 fails (causal effects exist but don't track the predicted
  magnitudes), Proposition 4 is reported as falsified, but the qualitative
  causal claims remain.

Falsifiability of each step makes the paper robust to any single negative
result. The full positive result is the strongest possible outcome:
correlational test fires, localizes, and the localization is causally
necessary, sufficient, and specific.

---

## 8. Reviewer-defense matrix

For each anticipated reviewer concern, the pre-loaded answer.

### 8.1 "This is just a goodness-of-fit test on a manifold."

Yes, structurally Theorem 1 is. The contribution is not the test object but
(a) the application to the manifold-sufficiency question in mech-interp,
(b) the explicit power against directional alternatives in Theorem 2,
(c) the localization machinery, and (d) the empirical demonstration on
three mid-sized open-weight transformer language models. Frame Theorem 2 as the headline.

### 8.2 "Why not distance correlation, HSIC, or kernel two-sample tests?"

Distance correlation and HSIC test for *general dependence* between two
variables. Kernel two-sample tests (MMD, Gretton et al. 2012) test whether
two *distributions* are equal. We test whether two populations live on the
same *submanifold*, which is a structural geometric question, not a
distributional or dependence question. The closest existing test is
Cuevas-Fraiman 2010 for set-valued data; we cite and differentiate.

### 8.3 "The isotropic-noise assumption is unrealistic."

Theorem 1 is stated cleanly under isotropic noise; Section 5.1 gives the
anisotropic extension with `‖Σ_ε‖_op` replacing `σ²`. The empirical
experiments use real activations directly; the theoretical bound becomes a
guideline rather than a sharp prediction.

### 8.4 "You only test on three models on one task."

The theorems are model-agnostic and task-agnostic. We test on the same
three models KT studied to enable direct comparison. We test only on
two-digit addition because KT studied two-digit addition. Future work
covers more.

### 8.5 "KT did causal patching; what is your causal story?"

We run a full four-intervention causal pipeline (Section 3.9): necessity
(remove off-manifold residual), localized necessity (remove the residual
within `V` only), sufficiency (inject `μ̂_ξ` into correct activations to
break them), and specificity (matched random-subspace baseline). All
four ACEs are reported with bootstrap CIs. We further state Proposition 4
formally connecting the causal-effect magnitude to the perturbation norm
and the model's behavioral gradient. This is the strongest causal evidence
available without a full circuit analysis (which is out of scope for an
8-page workshop paper). The causal pipeline is in the body of the paper,
not the appendix, because it is load-bearing for the mechanistic claim.

### 8.6 "Manifold construction depends on M̂'s estimation error."

We propagate the error explicitly via Theorem 3 + composition (Section 2.7).
Total error in `T_n^{M̂}` is bounded by Theorem 1 noise plus Theorem 3
manifold-error, both at `1/√n`. Stated as a corollary.

### 8.7 "The pre-registered V_k subspaces are cherry-picked."

Pre-registered before any wrong-population analysis (Section 3.8). FDR-
corrected across 5 V's. V4 is a random-direction baseline, V5 is the
unprojected test. Permutation calibration of `c_α` is robust to V-selection.

### 8.8 "What if the localization just recovers the helix?"

V1 (carry), V2 (T=2), V3 (higher-order Fourier) are explicitly *not* the
helix's main 9 dimensions. V1 is functionally distinct. V2 is in the helix
but the *fragile* component KT's Figure 12 identifies. V3 is outside the
helix entirely. If the test localizes to V5 (the full orthogonal complement,
i.e., not within any pre-registered direction), we report it as such.

### 8.9 "Why two-digit addition? KT did some three-digit work."

Two-digit gives single-token answers in all three models (Section 3.2),
making the test cleaner. KT's most rigorous experiments are two-digit. We
match. Multi-digit extension is in appendix.

### 8.10 "How is this different from other geometric mech-interp papers?"

Bricken-Templeton et al. on superposition: distinct: they study how features
share dimensions, we test where wrong cases sit relative to a known manifold.
Park-Veitch-Choe LRH: they formalize linear representation; we propose a
sufficiency test that includes nonlinear (helix) representations. Engels
2025 on circular features: they identify which features are circular, we
test whether wrong cases stay on those circles. We cite all three and
differentiate.

### 8.11 "Why these three models? Why not GPT-2 small, smaller models?"

We restrict to KT's models for direct comparability with their published
results. GPT-2 small was not studied by KT. Future work covers smaller
models.

### 8.12 "Llama 3.1 8B's helix fit is weak; how do you handle that?"

Section 3.6's layer-selection protocol picks the layer where the parametric
helix R² is highest. For Llama 3.1 8B, this might be a different layer than
GPT-J or Pythia (KT's Figure 23 suggests layer 16 vs final layers). We use
the per-model best layer; this is principled.

If the parametric R² is below 90% even at the best layer (likely for Llama),
Section 5.3's protocol switches to Diffusion Maps as the primary M̂. This
is pre-registered. The paper then reports both versions of the test.

### 8.13 "The math is dense. What is the contribution to mech-interp?"

The paper provides a formal way to ask "did the model's geometry break,
or did it just compute the wrong number on intact geometry?" KT's helix is
beautiful but only for correct cases. We extend it to a falsifiable test
on wrong cases. The output is a tool any mech-interp researcher can use:
plug in your own M̂ (a learned probe, an SAE feature, an attention subspace),
run T_n and T_n^V, get a quantitative answer with explicit statistical
guarantees.

### 8.14 "Wrong samples are systematically harder than correct samples; T_n could be confounded by difficulty distribution."

This is the most important methodological concern. We address it directly
via Section 3.3's matched permutation test. The headline statistic
`T_n^matched` is computed under a *conditional null*: samples within each
arithmetic-feature bin (sum, carry, magnitudes, tokenizer class) have
correct/wrong labels permuted independently. This blocks the confound at
the source. We report all three versions (naive, cross-fitted, matched +
cross-fitted) so readers see how much of the apparent effect each
correction removes.

### 8.15 "Your correct-residual baseline is artificially small because you fit M̂ on the same data."

Addressed via K-fold cross-fitting (Section 3.4) and a stricter 4-way
data split that separates fit-M̂ / identify-V_k / calibrate-correct-baseline
/ test-on-wrong. The cross-fitted statistic is unbiased for the correct-
residual distribution. We compare the cross-fitted statistic against the
single-fit statistic in the appendix to show the magnitude of the
correction.

### 8.16 "Your manifold definition is inconsistent: sometimes 1D curve, sometimes 9D subspace."

Addressed via Section 1.5: we formally distinguish the helix curve `M_C`
(1D) from the helix span `M_S` (m-dim ambient subspace, with `m = 9`
for continuous inputs and `m = 8` for integer-only inputs after the
`T = 2` degeneracy fix from paper_math.md Remark 2.2). We define
`T_curve` and `T_span` separately and report both, yielding three
structurally distinct failure modes (on-curve in-span, off-curve in-span,
off-span). This is now a feature of the paper, not a confusion.

### 8.17 "The 5,050-sample claim looks like (100·101)/2, not a tokenizer count."

Correct, and we no longer claim 5,050 as a tokenizer-derived count. The
primary dataset is 10,000 ordered pairs. Per-model retained counts after
tokenizer audit are reported empirically in Appendix H, with no estimates
in the main text. The 5,050 unordered subset appears only as a symmetry
robustness check in Appendix G.

### 8.18 "Theorem 2's rate has factor-r vs factor-r² ambiguity."

We have re-derived Theorem 2 with a single consistent achievable rate of
`O(r σ⁴ / Δ⁴ · log(1/β))`, linear (not quadratic) in `r`. The
corresponding sample-complexity reduction over the unprojected test is
`(d_m - dim(M)) / r`, e.g., `~4088×` for `d_m = 4096, dim(M_S) = 8`
(with the `T = 2` identifiability correction from
[paper_math.md Remark 2.2](paper_math.md)) and `r = 1`. We do not
claim quadratic speedups.

**Status of the lower bound.** We prove only the *two-point Le Cam*
lower bound at rate `σ²/Δ²` ([paper_math.md Theorem 5.2(b), eq (5.2a)](paper_math.md)):
any test with size `α` and uniform power `1 - β` over the alternative
class requires `n ≥ C_2^LC · σ²/Δ² · log(1/(β(1-α)))`. This proven
bound does *not* match the achievable `r σ⁴/Δ⁴` rate; it is weaker by
a factor of `r σ² / Δ²`. The conjectured matching rate (5.2b),
`n ≥ C_2 · r σ⁴/Δ⁴ · log(...)`, would close the gap and is the
standard Ingster-style minimax separation rate established for the
Gaussian sequence model (Ingster 1993, 2003; Baraud 2002;
Collier–Comminges–Tsybakov 2017) via a chi-squared mixture over a
packing of `S^{r-1}(Δ)`. We do *not* prove it in this submission; we
treat it as a conjecture pending follow-up work
([paper_math.md §5.6 and §11 item 4](paper_math.md)). An earlier
draft sketched a "Fano chaining" argument that would have given the
matching rate; on closer reading the sketch was incoherent and has
been removed. The BlackboxNLP submission presents only (b) as proven
and (b') as conjectural, with the achievability bound (a) as the
operationally relevant rate.

### 8.19 "Theorem 1's concentration bound looks dimension-free, but the residual lives in a k-dimensional space."

Correct, this was an error in the previous draft. Theorem 1's
concentration is now stated with explicit `k = d_m - dim(M)` in two
equivalent forms:

```
[Bernstein-style]   P[ |T_n - E[T_n]| > t ]  ≤  4 · exp(-c · n · min(t² / (k σ⁴), t / σ²))
[Laurent–Massart]   P[ T_n - E[T_n] ≥ 2 σ² √(2 k u / n) + 2 σ² u / n ]  ≤  2 e^{-u}
```

with `c ≥ 1/8` (the LM sub-exponential chi-square constant). The factor
4 (vs. 2 in the previous draft) is the union bound across upper/lower
tails for both populations. The chi-squared variance of `‖P^N ε‖² ∼
σ² · χ²_k` is `2 k σ⁴`. We also present the dimension-free normalized
form `T̄_n = T_n / k` for cross-model comparison where `k` differs.
Full statement: [paper_math.md Theorem 4.2(c)–(d)](paper_math.md).

### 8.20 "Your matched-permutation feature vector includes operands exactly; bins will have one sample each."

Correct, this was a critical bug in the previous draft. We have replaced
the operand-exact `φ` with a coarse binning scheme using sum bins,
carry pattern, operand deciles, and answer-token class. The expected
samples per non-empty cell are 30-50 correct and 3-5 wrong, supporting
within-bin permutation. We pre-register a minimum-cell-count threshold
(`n_w^bin ≥ 2`) and a fallback to even coarser bins if the primary
scheme leaves too few cells. We also report regression-residualization
and propensity-reweighting variants as a multi-method robustness check.

### 8.21 "Is your manifold a 1D answer helix or a higher-dimensional joint manifold?"

We test both, plus an intermediate operand-union version (Section 3.5).
Three pre-registered manifold parameterizations: answer helix
`M_S^answer`, operand-union `M_S^union`, and joint arithmetic manifold
`M_S^joint`. Selection rule: use the simplest manifold achieving R² ≥
0.9; report all three in the appendix. Disagreement between manifolds
is itself an interpretable result (which component breaks for wrong
cases).

### 8.22 "You're calling these 'production LLMs'; that's overclaiming."

Wording softened to "mid-sized open-weight transformer language models"
throughout. GPT-J and Pythia are research models; Llama 3.1 8B is from
a deployed family but at a small scale. The "open-weight" framing is
accurate and avoids inflation.

---

## 9. The paper's section-by-section content (8 pages, empirics-first)

The reviewer feedback is correct that BlackboxNLP archival rewards results
upfront and proofs in appendix. We restructure the paper accordingly:
results are introduced after methodology (page 5 onward), with the
theoretical machinery condensed into a single theorem box on page 3-4
and full proofs in appendix.

### 9.1 Section 1: Introduction (1.0 page)

- Paragraph 1: KT 2025's discovery of the helix and the Clock algorithm,
  including the unanswered question (KT Section 5.5: composition step
  unlocalized; Figure 23: Llama's algorithm fit weakest).
- Paragraph 2: Three failure modes (on-curve, off-curve in-span, off-span)
  with one-sentence interpretation per mode.
- Paragraph 3: Our contribution as a localized geometric diagnostic with
  causal validation, with the three methodological safeguards
  (cross-fitting, matched permutation, pre-registered subspaces).
- Paragraph 4: Roadmap.

Figure 1: schematic showing the three failure modes (on-curve, off-curve
in-span, off-span) with example correct and wrong activations color-coded.

### 9.2 Section 2: Related Work (0.4 page)

- Mechanistic interpretability of arithmetic: KT 2025, Nanda et al. 2023,
  Stolfo et al. 2023, Hanna et al. 2023. We extend KT's framework with
  wrong-case analysis using a curve/span decomposition they did not consider.
- Linear, circular, and manifold representations: Park-Veitch-Choe 2024,
  Engels 2025, Gurnee-Tegmark 2024, Mikolov 2013. We test sufficiency of
  representations rather than identify them.
- Statistical foundations: Wasserman 2018 (TDA two-sample), Cuevas-Fraiman
  2010 (set-valued data), Chernozhukov 2018 (cross-fitting), Tropp 2015
  (matrix concentration). We adapt these to manifold-sufficiency.

### 9.3 Section 3: Theoretical Framework (1.5 pages, condensed from 2.5)

The reviewer rightly flags that 2.5 pages of theory in an empirical paper
is too much. We condense to a single theorem box plus interpretive
discussion. Full proofs go to appendix.

- 3.1 Setup, notation, three failure modes (curve vs span). (0.4 page)
- 3.2 **Main theorem (boxed):** Under the normal-bundle perturbation
  model with K-fold cross-fitting and matched permutation, the localized
  residual statistic `T_n^V` is unbiased for perturbation energy in
  subspace `V`, has variance scaling with `dim(V)` not ambient dimension,
  and admits a finite-sample concentration bound at rate `1/√n`. (0.5
  page; box plus one paragraph of interpretation)
- 3.3 The causal connection (Proposition 4): localized perturbation
  norm × behavioral gradient lower-bounds the causal effect. (0.3 page)
- 3.4 Pointers to appendix for full proofs of Theorems 1, 2, 3 and
  Proposition 4. (0.3 page summary; cite Sections A-D of appendix)

### 9.4 Section 4: Methodology (2.0 pages)

- 4.1 Three models, tokenizer audit, exact data counts (no estimates). (0.4 page)
- 4.2 Cross-fitting and 4-way data splitting protocol. (0.3 page)
- 4.3 Difficulty stratification and matched permutation. (0.3 page)
- 4.4 Layer selection on a held-out fold. (0.15 page)
- 4.5 Five M̂ methods (figure + short discussion; convergence table in
  appendix). (0.3 page)
- 4.6 Pre-registered failure subspaces V_1..V_5. (0.25 page)
- 4.7 Causal pipeline: definitions of N, N-V, S, R interventions; logit-
  difference metric; Proposition 4 prediction. (0.3 page)

### 9.5 Section 5: Results (2.6 pages)

- 5.1 Manifold construction agreement (Figure 2: 5×5 principal-angle matrix).
  (0.4 page)
- 5.2 Three-version test statistic table (Figure 3: T_n^naive vs T_n^cross
  vs T_n^matched per model). The headline number is in this section.
  (0.5 page)
- 5.3 Curve vs span decomposition: per-model failure-mode classification
  (Table 1: which mode dominates per model). (0.4 page)
- 5.4 Localization across V_1..V_5 (Figure 4: T_n^V_k per model). (0.4 page)
- 5.5 Causal analysis: ACE_N, ACE_NV, ACE_S, ACE_R per model
  (Figure 5: bar chart with bootstrap CIs). (0.5 page)
- 5.6 Quantitative match to Proposition 4 (Figure 6: predicted vs measured
  ACE_S scatter). (0.4 page)

### 9.6 Section 6: Discussion (0.4 page)

What we learned. Llama-specific findings (the most novel part). The three-
mode classification of arithmetic-failure mechanisms. One paragraph on
SPD metric adaptation as future work (ICLR 2027 hook).

### 9.7 Section 7: Limitations (0.2 page)

Two-digit only, three models only, single-layer causal analysis. Honest
scope statement.

### 9.8 References (0.9 page)

### 9.9 Appendix (no length limit)

- A. Full proof of Theorem 1 (validity & concentration) with the
  anisotropic Hanson–Wright extension (Theorem 4.10).
- B. Full proof of Theorem 2: achievability (a) at rate `r σ⁴/Δ⁴`,
  proven two-point Le Cam lower bound (b) at rate `σ²/Δ²`, exact null
  (c). The matching minimax `r σ⁴/Δ⁴` lower bound (b') is presented
  as a conjecture with a pointer to Ingster's chi-squared mixture
  argument; not proven in this submission.
- C. Full proof of Theorem 3 (parametric M̂ recovery via operator-norm
  Wedin / Stewart–Sun 1990).
- D. Full proof of Proposition 4 with first-order Taylor analysis,
  the corrected displacement `Δh = U_V μ̂_ξ - P_V h_c`, and the
  Alignment Assumption (ALN) for the magnitude lower bound.
- E. Diffusion Maps and Kernel PCA recovery (informal rate analysis).
- F. Manifold-construction sensitivity ablation across `(model, layer)`.
- G. Multi-token sum robustness check (10,000 → 5,050 unordered subset).
- H. Tokenization audit details and exact retained counts.
- I. KT prompt-template comparison.
- J. Per-layer test statistic across all layers in all 3 models.
- K. Synthetic power-curve full results validating Theorem 2.
- L. Cross-fitting vs naive baseline (full comparison table).
- M. Datasheet for the addition benchmark.
- N. Reproducibility checklist.

This structure puts results on page 5 onward (4.5 pages of results +
discussion + limitations), proofs in appendix, and headline T_n table
on page 5. Reviewer concern about empirics-first is addressed.

---

## 10. The complete claims diagram

```
   KT 2025                              Our paper
   ─────────                            ─────────
   Helix structure for                  
   correct cases  ─────►  given M ────► Theorem 3 (M̂ recovery, parametric)
                                              │
                                              ▼
                                        Theorem 1 (T_n validity)
                                              │
                                              ▼
                                        Theorem 2 (T_n^V localization)
                                              │
                          ┌───────────────────┴───────────────┐
                          ▼                                   ▼
                  Empirical T_n on                  Empirical T_n^V on
                  correct/wrong split               pre-reg subspaces
                  in 3 models                       across 3 models
                          │                                   │
                          ▼                                   ▼
                  Cross-model story                   Localization story
                  (Llama is the                       (which V wins?)
                   inconvenient case)                          │
                          │                                   ▼
                          └─────────────────────►   Proposition 4 +
                                                    Causal pipeline (Sec 3.9):
                                                       N (necessity)
                                                       N-V (localized necessity)
                                                       S (sufficiency)
                                                       R (specificity)
                                                              │
                                                              ▼
                                                    Mechanistic claim:
                                                    V is causally necessary,
                                                    sufficient, and specific
                                                              │
                                                              ▼
                                                    ICLR 2027: SPD metric
                                                    adaptation for full recovery
                                                    (separate paper)
```

---

## 11. Why this paper exists, and why it should be accepted

### 11.1 The clean version of the contribution

Kantamneni and Tegmark made a beautiful empirical discovery about how three
language models represent and compute addition. Their work is, in their own
words, a forward characterization: under correct computation, integers live
on a helix and addition uses a Clock algorithm. They acknowledge two open
questions: where the algorithm composition step lives, and why Llama 3.1 8B
fits less cleanly than GPT-J or Pythia.

We close both questions, partially, with a test. The test takes any candidate
manifold and any candidate failure subspace and quantifies whether wrong
activations lie off the manifold and within the subspace, with explicit
finite-sample statistical guarantees and proven power against directional
alternatives.

The paper is *internally consistent and falsifiable* in the following
specific sense, and this design property is what we use to evaluate the
research direction (it is *not* claimed in the paper text itself).
Every claim has both a theoretical proof and an empirical experiment.
Every assumption is stated and either defended or noted as a limitation.
Every method choice is compared against alternatives and justified on
three criteria (theory, empirics, agreement). Every pre-registered
analysis is locked before any result. The headline result has a fallback
— if the test fails to detect off-manifold behavior, we report that as a
positive finding for KT's forward picture. The paper publishes regardless
of which way the experiment runs.

When writing the actual submission, we replace any internal phrase like
"reviewer-proof" with claim-level language: "we design the study so that
each interpretive claim is tied to a falsifiable statistical test and a
corresponding negative interpretation." That is the form for the paper text.

### 11.2 Why the format works for BlackboxNLP

BlackboxNLP rewards: real models, real tasks, cross-population contrast,
causal evidence, mechanistic localization, reusable methodology, math
rigor. We provide all seven.

BlackboxNLP penalizes: pure theory with synthetic-only validation, cookbook
methods without depth, replications without novel contribution. We avoid
all three.

### 11.3 What this paper contributes that nothing else does

1. To our knowledge, the first application of finite-sample manifold-
   residual testing with explicit power-against-directional-alternatives
   guarantees to localized failure analysis in language-model
   representations.
2. A formal three-mode failure decomposition (on-curve in-span, off-curve
   in-span, off-span) that distinguishes structurally different mechanisms.
3. Proposition 4 connecting the localized perturbation magnitude to a
   measurable behavioral causal effect via a first-order Taylor model,
   making the causal pipeline quantitatively predictive rather than just
   qualitative.
4. A direct comparison of five manifold-recovery methods on synthetic
   ground truth, with an explicit recommendation grounded in published
   theoretical results.
5. The first cross-model wrong-population analysis on KT's three models
   with full causal validation (to our knowledge), addressing the
   localization gap KT explicitly identified in Section 5.5 of their
   paper.
6. A four-intervention causal design (necessity, localized necessity,
   sufficiency, specificity) that converts the geometric finding into
   a mechanistic claim with reusable methodology.
7. Methodological safeguards (cross-fitting and stratified matched
   permutation) that pre-empt the two main threats to validity:
   overfitting bias and difficulty confounding.
8. A Llama-specific finding (assuming the experiment fires) addressing
   why Llama 3.1 8B's algorithm fit is weakest, with both correlational
   and causal evidence.

The paper is honest, rigorous, falsifiable, and directly aligned with what
the venue rewards.

---

## 12. Summary table of every step

| Step | What we do | Why | Theorem/Method |
|---|---|---|---|
| 1 | Define `M`, `r(h) = h - π_M(h)`, `T_n` | Formalize off-manifold drift | Section 1.3 |
| 2 | Distinguish curve `M_C` from span `M_S`; define `T_curve`, `T_span` | Three failure modes (on-curve, off-curve in-span, off-span) | Section 1.5 |
| 3 | State Assumption GM | Generative model for proofs | Section 1.4 |
| 4 | Prove Theorem 1 (validity) | Test reads ‖μ_ξ‖² + tr(Σ_ξ) | Section 2.1-2.2 |
| 5 | Prove Theorem 2 (localization power) | Pre-registered V → r-rate detection (linear in r, not r²) | Section 2.3-2.4 |
| 6 | Prove Theorem 3 (M̂ recovery, parametric) | Bound principal-angle error | Section 2.5 |
| 7 | Empirically compare 5 estimators | Justify Method 1 as primary | Section 2.6 |
| 8 | Compose Theorems 1 + 3 | Full finite-sample bound for sample-estimated M̂ | Section 2.7 |
| 9 | State Proposition 4 (causal sufficiency) | Connect ACE to perturbation norm and behavioral gradient | Section 3.9.5 |
| 10 | Generate addition data: 10,000 ordered pairs, exact tokenizer audit | Real LLM data with no count estimates | Section 3.2 |
| 11 | Difficulty-stratified matched permutation as primary null | Address confound: wrong samples are harder | Section 3.3 |
| 12 | K-fold cross-fitting for M̂ and V identification | Address confound: overfitting bias | Section 3.4 |
| 13 | Layer selection per model via R² of helix on a held-out fold | Fair across models, no double-dipping | Section 3.6 |
| 14 | Compute M̂ via 5 methods at chosen layer with cross-fitting | Cross-method validation | Section 3.5 |
| 15 | Pairwise principal angles between M̂s | Robustness of manifold | Section 3.5 |
| 16 | Choose primary M̂ (parametric if R² > 90%, else Diffusion Maps) | Pre-registered switch | Section 5.3 |
| 17 | Compute three versions of `T_n`: naive / cross-fitted / matched + cross-fitted | Headline test is V3; V1 and V2 reported for transparency | Section 3.7 |
| 18 | Bootstrap CIs and BH-FDR | Type-I and joint-test error control | Section 3.7-3.8 |
| 19 | Compute `T_curve` and `T_span` separately | Distinguish three failure modes | Section 3.7 |
| 20 | Compute `T_n^{V_k}` for k=1..5 with cross-fitting + matched permutation | Localization | Section 3.8 |
| 21 | Define Patch operation precisely | Causal pipeline notation | Section 3.9.1 |
| 22 | Run Intervention N (necessity, full off-manifold) | ACE_N | Section 3.9.3 |
| 23 | Run Intervention N-V (localized necessity) | ACE_NV; tests V's contribution | Section 3.9.3 |
| 24 | Estimate `μ̂_ξ` from wrong-population residuals (cross-fitted) | Sufficiency target with no double-dipping | Section 3.9.3 |
| 25 | Run Intervention S (sufficiency by injecting `μ̂_ξ`) | ACE_S; tests V's mechanistic role | Section 3.9.3 |
| 26 | Run Intervention R (random-subspace baseline, matched in r and norm) | ACE_R; tests specificity | Section 3.9.3 |
| 27 | Bootstrap CIs for all four ACEs | Effect-size uncertainty | Section 3.9.4 |
| 28 | BH-FDR across the four ACEs | Joint causal claim | Section 3.9.4 |
| 29 | Compute predicted ACE_S from Proposition 4 (gradient × `‖μ̂_ξ‖`) | Quantitative theory check | Section 3.9.5 |
| 30 | Synthetic power-curve validation of Theorem 2 | Theorem 2 sanity check | Section 3.10 |
| 31 | Cross-model comparison and Llama-specific finding | Most novel result | Section 6 |
| 32 | Report any negative results honestly | Falsifiable design | Section 5 |

---

## 13. The single sentence that captures the contribution

**Plan version (used now, before results):**

> *We introduce a localized geometric diagnostic for mechanistic failure
> modes in language-model arithmetic — distinguishing on-curve, off-curve
> in-span, and off-span error geometries — prove its finite-sample
> validity, explicit power against pre-registered directional alternatives
> (Theorem 2), and quantitative connection to behavioral causal effects
> (Proposition 4), with cross-fitting and conditional null calibration
> protecting against overfitting and difficulty-confounding bias, and
> we test whether wrong-population activations on GPT-J 6B, Pythia 6.9B,
> and Llama 3.1 8B localize to a pre-registered computational subspace
> with causal effect on behavior, addressing the localization gap
> explicitly identified by Kantamneni and Tegmark 2025.*

**Paper version (used only after experiments fire as predicted):**

> *We introduce a localized geometric diagnostic for mechanistic failure
> modes in language-model arithmetic ... and demonstrate via experiments
> on GPT-J 6B, Pythia 6.9B, and Llama 3.1 8B that wrong-population
> activations localize to a specific computational subspace that is
> causally [necessary | sufficient | specific], addressing the
> localization gap...*

The strong "demonstrate ... is causally necessary, sufficient, and specific"
wording is reserved for the camera-ready and only used if the empirical
section actually supports it. Pre-results, every claim is hedged as
"we test whether." Post-results, we choose the strongest wording the
data supports — promoting "test" to "demonstrate" only for claims whose
ACE thresholds are met (Section 3.9.4 pre-registered criteria).

This is the abstract's first sentence. Every section in the paper serves
it. Note that *localization* is the lead, not the test itself: the
generic off-manifold test alone is a goodness-of-fit object and would
not by itself distinguish this work; the localization, the three-mode
decomposition, and the causal validation are what carry the contribution.

---

## 14. The full math reference for the proofs

The authoritative source for every theorem statement and proof outline
in this plan is the standalone math document
[paper_math.md](paper_math.md). This section maps each plan-level
theorem/proposition to the specific section in paper_math.md and lists
the external references each proof depends on.

### Plan ↔ paper_math.md crosswalk

| Plan reference | paper_math.md section | Object |
|---|---|---|
| Section 1.4 Assumption GM | §3, Assumption 3.1 | Generative model |
| Section 1.4 Assumption BD | §3.1, Assumption 3.4 | Boundedness |
| Section 1.4 Assumption REG | §3.2, Assumption 3.5 | Regularity |
| Section 1.4 WLOG re-centering | §3, Lemma 3.2 | Tangent–normal decomposition |
| Section 2.1–2.2 Theorem 1 | §4, Theorem 4.2 | Validity & concentration |
| Section 2.1 Theorem 1(a)–(d) | §4 Lemma 4.3 / 4.5 / 4.7 / 4.8 | Linearization, mean, sub-exp, mean conc. |
| Section 2.3–2.4 Theorem 2 | §5, Theorem 5.2 | Localization power |
| Section 2.3 Theorem 2(a) achievability | §5.5 | Upper bound |
| Section 2.3 Theorem 2(b) two-point lower (proven) | §5.6, Lemmas 5.5–5.6 | Le Cam two-point at σ²/Δ² rate |
| Section 2.3 Theorem 2(b') matching rate (conjecture) | §5.6 prose, Ingster-style | rσ⁴/Δ⁴ via chi-squared mixture (not proven) |
| Section 2.3 Theorem 2(c) exact null | §5.7 | χ² difference distribution |
| Section 2.5 Theorem 3 | §6, Theorem 6.2 | Manifold recovery (parametric) |
| Section 2.5 OLS bound | §6.3, Lemma 6.3 | Op-norm OLS concentration |
| Section 2.5 Wedin step | §6.4, Lemma 6.4 | Operator-norm Wedin |
| Section 2.5.1 misspecification | §6.5, Theorem 6.5 | Bias + variance decomposition |
| Section 2.7 composition | §6.6, Theorem 6.6 | Composed Theorems 1 + 3 |
| Section 3.3 matched permutation | §8, Theorem 8.3 | Exact conditional validity |
| Section 3.4 cross-fitting | §7, Theorem 7.2 + Corollary 7.3 | Neyman-orthogonal influence function |
| Section 3.9.5 Proposition 4 | §9.3, Proposition 9.3 | Hessian-bounded ACE_S |

### External references used in paper_math.md proofs

The proof outlines in paper_math.md cite the following external results.
The same citations appear in the appendix of the eventual paper.

**Theorem 1 (validity & concentration).**

- Federer 1959, *Curvature measures*, TAMS 93:418–491. Reach and tubular
  neighborhood projection.
- Niyogi–Smale–Weinberger 2008, *Finding the homology of submanifolds...*,
  Discrete Comput. Geom. 39(1-3):419–441. Modern restatement of the
  Federer projection bound.
- Laurent–Massart 2000, *Adaptive estimation of a quadratic functional*,
  Ann. Stat. 28(5):1302–1338. The sharp `2σ²√(2ku/n) + 2σ²u/n` chi-square
  tail bound.
- Vershynin 2018, *High-Dimensional Probability*. Sub-Gaussian and
  sub-exponential machinery.
- Wainwright 2019, *High-Dimensional Statistics*. Bernstein-MGF.
- Magnus–Neudecker 1999, Theorem 11.21. Anisotropic chi-square variance
  `2 tr((P^N Σ_ε)²)`.
- Tyurin 2010, *Doklady Mathematics* 82(2):760–762. Modern Berry–Esseen
  constant 0.4748.

**Theorem 2 (localization power).**

- Tsybakov 2009, *Introduction to Nonparametric Estimation*, Theorem 2.2.
  Le Cam two-point lemma (used for the proven (b) at rate σ²/Δ²).
- Ingster 1993; Ingster–Suslina 2003 (Springer LNS 169). Asymptotic
  minimax separation in Gaussian sequence — template for the conjectured
  matching rate (b').
- Baraud 2002, *Bernoulli* 8(5):577–606. Non-asymptotic minimax rates of
  testing in signal detection.
- Collier–Comminges–Tsybakov 2017, *Annals of Statistics* 45(3):923–958.
  Finite-sample chi-squared mixture argument.
- Romano–Wolf 2005, *Econometrica* 73(4):1237–1282. Step-down maxT for
  adaptive `V`.
- Davies 1980, *Applied Statistics* 29(3):323–333. Generalized
  chi-squared distribution (exact null distribution for `T_n^V`).

**Theorem 3 (manifold recovery).**

- Vershynin 2018, Section 4.7. OLS finite-sample concentration.
- Tropp 2015, *Foundations and Trends in ML* 8(1-2):1–230. Matrix
  Bernstein for `(B^T B / n_c)^{-1}`.
- Wedin 1972, *BIT* 12(1):99–111. Original perturbation theorem.
- Stewart–Sun 1990, *Matrix Perturbation Theory*, Theorem 3.6. Modern
  operator-norm Wedin restatement.

**Theorem 3 (other estimators) — informal rate analysis only.**

- Singer–Wu 2012, *Comm. Pure Appl. Math.* 65(8):1067–1144. Local PCA /
  vector diffusion maps.
- Coifman–Lafon 2006, *Appl. Comput. Harmonic Anal.* 21(1):5–30.
  Diffusion maps operator-norm convergence.
- Schölkopf–Smola–Müller 1998, *Neural Computation* 10(5):1299–1319.
  Kernel PCA consistency.

**Cross-fitting (Section 3.4) and matched permutation (Section 3.3).**

- Chernozhukov et al. 2018, *Econometrics J.* 21(1):C1–C68.
  Double/debiased ML, Neyman-orthogonal influence functions.
- Lehmann–Romano 2005, *Testing Statistical Hypotheses*, Theorem 15.2.1.
  Permutation-group exactness.
- Buja–Eyuboğlu 1992 / Westfall–Young 1993. Permutation calibration in
  practice. (Modern form.)
- Benjamini–Hochberg 1995. BH FDR control.

**Causal pipeline (Section 3.9) and Proposition 4.**

- Wang–Variengien–Conmy–Shlegeris–Steinhardt 2022, ICLR 2023 (IOI paper).
  Logit-difference behavioral metric.
- Vig et al. 2020 / Geiger et al. 2021. Activation patching formalization.
- Conmy et al. 2023 (ACDC). Effect-size measurement.
- (No external bound for the `L`-Hessian assumption; it is empirically
  estimated.)

**Other math references used in the plan but not in paper_math.md proofs.**

- Björck–Golub 1973. Principal angles via SVD of cross-Gram.
- Marchenko–Pastur 1967. Noise eigenvalue spectrum.
- Wasserman 2018, *Annual Review of Statistics*. TDA two-sample tests
  (cited as a non-comparable alternative).
- Cuevas–Fraiman 2010, *Bernoulli*. Set-valued two-sample (cited and
  differentiated).
- Bickel–Lindner 2008. Tangent–normal noise decomposition (cited in §1.4).
- Hein–Audibert–von Luxburg 2007. Diffusion-map kernel-bandwidth choice
  (cited in §2.6 Method 4 discussion).
- Bengio et al. 2003. Out-of-sample diffusion-map extension.
- Davis–Kahan 1970. Original Davis–Kahan (cited only for historical
  context; the actual proof uses Wedin).

### Kantamneni–Tegmark protocol references

- KT 2025 main reference (final version, attached). arXiv:2502.00873.
- Helix basis dimensions and periods: KT Equation 1, periods
  `T ∈ {2, 5, 10, 100}`.
- Activation patching protocol: KT Section 5, Appendix C.
- Llama 3.1 8B helix-fit weakness: KT Figure 23.
- Composition step unlocalized: KT Section 5.5, explicit admission.
- `T = 2` fragility on integer inputs: KT Figure 12.

---

## 15. Concluding remarks (for the plan, not the paper)

This plan has every step needed to execute the paper. Three theorems with
clean proofs. Five methods with empirical comparison and pre-registered
choice of primary. Three models with locked layer-selection and consistent
prompt protocols. Five pre-registered subspaces with FDR correction.
Lightweight causal validation. Synthetic theorem-validation experiments.
Twelve appendices. Twelve reviewer-defense entries.

If executed as written, the paper:

- Provides theoretical contributions of independent interest (Theorems 1,
  2, 3).
- Provides empirical contributions on three mid-sized open-weight transformer LLMs.
- Pre-empts the standard reviewer concerns explicitly.
- Falls back to publishable negative results gracefully if any single
  experiment fails.
- Sets up a coherent ICLR 2027 follow-up (SPD metric adaptation for
  causal recovery).

The paper has falsifiable design in the specific sense that every claim
has both a proof and an experiment, every method is compared against
alternatives, and every limitation is stated openly. This is the
structural property we plan around; in the submission text we describe
it as "each interpretive claim is tied to a falsifiable statistical
test and a corresponding negative interpretation."

---

*End of plan.*
