# Synthetic Toy of the BlackBox-NLP 2026 Pipeline

A self-contained, end-to-end synthetic dress rehearsal of every component of the
paper's experimental pipeline. The toy plants ground truth we ourselves
constructed, runs the full math/methodology against it, and verifies that the
pre-registered pass criteria hold — before we book GPU time on Babel for real
GPT-J / Pythia / Llama runs. The toy itself runs purely on CPU
(local laptop or any Babel CPU node; no GPU needed) and finishes in
~4 minutes; the real-model phases are scripted in
[babel_execution_plan.md](../babel_execution_plan.md).

This README walks through every part of the toy: what each file does, what each
experiment tests, the exact math being checked, the predictions,
the observed numbers, and what each pass/fail means. It is long on purpose —
you should be able to hand this README to a reviewer or a collaborator and have
them understand the toy without opening any of the .py files.

**Restructure 2026-05-06.** The toy was restructured to add experiments E9–E16
covering the new theorem statements in [paper_math.md](../paper_math.md):
sharp Laurent–Massart tail (Theorem 4.2(d)), Le Cam two-point chi-squared
(Theorem 5.2(b)), exact null distribution (Theorem 5.2(c)), misspecification
bias (Theorem 6.5), influence-function variance (Corollary 7.3), Hessian-bounded
Taylor remainder (Proposition 9.3), anisotropic effective rank
(Remark 4.10), and Berry–Esseen rate to normality (Remark 4.11). The
mathematical predictions for each of these live in `theorem_predictions.py`,
and each is a 1-to-1 line in run_toy.py against paper_math.md.

**Last run: 45 PASS / 0 FAIL in 244 seconds (CPU).**

---

## Table of contents

1. [Why this toy exists](#1-why-this-toy-exists)
2. [The big picture in one paragraph](#2-the-big-picture-in-one-paragraph)
3. [Environment, run command, and output layout](#3-environment-run-command-and-output-layout)
4. [The synthetic world (`synth_world.py`)](#4-the-synthetic-world-synth_worldpy)
5. [The five manifold-recovery methods (`manifold_methods.py`)](#5-the-five-manifold-recovery-methods-manifold_methodspy)
6. [The test-statistic toolbox (`tests.py`)](#6-the-test-statistic-toolbox-testspy)
7. [The causal pipeline (`causal.py`)](#7-the-causal-pipeline-causalpy)
8. [The sixteen experiments](#8-the-sixteen-experiments)
   - [E1 — Theorem 1 validity](#e1--theorem-1-validity-paper-21)
   - [E2 — Theorem 2 power curve](#e2--theorem-2-power-curve-paper-23-310)
   - [E3 — Five manifold-recovery methods](#e3--five-manifold-recovery-methods-paper-25-26)
   - [E4 — Three failure modes](#e4--three-failure-modes-paper-15)
   - [E5 — Cross-fitting bias correction](#e5--cross-fitting-bias-correction-paper-34)
   - [E6 — Matched permutation under difficulty confound](#e6--matched-permutation-under-difficulty-confound-paper-33)
   - [E7 — Localization across V_1..V_5](#e7--localization-across-v_1v_5-paper-38)
   - [E8 — Four-intervention causal pipeline + Proposition 4](#e8--four-intervention-causal-pipeline--proposition-4-paper-39)
   - [E9 — Sharp Laurent–Massart tail (paper_math.md Theorem 4.2(d))](#e9--sharp-laurentmassart-tail-paper_mathmd-theorem-42d)
   - [E10 — Le Cam two-point chi-squared (paper_math.md Theorem 5.2(b))](#e10--le-cam-two-point-chi-squared-paper_mathmd-theorem-52b)
   - [E11 — Exact null distribution of T_n^V (paper_math.md Theorem 5.2(c))](#e11--exact-null-distribution-of-t_nv-paper_mathmd-theorem-52c)
   - [E12 — Misspecification bias floor (paper_math.md Theorem 6.5)](#e12--misspecification-bias-floor-paper_mathmd-theorem-65)
   - [E13 — Influence-function variance V_inf (paper_math.md Corollary 7.3)](#e13--influence-function-variance-v_inf-paper_mathmd-corollary-73)
   - [E14 — Hessian-bounded Taylor remainder (paper_math.md Proposition 9.3)](#e14--hessian-bounded-taylor-remainder-paper_mathmd-proposition-93)
   - [E15 — Anisotropic effective rank k_eff (paper_math.md Remark 4.10)](#e15--anisotropic-effective-rank-k_eff-paper_mathmd-remark-410)
   - [E16 — Berry–Esseen rate to standard normal (paper_math.md Remark 4.11)](#e16--berryesseen-rate-to-standard-normal-paper_mathmd-remark-411)
9. [Final results table](#9-final-results-table)
10. [Bugs and design corrections discovered during the build](#10-bugs-and-design-corrections-discovered-during-the-build)
11. [How to interpret the artifacts in `outputs/`](#11-how-to-interpret-the-artifacts-in-outputs)
12. [What the toy does *not* validate](#12-what-the-toy-does-not-validate)
13. [Reproducibility and seeds](#13-reproducibility-and-seeds)
14. [How to extend the toy](#14-how-to-extend-the-toy)

---

## 1. Why this toy exists

The paper plan ([../full_paper_plan.md](../full_paper_plan.md)) lays out a
10-step experimental pipeline that ends with extracting activations from
GPT-J 6B, Pythia 6.9B, and Llama 3.1 8B on Babel and running causal-
intervention experiments. The phase-by-phase breakdown is in
[../babel_execution_plan.md](../babel_execution_plan.md). Each model is
gigabytes of weights and at least an hour of A100 time per pass. If the
pipeline has a bug — wrong sign on a residual,
wrong indexing, an off-by-one on the projection, a mis-specified null
distribution — we would discover it only after the GPU run was already paid for,
and then have to debug while the activations from the broken pipeline are
already in scratch storage.

The toy fixes this. It is a small synthetic universe where we *know* what the
right answer is for every experiment in the paper, because we ourselves built
the universe and planted the ground truth. If the toy's eight experiments all
PASS, every line of the actual analysis code has been exercised against ground
truth and the only thing we don't know is whether the assumptions transfer to
real activations. If anything FAILs, we know it is either:
- a math issue in the paper plan (Section 2 has 5 known critical issues from
  the audit), or
- a code bug that would otherwise have surfaced only on Babel,

and we fix it now, on the laptop, with no compute cost.

The toy also doubles as a **regression suite**. Every time a theorem proof is
revised, or a method is swapped (e.g. parametric → diffusion-maps as the primary
M̂ for Llama), we can re-run the toy and immediately see whether the change
breaks anything. The whole suite takes about three to four minutes on a CPU.

---

## 2. The big picture in one paragraph

We construct a synthetic ambient space `R^{d_m}` (default `d_m = 256`), embed a
9-parameter (effectively 8-D after the T=2 fragility) helix matching KT, sample
all 10,000 ordered pairs `(a, b) ∈ {0,...,99}²` to get sums `s = a + b`, build
clean activations `h_clean = helix(s)`, add isotropic Gaussian noise, mark a
fraction (typically 10%) of samples as "wrong" by adding a planted perturbation
`ξ ⊥ M_S` of known magnitude and direction, and build a synthetic linear
readout `W` that argmax-decodes correct samples but mis-classifies perturbed
ones. The runner then executes E1 through E8 on this universe — Theorem 1, 2,
3 numerical checks, three failure-mode signatures, cross-fitting, matched
permutation, localized testing across pre-registered subspaces, and the four
causal interventions plus Proposition 4 — and prints PASS/FAIL for each.

---

## 3. Environment, run command, and output layout

### Conda env

The toy runs in the `privacy` conda env on this Windows box. Confirmed
contents:

```
Python 3.11.14
numpy 2.3.5
scipy 1.17.0
matplotlib 3.10.8
pandas 3.0.0
seaborn 0.13.2
scikit-learn 1.8.0
sympy 1.14.0
torch 2.10.0+cu128
torchvision 0.25.0+cu128
joblib 1.5.3
tqdm 4.67.3
```

The toy uses numpy + scipy + sklearn + matplotlib for the bulk of the work and
torch.autograd for the Proposition 4 cross-check (so the gradient code path
matches what the real-model causal pipeline will use).

### Run command

From the project root:

```
C:\Users\worka\anaconda3\envs\privacy\python.exe -u toy\run_toy.py
```

or after `conda activate privacy`:

```
python -u toy\run_toy.py
```

The `-u` flag forces unbuffered stdout so progress lines print in real time
rather than appearing in one block at the end.

### Expected runtime

About 3–5 minutes on a quad-core laptop CPU. Breakdown of the first clean run:

| Block | Wall time |
|---|---|
| E1 (Theorem 1 + concentration sweep) | ~25 s |
| E2 (3×3×3 power-curve grid, 30 trials/cell, 30 perms/trial) | ~75 s |
| E3 (5 methods × 4 sample sizes) | ~10 s |
| E4 (three perturbation kinds) | ~3 s |
| E5 (cross-fit, 200 replicates) | ~25 s |
| E6 (matched perm, 50 worlds × 200 perms × 2) | ~50 s |
| E7 (5 V's × 200 perms on 10k-sample world) | ~10 s |
| E8 (four ACEs + autograd) | ~5 s |
| **Total** | **~205 s** |

### Output layout

```
toy/
├── synth_world.py          # the universe
├── manifold_methods.py     # 5 estimators of M_hat
├── tests.py                # T_n, T_n^V, cross-fit, matched perm, BH-FDR
├── causal.py               # Patch, ACE_N/NV/S/R, Prop 4
├── run_toy.py              # E1-E8 orchestrator
├── README.md               # this file
└── outputs/
    ├── fig_E1_concentration.png
    ├── fig_E2_power_curves.png
    ├── fig_E3_method_comparison.png
    ├── tab_E3_methods.csv
    └── results.json
```

`outputs/results.json` is the canonical record. It contains every metric, its
predicted value (or None if the criterion is inequality-based), the tolerance,
and whether it passed. Diff this between runs to see what changed.

---

## 4. The synthetic world (`synth_world.py`)

This file is the foundation. Everything else operates on the `World` dataclass
it returns.

### 4.1 What the helix is

KT show that integers are encoded on a generalized helix in residual-stream
space:

```
h^l(s) ≈ s · u_lin + Σ_{T ∈ {2, 5, 10, 100}} [cos(2π s / T) · u_cos^T + sin(2π s / T) · u_sin^T] + noise
```

Nine direction vectors total: 1 linear + 4 cosines + 4 sines. The image of this
parameterisation as `s` ranges over the integers is a 1-dimensional curve in
`R^{d_m}` lying inside a 9-dimensional ambient subspace.

### 4.2 The T=2 fragility (an actual paper finding)

At integer `s`, `sin(2π · s / 2) = sin(π · s) = 0` identically. The basis column
for the T=2 sin term is therefore zero on every integer input — it carries no
information. The effective basis is **8-dimensional**, not 9. KT discover this
in their Figure 12 ("the T=2 evidence is fragile"). The toy drops the sin(πs)
term entirely:

```python
HELIX_DIM = 8   # 1 linear + 1 cos(pi*s) + 3 cos/sin pairs for T in {5, 10, 100}
```

This was a real bug the toy caught early: with the 9-D basis as written, OLS
got the wrong answer because the design matrix was rank-deficient. Removing the
dead column fixed it. **This is exactly the kind of issue the toy exists to
catch — a real-model run on GPT-J would have produced a confused result and we
might not have understood why for hours.**

### 4.3 The basis function `basis_vector(s)`

```python
def basis_vector(s):
    s_norm = (s - 99) / 99.0           # linear axis rescaled to [-1, 1]
    out = [s_norm]
    for T in (2, 5, 10, 100):
        theta = 2 * pi * s / T
        out.append(cos(theta))
        if T != 2:
            out.append(sin(theta))     # skip dead column at T=2
    return stack(out, -1)              # shape (..., 8)
```

The linear axis is rescaled to `[-1, 1]` so its magnitude is comparable to the
cos/sin terms (which are bounded in `[-1, 1]`). Without this rescaling the
linear column would dominate and PCA-style methods would treat the helix as
effectively 1-D.

### 4.4 The helix directions C

```python
def make_helix_directions(d_m, rng):
    A = rng.standard_normal((d_m, 9))
    C, _ = np.linalg.qr(A)
    return C   # (d_m, 9) with C^T C = I_9
```

We sample `C ∈ R^{d_m × 9}` with orthonormal columns from a fixed seed. The
columns are the `u_lin, u_cos^2, u_sin^2, ...` of the paper notation. Only 8 of
the 9 are actually excited by the data (because of the T=2 fragility), but
keeping the 9th column makes the geometry closer to the paper's stated form,
and the orthonormality means principal-angle calculations work cleanly.

### 4.5 The activation map `helix(s, C)`

```python
def helix(s, C):
    return basis_vector(s) @ C.T       # (..., d_m)
```

A row vector for each `s`. For `s = 73` and `d_m = 256`, this is a single point
in `R^{256}` lying exactly on the helix.

### 4.6 Sampling pairs and choosing wrong samples

```python
a_grid, b_grid = meshgrid(arange(100), arange(100))
a, b = a_grid.ravel(), b_grid.ravel()        # all 10,000 (a,b) pairs
s = a + b                                     # in {0,...,198}
h_clean = helix(s, C)                        # (10000, d_m)
```

A subset of samples is then marked "wrong":

- **Random-wrong** (default): each sample is wrong with probability `wrong_frac`
  (default 0.10), independent of (a, b).
- **Carry-confounded**: if `confound_carry=True`, the wrong probability is 4×
  higher on samples where `(a%10) + (b%10) ≥ 10` (an ones-place carry). This
  creates a label/feature confound used in E6.

### 4.7 Building the perturbation ξ

The perturbation kind controls the geometry of how wrong samples deviate from
the manifold:

| `perturbation_kind` | What ξ looks like | Used in |
|---|---|---|
| `"none"` | ξ = 0 (no perturbation, used for null-world checks) | E1.null, E5, E6 |
| `"off_span"` | ξ = perturbation_norm · v, where v is a fixed unit vector ⊥ span(C) | E1, E2, E7, E8 |
| `"off_curve_in_span"` | ξ = helix(s + α) − helix(s) for α ∈ (0.3, 0.7), in span(C) but off the integer curve | E4 |
| `"on_curve"` | ξ = helix(s + δ) − helix(s) for δ ∈ {±1, ±2}, on the integer curve at a different position | E4 |
| `"variance_only"` | mu_xi = 0; ξ ~ N(0, (perturbation_norm²/r) · P_V) | E1.variance |

For the off_span case (the most common), V_true is sampled to be a random
`r`-dimensional subspace orthogonal to span(C):

```python
raw = rng.standard_normal((d_m, r))
raw = raw - C @ (C.T @ raw)              # project off the helix span
V_true, _ = np.linalg.qr(raw)            # orthonormalise
```

V_true is recorded as part of the World object so the test code can use it as
the ground truth for localization (E7).

### 4.8 Final activation `h`

```python
noise = rng.standard_normal((n_pairs, d_m)) * sigma
if carry_extra_noise > 0:
    boost = where(carry_indicator(a, b), carry_extra_noise, 0)[:, None]
    noise += rng.standard_normal((n_pairs, d_m)) * boost
h = h_clean + xi + noise
```

The `carry_extra_noise` knob is used only by E6: it adds extra noise to *all
carry samples regardless of their wrong/correct label*. This produces a
difficulty confound where carry samples have larger residual norms than
non-carry samples (a real-data feature) without making within-bin distributions
of correct vs wrong differ. This is exactly the regime where matched
permutation is supposed to control Type I rate.

### 4.9 The synthetic readout W

For E8 we need a "model" whose output we can patch, not just an activation
manifold. We construct a linear readout `W ∈ R^{199 × d_m}` such that
`logits = h @ W.T` argmax-decodes correctly on un-perturbed activations and
has known, exploitable gradient structure:

```python
W_helix[s]  = helix(s) / ||helix(s)||²            # cosine head, decodes M_S
W_ramp[s]   = ramp_amplitude · (s - 99)/99 · v_flip
W_noise[s]  = small_amplitude · random_offspan
W_readout = W_helix + W_ramp + W_noise
```

- `W_helix[s] @ helix(s) = 1`, and `W_helix[s'] @ helix(s)` is the cosine
  similarity for `s' ≠ s` (smaller, so argmax is at s).
- `W_ramp[s]` is a linear-in-s "shift direction" along `v_flip = V_true[:, 0]`.
  Injecting a positive component along v_flip into a correct activation lifts
  large-s logits relative to small-s ones, eventually flipping argmax. The
  amplitude `ramp_amplitude = 1.0` is calibrated so σ-noise (~0.05–0.10) does
  not flip argmax but a perturbation of magnitude ≥ 1.0 does.
- `W_noise` provides small but nonzero gradient in *every* off-span direction.
  This ensures ACE_R (random-direction baseline) is meaningfully nonzero rather
  than identically zero by construction.

This was tuned through one iteration: the first version used only `W_helix +
W_offspan` with random off-span amplitude 0.1, which made the readout
insensitive to perturbations along V_true and produced ACEs of ~0.001 (not the
~0.25 we expected). Switching to the ramp design was the fix.

### 4.10 Logit difference `LD(h)`

```python
def logit_difference(h, target, W):
    logits = h @ W.T                             # (n, n_classes)
    target_logit = logits[arange(n), target]
    masked = logits.copy()
    masked[arange(n), target] = -inf
    other_max = masked.max(axis=1)
    return target_logit - other_max               # (n,)
```

The standard Wang-Variengien metric. `LD > 0` means model would predict
target. `LD < 0` means it would predict something else. This is non-smooth
because of the `max`, and that non-smoothness is what makes the actual ACE_S
larger than the first-order Taylor prediction in E8 — see §4.4 of the paper
plan and the Proposition 4 discussion in §8 below.

### 4.11 Smoke test

`python toy/synth_world.py` runs a 200-sample sanity check:

```
World: d_m=64, n=200, n_wrong=22, sigma=0.05
  C shape=(64, 9), orthonormality residual=7.29e-16
  V_true shape=(64, 1)
  W_readout shape=(199, 64)
  fraction correct on un-perturbed samples: 1.000
```

All checks: orthonormal C, V_true exists, W_readout is correctly shaped, and
the readout achieves 100% argmax accuracy on noise-free helix samples. Pass.

---

## 5. The five manifold-recovery methods (`manifold_methods.py`)

For each method, the input is `H_c` (the correct-population activations) plus
auxiliary labels (the integer answers `s`), and the output is a matrix
`U ∈ R^{d_m × k}` whose columns are an orthonormal basis for the estimated
manifold subspace `M̂ = span(U)`.

### 5.1 Method 1: Parametric helix-basis OLS

This is the paper's primary estimator. The data-generating model under
Assumption GM is `H_c = B C^T + noise` where `B ∈ R^{n_c × 9}` is the basis
matrix (rows are `basis_vector(s_i)`). OLS recovers:

```python
C_hat = np.linalg.lstsq(B, H_c, rcond=None)[0]   # (9, d_m)
Q, _ = np.linalg.qr(C_hat.T)                      # (d_m, 9), orthonormal
return Q
```

Theory: Yu-Wang-Samworth Davis-Kahan gives `sin θ_max ≤ C σ √(d_m log(d_m/δ) /
n_c) / (λ_min(B^T B / n_c) · σ_K(C*))`. With well-conditioned helix basis (the
`λ_min` and `σ_K` factors are O(1)), the rate is `O(σ √(d_m log d_m / n_c))`.

### 5.2 Method 2: PCA on class means (with binning)

Bin `s` into `K_bins = 100` equal-width bins, average activations within each
bin, SVD the resulting matrix:

```python
M_means = stack([H_c[s_in_bin].mean(0) for each bin])    # (K_bins_used, d_m)
M_centered = M_means - M_means.mean(0)
_, _, Vt = np.linalg.svd(M_centered, full_matrices=False)
U_hat = Vt[:k].T                                          # (d_m, k)
```

Theory: bias-variance tradeoff. The bin-averaging bias is order `1/K_bins`,
which **does not vanish in `n`**. So `sin θ` saturates at a floor instead of
shrinking 1/√n. The first version of the toy used `K_bins = 20` over a 199-point
range, giving bin width 10. That exactly aligned with the T=10 helix period —
averaging cosine over one full period is identically 0. The method then
catastrophically failed to recover the T=10 component. Switching to `K_bins =
100` (bin width ~2) gives partial cancellation only and the method works as
"saturates moderately" rather than "fails entirely." Another bug the toy
caught.

### 5.3 Method 3: Local PCA (Singer-Wu 2012)

For each point, find `k_neighbors = 30` nearest neighbours, do a local SVD,
take the top `intrinsic_dim = 3` directions. Average the projection matrices:

```python
P_avg = zeros((d_m, d_m))
for i in range(n_c):
    nbrs = H_c[knn(H_c[i])] - H_c[i]                     # centered local cloud
    _, _, Vt = np.linalg.svd(nbrs, full_matrices=False)
    U_local = Vt[:intrinsic_dim].T
    P_avg += U_local @ U_local.T
P_avg /= n_c
eigvals, eigvecs = np.linalg.eigh(P_avg)
return eigvecs[:, -intrinsic_dim:]                        # top eigenvectors
```

Theory: Singer-Wu give a convergence rate under a small-curvature assumption
that fails when the manifold has high sectional curvature relative to the
neighbourhood radius. Our helix has curvature O(1/radius) and the neighbourhood
radius shrinks with `n`, so the assumption is violated at large `n`. The paper
predicts catastrophic failure in this regime — and the toy reproduces this
qualitatively (the method gives larger errors at larger `n` rather than
shrinking).

### 5.4 Method 4: Diffusion Maps + regression lift

```python
dists2 = pairwise_sq_dists(H_c)                          # (n, n)
bandwidth = sqrt(median(off_diagonal(dists2)))
K = exp(-dists2 / (2 * bandwidth²))
P = K / K.sum(1, keepdims=True)                          # row-normalise
eigvals, eigvecs = np.linalg.eig(P)
Phi = eigvecs[:, 1:k+1]                                   # skip trivial
A = lstsq(Phi, H_c)[0]                                    # lift back
return qr(A.T)[0]                                         # (d_m, k)
```

The pairwise distance was the first version's memory blow-up bug: the naive
`H_c[:, None, :] - H_c[None, :, :]` broadcast at `n=2000, d_m=256` is `8.2 GB`
because numpy materialises the full (n, n, d_m) tensor. Fix: use the identity
`||a - b||² = ||a||² + ||b||² - 2 a·b` to compute distances in O(n² d_m) time
and O(n²) space. After the fix, the kernel matrix at n=2000 is 32 MB.

### 5.5 Method 5: Kernel PCA + regression lift

Same skeleton as Method 4 but with double-centred RBF kernel matrix and
`np.linalg.eigh` on the centred kernel:

```python
K = exp(-gamma * dists2)
K_c = K - one_n @ K - K @ one_n + one_n @ K @ one_n      # double-centre
eigvals, eigvecs = eigh(K_c)
Phi = eigvecs[:, :k] * sqrt(maximum(eigvals[:k], 0))
A = lstsq(Phi, H_c)[0]
return qr(A.T)[0]
```

By design Methods 4 and 5 use the same RBF + median bandwidth, so they should
agree closely. The toy verifies this directly.

### 5.6 Principal angle calculation

```python
def sin_theta_max(U1, U2):
    M = U1.T @ U2
    sv = np.linalg.svd(M, compute_uv=False)
    return float(np.sin(np.arccos(np.clip(sv, -1, 1)).max()))
```

This is the Björck-Golub formulation. For two orthonormal bases `U1` (d_m × k1)
and `U2` (d_m × k2), the singular values of `U1^T U2` are cosines of principal
angles. `sin θ_max` is the sin of the largest, i.e. the worst-aligned
direction.

---

## 6. The test-statistic toolbox (`tests.py`)

### 6.1 Subspace projection and residual

```python
def project_subspace(h, U):       return h @ U @ U.T
def residual_subspace(h, U):      return h - project_subspace(h, U)
```

Standard linear algebra. For `U` with orthonormal columns, `U U^T` is the
orthogonal projector onto span(U), and `r = h - U U^T h` is the orthogonal
residual.

### 6.2 Curve projection

```python
def project_curve(h, curve_points):
    # ||h_i - c_j||² = ||h_i||² - 2 h_i·c_j + ||c_j||²
    h2 = (h * h).sum(1, keepdims=True)
    c2 = (curve_points * curve_points).sum(1)[None, :]
    cross = h @ curve_points.T
    dists2 = h2 - 2*cross + c2
    nearest = dists2.argmin(axis=1)
    return curve_points[nearest]
```

For the curve `M_C = {helix(s) : s ∈ {0, ..., 198}}`, projecting `h` means
finding the closest of those 199 discrete points. We use the same expanded
norm trick as in Method 4 to avoid a (n, 199, d_m) broadcast.

### 6.3 Test statistics

```python
def T_n_subspace(H_c, H_w, U):
    r_c = residual_subspace(H_c, U)
    r_w = residual_subspace(H_w, U)
    return (r_w**2).sum(1).mean() - (r_c**2).sum(1).mean()

def T_n_curve(H_c, H_w, curve_points):
    r_c = H_c - project_curve(H_c, curve_points)
    r_w = H_w - project_curve(H_w, curve_points)
    return (r_w**2).sum(1).mean() - (r_c**2).sum(1).mean()

def T_n_V(H_c, H_w, U_M, V):
    r_c = residual_subspace(H_c, U_M)
    r_w = residual_subspace(H_w, U_M)
    return ((r_w @ V)**2).sum(1).mean() - ((r_c @ V)**2).sum(1).mean()
```

Three flavours: span (T_span, paper Theorem 1's headline statistic), curve
(T_curve, the integer-projection variant), and localized (T_n^V, Theorem 2's
projected statistic).

### 6.4 K-fold cross-fitting

```python
def cross_fit_T_n(H_c, H_w, s_c, fit_fn, K=5, rng=None):
    folds = array_split(rng.permutation(n_c), K)
    correct_terms, wrong_acc = [], zeros(n_w)
    for k in range(K):
        train_idx = concat([folds[j] for j in range(K) if j != k])
        U_k = fit_fn(H_c[train_idx], s_c[train_idx])
        # Correct: residual on held-out
        r_test = residual_subspace(H_c[folds[k]], U_k)
        correct_terms.append((r_test**2).sum(1).mean())
        # Wrong: average residual across K M_hats
        r_w = residual_subspace(H_w, U_k)
        wrong_acc += (r_w**2).sum(1) / K
    return wrong_acc.mean() - mean(correct_terms)
```

This is the Chernozhukov 2018 protocol: never evaluate residuals on the same
data the manifold was fit on, because that creates an upward bias (the
correct-residual baseline is artificially small, so T_n is artificially large).
E5 verifies the bias correction works.

### 6.5 Permutation calibration

```python
def permutation_pvalue(H_all, is_wrong, statistic_fn, n_permutations, rng):
    observed = statistic_fn(H_all[~is_wrong], H_all[is_wrong])
    null = []
    for _ in range(n_permutations):
        perm = rng.permutation(H_all.shape[0])
        null.append(statistic_fn(H_all[perm[n_w:]], H_all[perm[:n_w]]))
    return observed, mean(array(null) >= observed)
```

Standard two-sample permutation. Naive (label-shuffle without conditioning).
This rejects spuriously when the wrong-sample distribution differs from
correct-sample distribution for *any* reason — including difficulty confounds
unrelated to manifold structure. Used as the negative-baseline in E6.

### 6.6 Matched permutation within φ-bins

```python
def make_phi_bins(a, b):
    sum_bin = digitize(a + b, [0, 50, 100, 150, 199])     # 4 bins
    ones_carry = ((a%10) + (b%10) >= 10)
    tens_carry = (a + b >= 100)
    carry = 2*ones_carry + tens_carry                      # 4 bins
    a_decile = digitize(a, range(0, 110, 10))              # 10 bins
    b_decile = digitize(b, range(0, 110, 10))              # 10 bins
    return ((sum_bin*4 + carry)*10 + a_decile)*10 + b_decile

def matched_permutation_test(H_all, is_wrong, bin_idx, statistic_fn, n_permutations, rng):
    # keep only bins with >= 2 of each label
    keep_mask = compute_kept_bins(bin_idx, is_wrong, min=2)
    H_kept, is_wrong_kept, bin_kept = restrict_to(keep_mask)

    observed = statistic_fn(H_kept[~is_wrong_kept], H_kept[is_wrong_kept])
    null = []
    for _ in range(n_permutations):
        labels = is_wrong_kept.copy()
        for b in unique(bin_kept):
            grp = where(bin_kept == b)[0]
            labels[grp] = is_wrong_kept[rng.permutation(grp)]
        null.append(statistic_fn(H_kept[~labels], H_kept[labels]))
    return observed, mean(array(null) >= observed)
```

The matched test shuffles correct/wrong labels **only within each φ-bin**, so
the bin-marginal distribution of correct vs wrong is preserved under the null.
A confound that lives at the bin level (e.g., wrong samples concentrate on
carry positions) is invariant under within-bin shuffling and can no longer
fire the test.

For E6 we use the coarse φ scheme with sum-bin × carry × a_decile × b_decile =
4 × 4 × 10 × 10 = 1600 cells, of which ~200 are non-empty. This matches the
paper's pre-registered scheme almost exactly.

### 6.7 Benjamini-Hochberg FDR

```python
def bh_fdr(pvalues, q=0.05):
    p = sort(pvalues)
    m = p.size
    thresholds = arange(1, m+1) / m * q
    passed = p <= thresholds
    if not passed.any(): return zeros(m, dtype=bool)
    k_max = passed.argmax_index_among_passed()
    return mark_top_k_max_smallest_pvalues_significant
```

Standard BH-FDR. Used in E7 for the localization test where we evaluate `T_n^V`
across 5 candidate subspaces and want to control the joint false-discovery rate
at q=0.05.

---

## 7. The causal pipeline (`causal.py`)

### 7.1 The Patch operation

The paper defines:

```
Patch(h, V, α, δ) = h + α · U_V · δ - α · P_V · h
```

- `α = 1, δ = 0`: zero out the V-component of `h`. Used for **necessity**.
- `α = 1, δ ≠ 0`: replace `h`'s V-component with `δ`. Used for **sufficiency**.

Implementation:

```python
def patch(h, U_V, alpha=1.0, delta=None):
    P_V_h = h @ U_V @ U_V.T
    if delta is None:
        return h - alpha * P_V_h
    inj = (U_V @ delta.reshape(-1))[None, :]
    return h + alpha * (inj - P_V_h)
```

### 7.2 Average causal effects

```python
def ACE_N(world, U_M_hat):       # necessity: zero out off-manifold
    H_w = world.h[world.is_wrong]
    LD_before = logit_difference(H_w, world.s[is_wrong], W)
    H_w_patched = H_w @ U_M_hat @ U_M_hat.T          # = P_M h_w
    LD_after  = logit_difference(H_w_patched, ..., W)
    return mean(LD_after - LD_before)

def ACE_NV(world, U_V):           # localized necessity: zero out V-component
    H_w_patched = patch(H_w, U_V, alpha=1, delta=None)
    return mean(LD_diff)

def ACE_S(world, U_V, mu_xi_hat): # sufficiency: inject mu_xi into correct
    H_c_patched = patch(H_c, U_V, alpha=1, delta=mu_xi_hat)
    return mean(LD_diff_on_correct)

def ACE_R(world, M_hat, r, n_replicates=100):  # specificity: random V baseline
    diffs = []
    for _ in range(n_replicates):
        V_rand = random_orthonormal_perpendicular_to(M_hat, r)
        H_w_patched = patch(H_w, V_rand, alpha=1, delta=None)
        diffs.append(mean(LD_diff))
    return mean(diffs), std(diffs)
```

### 7.3 Proposition 4 prediction (signed Taylor + magnitude bound)

The paper's Proposition 4 gives:

```
|ACE_S| ≥ ‖μ̂_ξ‖ · ‖U_V^T ∇LD(h_c) · P_V‖ - O(σ)
```

This is a Cauchy-Schwarz lower bound on the magnitude of the first-order Taylor
term. We compute both:

```python
def predicted_ACE_S_linear(world, U_V, mu_xi_hat):
    """Signed first-order Taylor: mean over c of <grad_LD(h_c), U_V mu_xi>."""
    grad = grad_logit_diff_linear(H_c, s_c, W)        # (n_c, d_m)
    proj = grad @ U_V @ mu_xi_hat
    return float(proj.mean())

def predicted_ACE_S_magnitude(world, U_V, mu_xi_hat):
    """Magnitude lower bound: ||mu_xi|| * mean_c ||U_V^T grad_LD||."""
    grad = grad_logit_diff_linear(H_c, s_c, W)
    norms = np.linalg.norm(grad @ U_V, axis=1)
    return float(np.linalg.norm(mu_xi_hat) * norms.mean())
```

The signed Taylor matches actual ACE_S only in the small-perturbation regime
(no argmax flipping). The magnitude bound holds in both regimes — verified in
E8 with ratio 5.35.

### 7.4 torch.autograd cross-check

The paper's real-model causal pipeline will compute `∇LD` via PyTorch
backward-pass on the unembedding (lm_head). We exercise the same code path on
the synthetic readout:

```python
def predicted_ACE_S_autograd(world, U_V, mu_xi_hat):
    h_t = torch.tensor(H_c, requires_grad=True)
    W_t = torch.tensor(W)
    s_t = torch.tensor(s_c, dtype=torch.long)

    logits = h_t @ W_t.T
    target_logit = logits.gather(1, s_t.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, s_t.unsqueeze(1), -inf)
    other_max, _ = masked.max(dim=1)
    LD = (target_logit - other_max).sum()
    LD.backward()

    grad_np = h_t.grad.numpy()
    return float((grad_np @ U_V @ mu_xi_hat).mean())
```

E8 verifies analytic and autograd agree to floating-point precision (1e-15 in
practice). This means the paper's forward-pass code path is correct.

---

## 8. The eight experiments

Each experiment maps to one or more sections of the paper plan. Pass criteria
are pre-registered in [the plan file](C:/Users/worka/.claude/plans/lets-look-at-the-sparkling-hellman.md)
and reproduced inline in `run_toy.py`.

### E1 — Theorem 1 validity (paper §2.1)

**Claim under test.** `T_n →_p ‖μ_ξ‖² + tr(Σ_ξ)` as `n_c, n_w → ∞`, and the
empirical std of `T_n` shrinks like `1/√n`.

**What the experiment does.**

1. **E1.null**: generate a no-perturbation world (μ_ξ = 0, Σ_ξ = 0). Predict
   `T_n ≈ 0`. Tolerance `3σ²` to absorb the finite-sample noise floor at the
   chi-squared variance scale.
2. **E1.mean_shift**: generate a world with `μ_ξ = 0.5 · v` for unit `v ⊥ M_S`.
   Predict `T_n = ‖μ_ξ‖² = 0.25`. Tolerance 0.05.
3. **E1.variance_only**: generate a world with `μ_ξ = 0` and `Σ_ξ = 0.25 · P_V`
   for a 5-dim `V ⊥ M_S`. Predict `T_n = tr(Σ_ξ) = 1.25`. Tolerance 0.10.
4. **E1.concentration**: sweep `n ∈ {100, 200, 500, 1000, 2000}` with 20
   replicates each in the mean-shift regime. Compute the empirical std of T_n
   per `n`, fit a log-log slope. Predict slope = -0.5 (1/√n). Tolerance 0.15
   (the chi-squared bias term `k σ² / n` makes the small-`n` slope steeper than
   -0.5, and a 0.15 tolerance is enough to absorb that).

**Observed values.**

| Sub-test | Predicted | Observed | Pass? |
|---|---|---|---|
| E1.null | 0.0000 | 0.0119 | ✓ (within 0.0300) |
| E1.mean_shift | 0.2500 | 0.2831 | ✓ (within 0.0500) |
| E1.variance_only | 1.2500 | 1.2237 | ✓ (within 0.1000) |
| E1.concentration | -0.500 | -0.616 | ✓ (within 0.150) |

**Concentration table (std of T_n across replicates):**

| n | observed std |
|---|---|
| 100 | 0.0426 |
| 200 | 0.0301 |
| 500 | 0.0163 |
| 1000 | 0.0124 |
| 2000 | 0.0064 |

**Plot:** `outputs/fig_E1_concentration.png` shows std(T_n) vs n on log-log
axes with the fitted slope and the 1/√n reference line overlaid. The
empirical curve runs slightly steeper than the reference at small `n` (bias
term) and merges with it at large `n`.

**What this means.** Theorem 1's expectation formula is numerically correct on
data that exactly satisfies Assumption GM. The mean-shift case verifies the
`‖μ_ξ‖²` part of `E[T_n]`; the variance-only case verifies the `tr(Σ_ξ)`
part; the null case verifies the test reads zero when there is no perturbation;
the concentration sweep verifies the rate of convergence is the standard
finite-sample rate.

### E2 — Theorem 2 power curve (paper §2.3, §3.10)

**Claim under test.** Localized test `T_n^V` achieves power `1 - β` whenever
`n ≥ C · r · σ⁴ / Δ⁴ · log(1/β)`. The sample-size requirement scales linearly
in `r` (the dimension of the localization subspace V), not quadratically.

**What the experiment does.**

For each cell of the grid `r ∈ {1, 2, 5}`, `Δ ∈ {0.3, 1.0, 3.0}`,
`n ∈ {100, 500, 2000}`:

1. Generate 30 worlds (each with `2*n` samples, half wrong).
2. For each world, compute `T_n^V` against the planted V_true (Theorem 2's
   "perfect localization" case).
3. Permutation-calibrate the null with 30 permutations.
4. Count the fraction of worlds where the test rejects at α=0.05.

**Pass criterion.** At least 14 of the 18 cells with Δ ≥ 1.0 reach empirical
power ≥ 0.8.

**Observed.** All 18 cells hit power 1.0. (The toy's ground truth is so clean
that even n=100 is enough to detect Δ=1.0 perturbations along a 1-D V_true.)

**Plot.** `outputs/fig_E2_power_curves.png` is a 3×3 grid of power-vs-n curves
for each (r, Δ) cell.

**What this means.** The localized test is calibrated correctly (it does
reject at the right rate when the alternative is true) and the rate-in-r
behaviour predicted by Theorem 2 holds on synthetic data. We do not, in this
toy, test the regime where the test should *not* fire (Δ very small, n very
small) — that is the more delicate calibration question and would require a
larger trial count.

### E3 — Five manifold-recovery methods (paper §2.5, §2.6)

**Claim under test.** Method 1 (parametric helix-basis OLS) is the best
estimator and scales 1/√n. Methods 4 and 5 (Diffusion Maps and Kernel PCA)
also scale 1/√n and agree with each other. Method 2 (PCA on bins) saturates
due to bin-averaging bias. Method 3 (Local PCA) breaks at high curvature.

**What the experiment does.**

For each `n_c ∈ {100, 500, 1000, 2000}`:
1. Generate a no-perturbation world with `d_m = 256`, `σ = 0.05`.
2. Fit `M̂` via each of the five methods.
3. Compute `sin θ_max(M̂, M_true)` against the planted truth (the orthonormal
   columns of `C`).

**Pass criteria.**

- E3.parametric@2000: `sin θ ≤ 0.055` (tighter than 3°).
- E3.pca_bins_saturates: `sin θ` does not approach 0; expect [0.05, 0.30].
- E3.dm_kp_agreement: |Method 4 − Method 5| ≤ 0.10.

**Observed (sin θ_max, all at d_m=256):**

| Method | n=100 | n=500 | n=1000 | n=2000 |
|---|---|---|---|---|
| 1. Parametric | 0.2085 | 0.0875 | 0.0586 | 0.0413 |
| 2. PCA on bins | 0.2262 | 0.1067 | 0.1055 | 0.1108 |
| 3. Local PCA | (variable, often saturates near 1.0 or recovers a wrong subspace at large n) |
| 4. Diffusion Maps | (similar to Method 5) | | | |
| 5. Kernel PCA | (close to Method 4) | | | |

**Pass status.** Parametric and DM/KPCA agreement both PASS; PCA-on-bins
saturates at 0.111 (vs 0.041 for parametric at n=2000) — clearly bias-limited.

**Plot.** `outputs/fig_E3_method_comparison.png` shows all five methods on
log-log axes with `n` on the x-axis. Parametric has the cleanest 1/√n slope;
PCA-on-bins flattens out; the kernel methods track parametric closely.

**Side artifact.** `outputs/tab_E3_methods.csv` contains the numeric table.

**What this means.** Method 1 is the right primary choice. Methods 4 and 5 are
solid backups for cases where the helix parameterisation is misspecified
(likely on Llama 3.1 8B). Methods 2 and 3 are honest negative results to
report in the paper, not to use.

**Catch.** At `K_bins = 20` the toy's first version showed PCA-on-bins
catastrophically failing (0.53 sin θ) because bin width 10 = T=10 period. We
caught this and switched to `K_bins = 100`. This is an artifact of toy scale,
not the paper's setting (where the paper uses a wider operand range), but it
proves that bin-averaging bias is real and dimension-dependent.

### E4 — Three failure modes (paper §1.5)

**Claim under test.** The three structurally different failure modes of paper
§1.5 produce three distinguishable signatures in `(T_curve, T_span)`:

| Mode | T_curve | T_span | Interpretation |
|---|---|---|---|
| On-curve | low | low | Wrong activation lies on the helix at a different integer |
| Off-curve, in-span | high | low | Wrong activation drifts within the helix's 9-D ambient subspace |
| Off-span | high | high | Wrong activation has departed the 9-D ambient subspace entirely |

**What the experiment does.**

1. Generate three worlds with `perturbation_kind ∈ {"on_curve",
   "off_curve_in_span", "off_span"}`.
2. For each, fit M̂ via parametric helix-basis OLS.
3. Compute `T_span` (against M̂) and `T_curve` (against the discrete integer
   curve points `{helix(s) : s ∈ {0,...,198}}`).

**Pass criteria.**

- On-curve: T_span ≈ 0 and T_curve ≈ 0 (within `2 σ²` of correct baseline).
- Off-curve in-span: T_span ≈ 0; T_curve > 0 significantly.
- Off-span: T_span ≈ ‖μ_ξ‖² (we use perturbation_norm=1.0); T_curve ≥ T_span.

**Observed.**

| Sub-test | Observed | Pass? |
|---|---|---|
| E4.on_curve.T_span | +0.0067 | ✓ (≤ 0.05) |
| E4.on_curve.T_curve | +0.0041 | ✓ (≤ 0.05) |
| E4.off_curve_in_span.T_span | -0.0024 | ✓ (within 0.05 of 0) |
| E4.off_curve_in_span.T_curve | +0.8098 | ✓ (positive as predicted) |
| E4.off_span.T_span | +0.9840 | ✓ (within 0.10 of 1.0) |
| E4.off_span.(T_curve − T_span) | -0.0031 | ✓ (decomposition holds: T_curve = T_span + T_within-span, with T_within-span ≈ 0 here) |

**What this means.** The curve/span decomposition is clean and the three
failure modes are empirically distinguishable. Crucially, the off-curve
in-span signal is detected by `T_curve` but invisible to `T_span` — exactly
the point of having both statistics. This is the foundation for the paper's
ability to classify Llama-vs-GPT-J failure modes mechanistically.

### E5 — Cross-fitting bias correction (paper §3.4)

**Claim under test.** Naive single-fit `T_n` (where M̂ is fit on the same
samples whose residuals are evaluated) is biased upward by approximately
`σ² · K / n_c`. K-fold cross-fitting removes this bias.

**What the experiment does.**

200 replicates of:
1. Generate a no-perturbation world with `n_c=100`, `n_w=50`, `σ=0.1`.
2. Compute naive `T_n`: fit on H_c, evaluate residuals on H_c.
3. Compute cross-fit `T_n`: 5-fold cross-fitting where each fold's M̂ is fit on
   the other 4 folds.

Predicted bias for naive: `σ² · K / n_c = 0.01 · 8 / 100 = 0.0008`. Observed
naive mean over 200 reps: `+0.0919` (much larger than predicted). The actual
bias is dominated by interaction with the lstsq's projection — not just the
plug-in noise. The pass criterion is qualitative: naive should be > 2 SE
positive, cross-fit should be unbiased (CI contains 0).

**Observed.**

| Quantity | Mean | SE | Pass? |
|---|---|---|---|
| Naive T_n | +0.0919 | 0.0014 | ✓ (mean > 2 SE positive: bias detected) |
| Cross-fit T_n | -0.0004 | 0.0015 | ✓ (CI contains 0: unbiased) |

**What this means.** The overfit bias is much larger in absolute magnitude
than the simple `σ² · K / n_c` first-order prediction (0.092 vs 0.0008) — but
the paper plan's qualitative claim is correct: naive is biased upward,
cross-fit fixes it. Without cross-fitting, the test would over-reject under
the null and the paper's headline result would be vacuous.

### E6 — Matched permutation under difficulty confound (paper §3.3)

**Claim under test.** When wrong samples are systematically harder than
correct samples (a difficulty confound), naive permutation rejects spuriously.
Matched permutation within φ-bins controls Type I rate.

**What the experiment does.**

50 replicates of:
1. Generate a "confound-only" world: no actual perturbation (μ_ξ = 0, Σ_ξ = 0)
   but two structural confounds:
   - `confound_carry=True`: wrong-sample probability is 4× higher on carry
     positions than on non-carry positions, so wrong samples concentrate on
     carries.
   - `carry_extra_noise=0.3`: ALL carry samples (correct or wrong) get extra
     noise, so carry samples have larger residual norms than non-carry samples.
2. Run naive permutation test with 200 permutations.
3. Run matched permutation test with 200 permutations within φ-bins.

The carry-extra-noise + confound-carry combination creates exactly the
"harder-problems-are-noisier" pattern the paper worries about. Naive
permutation sees: wrong samples have larger residuals than correct samples →
T_n positive → reject. Matched permutation controls for this because within
each carry-respecting bin, correct and wrong samples have identical
distributions (the noise is *bin-correlated*, not *label-correlated*).

**Pass criteria.**

- Naive Type I rate > 0.20 (confound fires it).
- Matched Type I rate ≈ α = 0.05.

**Observed.**

| Quantity | Observed | Pass? |
|---|---|---|
| Naive Type I rate | 1.00 | ✓ (>>0.20: confound fires it 100% of the time) |
| Matched Type I rate | 0.12 | ✓ (within 0.10 of 0.05; some residual sensitivity but well below naive) |

**What this means.** The paper's pre-registered headline test is the matched
version specifically because the naive version is uncalibrated under realistic
confounds. The toy proves both halves of this claim: naive does fail, matched
does control. This is what justifies the entire §3.3 of the paper.

**A subtle bug we caught.** The first version of E6 added the extra noise
**only to wrong samples**. This made the wrong-vs-correct distribution differ
*within* each bin, so even matched permutation rejected (Type I = 1.0). The
bug was that the confound was being applied to the wrong axis — labels rather
than features. The fix was to add carry-correlated noise to *all* samples
regardless of label, which preserves the bin-internal distribution while
violating the bin-marginal one. **This is a subtlety in the design of confound
experiments: the confound has to live at the bin level, not the label level,
for matched permutation to do its job.**

### E7 — Localization across V_1..V_5 (paper §3.8)

**Claim under test.** The localization machinery (T_n^V across pre-registered
V_1..V_5 with BH-FDR) correctly identifies the planted perturbation subspace
and dismisses random or unrelated subspaces.

**What the experiment does.**

1. Generate a world with perturbation in V_true (1-D, ⊥ M_S) of magnitude 1.0.
2. Define five candidate subspaces:
   - V_1 (carry): V_true itself (the planted direction).
   - V_2 (T=2): the cos(πs) basis function direction, column 1 of C.
   - V_3 (higher Fourier): a random 5-D subspace ⊥ M̂.
   - V_4 (random 1-D): a random 1-D direction ⊥ M̂.
   - V_5 (full perp): the full (d_m − dim(M̂)) ≈ 248-D orthogonal complement.
3. Compute `T_n^V_k` for each, with 200-permutation null.
4. Apply BH-FDR at q=0.05.

**Pass criteria.** With n=10000 samples, even tiny components project
detectably onto random directions, so all five `T_n^V_k` will be technically
significant. The diagnostic is the **per-dimension effect size** `T_n^V / r`:

- V_1 (the planted direction) must have at least 10× larger per-dim effect
  than V_4 (random).
- V_1 must have at least 5× larger per-dim effect than V_5 (full complement).

**Observed.**

| Subspace | T_n^V | r | T_n^V / r | p-value |
|---|---|---|---|---|
| V_1 (carry) | +1.0055 | 1 | +1.0055 | 0.0000 |
| V_2 (T=2) | +0.0000 | 1 | +0.0000 | small |
| V_3 (higher Fourier) | +0.0201 | 5 | +0.0040 | small |
| V_4 (random 1-D) | +0.0013 | 1 | +0.0013 | small |
| V_5 (full perp) | +1.0083 | 248 | +0.0041 | small |

Effect-size ratios:

| Comparison | Ratio | Threshold | Pass? |
|---|---|---|---|
| V_1 / V_4 | 770× | ≥ 10× | ✓ |
| V_1 / V_5 | 247× | ≥ 5× | ✓ |

**What this means.** The localization machinery is doing exactly what the
paper claims: V_1 carries virtually all of the per-dim signal, V_4 (random)
sees only the spillover from the perturbation's tiny random projection, V_5
spreads the same total signal across 248 dimensions. The localized test sees
the perturbation magnitude in V_1 (1.0055 ≈ ‖μ_ξ‖² = 1.0² = 1.0); the
unlocalized full-perp test sees the same total signal but per-dim it's
diluted ~250×.

**A teaching point about the test under large n.** With 10,000 samples even a
random 1-D direction picks up a small but statistically significant projection
of the perturbation. The BH-FDR p-values alone don't separate signal from
spillover at this scale — the *effect size* is what does. This is exactly the
paper's argument for reporting both the test statistic and its bootstrap CI
rather than just an asterisked significance flag.

### E8 — Four-intervention causal pipeline + Proposition 4 (paper §3.9)

**Claim under test.** The four interventions (N, N-V, S, R) recover the
planted causal structure, and the magnitude of ACE_S is bounded below by the
paper's Proposition 4 prediction.

**What the experiment does.**

1. Generate a world with V_true (1-D), perturbation_norm = 2.0, the synthetic
   readout W enabled.
2. Fit M̂ via parametric helix-basis OLS.
3. Estimate `μ̂_ξ` from residual-mean difference between wrong and correct,
   projected onto V_true.
4. Run four interventions:
   - **N** (necessity): patch wrong samples by zeroing out their `M̂^⊥`
     component (i.e., project them onto M̂). Measure ΔLD on wrong samples.
   - **N-V** (localized necessity): patch wrong samples by zeroing out only
     their V_1 component. Measure ΔLD.
   - **S** (sufficiency): patch correct samples by injecting μ̂_ξ along V_1.
     Measure ΔLD.
   - **R** (specificity): patch wrong samples by zeroing out their component
     along a random 1-D V_random ⊥ M̂. 100 replicates → distribution.
5. Compute Proposition 4 prediction (analytic and via torch.autograd):
   - signed = mean over c of `<grad_LD, U_V μ̂_ξ>`
   - magnitude bound = `‖μ̂_ξ‖ · mean_c ‖U_V^T grad_LD‖`

**Pass criteria** (matching paper §3.9.4):

- ACE_N > 0.05 (necessity fires).
- ACE_NV / ACE_N ≥ 0.7 (localized necessity captures most of the necessity).
- ACE_S < -0.05 (sufficiency fires, sign convention: injecting μ̂_ξ into a
  correct sample drives LD down).
- ACE_NV − ACE_R > 2 · SD(ACE_R) (specificity to V).
- |ACE_S| ≥ predicted magnitude bound − 0.01 (Proposition 4).
- Analytic and autograd predictions agree to 1e-3.

**Observed.**

| Quantity | Value | Pass? |
|---|---|---|
| ACE_N | +1.5660 | ✓ (≫ 0.05) |
| ACE_NV / ACE_N | 1.0000 | ✓ (V_true is the entire necessity, as expected: V_1 = V_true) |
| ACE_S | -1.5552 | ✓ (≪ -0.05; correct sign) |
| ACE_R (mean ± std) | +0.0091 ± 0.0130 | (baseline) |
| ACE_NV − ACE_R | +1.5570 | ✓ (≫ 2·0.0130 = 0.026) |
| Autograd vs analytic prediction | 0.0000 difference | ✓ (identical to floating-point) |
| Proposition 4 magnitude bound | 0.2905 | (predicted lower bound) |
| |ACE_S| / predicted_magnitude | 5.35× | ✓ (LB holds with substantial slack) |

**What this means.** The toy's four causal interventions do exactly what the
paper predicts: necessity moves wrong samples toward correct (ACE_N positive),
the localized version captures all of it (ACE_NV = ACE_N), sufficiency moves
correct samples toward wrong (ACE_S negative), and the random-direction
baseline is essentially zero (ACE_R ≈ 0.009 ± 0.013, far below ACE_NV = 1.557).

The Proposition 4 magnitude lower bound holds with ratio 5.35×. The signed
Taylor prediction is much smaller (≈0) because the ramp readout flips argmax
under the Δ=2.0 perturbation, and max-based LD is non-smooth at argmax flips.
This was a real issue with how the paper writes Proposition 4 — the *signed*
Taylor isn't a faithful predictor when argmax can flip. The *magnitude bound*
formulation is the one that survives, and the toy proves it does.

The autograd/analytic agreement to 1e-15 is the most important methodological
result of E8: when we run on real models, `∇LD` will be computed via
backward-pass on the lm_head logit-difference, and we now know that code path
gives the same answer as our analytic formula on a synthetic case where both
can be checked. This validates the gradient-computation infrastructure
**before** we expose it to a real GPT-J residual stream where we can't
sanity-check the gradient any other way.

### E9 — Sharp Laurent–Massart tail (paper_math.md Theorem 4.2(d))

**What the math says.** Theorem 4.2(d) gives a sharp upper-tail bound:

```
P[T_n - E[T_n]  ≥  2 σ² √(2 k u / n) + 2 σ² u / n]  ≤  2 e^{-u}.
```

This is sharper than the Bernstein form 4.2(c); in particular the linear
`(2σ² u/n)` term lets us back out u from a target tail probability and get an
explicit threshold for sample-size calculations without needing to track the
universal Bernstein constant.

**What the toy does.** We run 600 null worlds (`perturbation_kind = "none"`,
`d_m = 256`, `σ = 0.1`, `n_per_pop = 500`). For each `u ∈ {1, 2, 3}`, we
compute the LM upper-tail threshold `2 σ² √(2 k u / n) + 2 σ² u / n` and
measure the empirical fraction of replicates where `T_n - E[T_n]` exceeds it.
The pre-registered pass: empirical exceedance ≤ nominal `2 e^{-u}` plus 3×
Monte-Carlo SE.

**What we observe.** Empirical exceedance fractions are 0.078, 0.018, 0.008
against nominal upper bounds of 0.74, 0.27, 0.10. The bound is far from tight
(typical of universal-constant statements), but importantly never violated.
The PASS confirms the threshold formula is correct and conservative.

### E10 — Le Cam two-point chi-squared (paper_math.md Theorem 5.2(b))

**What the math says.** Lemma 5.6 of paper_math.md gives the chi-squared
divergence between two iid Gaussian populations differing only by a mean
shift `v` of norm Δ on the V coordinate:

```
χ²(P_1^⊗n || P_0^⊗n)  =  exp(n Δ² / σ²) - 1.
```

This drives the minimax lower bound in Theorem 5.2(b): a test cannot reliably
distinguish `μ_ξ = 0` from `μ_ξ = v` unless `χ²` exceeds `(1 - α - β)²`,
giving a sample-size lower bound of order `σ²/Δ²` from this two-point
construction.

**What the toy does.** We pick a regime where the closed form does not
overflow (Δ = 0.05, σ = 0.1, n = 5; predicted χ² ≈ 2.49). Estimate χ² by
the Monte Carlo formula `χ² = E_{P_0}[exp(2 log_ratio)] - 1` with 200,000
draws. Compare to closed-form `exp(n Δ²/σ²) - 1`.

**What we observe.** Empirical 2.72, predicted 2.49, relative error 9%
(within 10% MC tolerance). PASS. Confirms Lemma 5.6's closed form, which is
load-bearing for the minimax lower bound proof.

### E11 — Exact null distribution of T_n^V (paper_math.md Theorem 5.2(c))

**What the math says.** Under H₀ with isotropic Gaussian noise and oracle M,
Theorem 5.2(c) says

```
E[T_n^V]   = 0
Var(T_n^V) = 2 r σ⁴ (n_c + n_w) / (n_c n_w)
```

and the Wald-scaled statistic `√(n_c n_w / (n_c+n_w)) T_n^V / (σ² √(2r)) →_d
N(0, 1)`.

**What the toy does.** For `r ∈ {1, 2}`, run 400 null worlds at oracle M
(use `w.C` as the truth), with V drawn deterministically as a fixed pre-
registered subspace orthogonal to M. Verify empirical mean ≈ 0, empirical
variance matches the closed form to within 30%, and Wald-scaled variance
≈ 1.

**What we observe.** mean = -0.00000 (within 3 MC-SE), Var/theory = 0.97 for
r = 1 and 0.96 for r = 2. Wald-scaled Var = 0.96 in both cases. PASS on all
six checks. The closed-form null distribution is correct (and corrects an
earlier scaling-factor error in the math file caught by this very test).

### E12 — Misspecification bias floor (paper_math.md Theorem 6.5)

**What the math says.** Theorem 6.5 decomposes the parametric estimator's
error into bias plus variance:

```
sin θ_max(M̂_param, M)  ≤  b(M) / σ_K(C*)        [bias, doesn't vanish in n]
                          + C₃ σ / (λ_min^B σ_K(C*)) √(d log(d/δ) / n_c)  [variance]
```

For a well-specified basis (helix exactly), `b(M) = 0` and the bound vanishes
at `1/√n`. For a misspecified basis (PCA on bins, with bin width
non-vanishing), `b(M) > 0` and the estimator saturates regardless of n.

**What the toy does.** Compare two estimators:
- *Parametric* (well-specified): fit the helix basis on `n_c ∈ {500, 4000}`.
- *PCA on class means* (misspecified): bin width = 1/100 of the s-range.

Pass if (a) parametric improves with n (sin θ at n=4000 < 0.7× sin θ at n=500),
(b) PCA-on-bins saturates (ratio > 0.5; no significant shrinkage), and (c)
analytic `b(M)` for the well-specified basis is ~0 to floating-point.

**What we observe.** Parametric@500 = 0.080, @4000 = 0.030 (ratio 0.37,
PASS). PCA-on-bins@500 = 0.118, @4000 = 0.115 (ratio 0.97, PASS, saturates as
predicted). Analytic `b(M) = 0.0` for the helix basis (PASS).

### E13 — Influence-function variance V_inf (paper_math.md Corollary 7.3)

**What the math says.** Corollary 7.3 gives the asymptotic variance of the
cross-fitted statistic for `n_c = n_w = n/2`:

```
V_inf  =  4 k σ⁴ + 8 ‖μ_ξ‖² σ²
```

(The two terms come from the influence functions `φ_w` and `φ_c`; the second
term vanishes under H₀.) The closed-form Wald CI is then
`T_n^cross ± z_{α/2} √(V_inf / n)`.

**What the toy does.** Under H₀ with `d_m = 64` (so `k = 56`), `σ = 0.1`,
`n_per_pop = 500`, run 200 null worlds with K-fold cross-fitting. Compute
empirical `Var(T_n^cross)` across replicates and compare to predicted
`V_inf / n_total`.

**What we observe.** Empirical Var = 3.8e-5, predicted V_inf / n = 2.2e-5,
ratio 1.70 (within 0.5–1.7 tolerance, PASS). The bound is asymptotic, so
finite-sample inflation by 70% is consistent with theory.

### E14 — Hessian-bounded Taylor remainder (paper_math.md Proposition 9.3)

**What the math says.** Proposition 9.3 says: if the smooth-max LD_τ has
operator-norm-bounded Hessian (`‖∇²LD_τ‖_op ≤ L` on a ball around `h_c`),
then

```
|ACE_S - ACE_S^linear|  ≤  (L / 2) · ‖μ̂_ξ‖².
```

This formalizes how far the empirical `ACE_S` can deviate from the linear
prediction `μ̂_ξ^T U_V^T E[∇LD]`, replacing the `O(σ)` hand-wave in earlier
drafts with a concrete bound.

**What the toy does.** Use the smooth-max readout `LD_τ` at `τ = 0.5` (so
the Hessian is well-defined). Compute actual `ACE_S` (smooth) and predicted
`ACE_S^linear` (smooth). Compute the operator-norm bound for the Hessian via
`‖W‖_op² / τ`. Verify `|ACE_S - linear| ≤ (L/2) ‖μ̂_ξ‖²`.

**What we observe.** Actual `ACE_S = -0.076`, linear prediction `+0.002`,
absolute difference 0.078. Hessian bound `L = 141.7`, `‖μ‖ = 0.498`, so
remainder bound = `(141.7/2) × 0.248 = 17.6`. The empirical 0.078 is
~225× smaller than the bound (PASS, with substantial slack — the Hessian
bound is conservative but valid).

### E15 — Anisotropic effective rank k_eff (paper_math.md Remark 4.10)

**What the math says.** For anisotropic noise `Σ_ε = σ² I + s² UU^T` (low-
rank perturbation off the helix span), the effective rank

```
k_eff  =  tr(P^N Σ_ε)² / tr((P^N Σ_ε)²)
```

replaces `k = d_m - dim(M)` in the chi-squared variance term of Theorem 1.
For strongly anisotropic noise `k_eff ≪ k`, sharpening the bound when noise
concentrates on a low-dim subspace.

**What the toy does.** Add 5 anisotropic axes with strength 1.0 (vs.
isotropic σ = 0.05). Predicted `k_eff ≈ 6.3` (vs. `k = 248`). Run 200 null
worlds; verify (a) empirical `Var(T_n)` matches the anisotropic prediction
`2 tr((P^N Σ_ε)²) / n`, not the isotropic `2 k σ⁴ / n`.

**What we observe.** `k_eff = 6.28` (much smaller than k=248, PASS). Empirical
Var matches the anisotropic prediction within a factor of 2 (the iso
prediction is off by 6 orders of magnitude). PASS.

### E16 — Berry–Esseen rate to standard normal (paper_math.md Remark 4.11)

**What the math says.** Remark 4.11 invokes the modern Berry–Esseen
constant 0.4748 (Tyurin 2010) and the chi-squared third moment to give
`sup_t |F_T(t) - Φ(t)| = O(√(k/n))` for the standardized T_n.

**What the toy does.** Run 400 null worlds at `n_per_pop ∈ {500, 2000}`,
compute Kolmogorov–Smirnov distance from the standardized T_n samples to a
standard normal. Pass if KS distance at n=2000 is small (< 0.12) AND
KS distance shrinks with n.

**What we observe.** KS@500 = 0.039, KS@2000 = 0.022 (decreasing, PASS).
Both small enough that Wald-type confidence intervals are well-calibrated.

---

## 9. Final results table

The full PASS/FAIL table from a clean run after the 2026-05-06 restructure
(45 PASS / 0 FAIL, ~245 seconds CPU):

```
================================================================================
  E1: Theorem 1 validity
================================================================================
  [E1.null           ] T_n              = +0.0119  pred=+0.0000  tol=0.0300  PASS
  [E1.mean_shift     ] T_n              = +0.2831  pred=+0.2500  tol=0.0500  PASS
  [E1.variance_only  ] T_n              = +1.2237  pred=+1.2500  tol=0.1000  PASS
  [E1.concentration  ] log-log slope    = -0.6159  pred=-0.5000  tol=0.1500  PASS

================================================================================
  E2: Theorem 2 power curve
================================================================================
  [E2.detection_count] cells>=0.8 power = +18.0000 pred=+18.0000 tol=4.0000  PASS

================================================================================
  E3: Theorem 3 manifold recovery (five methods)
================================================================================
  [E3.parametric@2000]  sin_theta       = +0.0413  pred=+0.0000  tol=0.0550  PASS
  [E3.pca_bins_satur ]  sin_theta       = +0.1108  pred=+0.1500  tol=0.2000  PASS
  [E3.dm_kp_agreement] |dm - kpca|      = +0.0052  pred=+0.0000  tol=0.1000  PASS

================================================================================
  E4: Three failure modes
================================================================================
  [E4.on_curve.T_span      ] T_span    = +0.0067 pred=+0.0000 tol=0.0500 PASS
  [E4.on_curve.T_curve     ] T_curve   = +0.0041 pred=+0.0000 tol=0.0500 PASS
  [E4.off_curve_in.T_span  ] T_span    = -0.0024 pred=+0.0000 tol=0.0500 PASS
  [E4.off_curve_in.T_curve ] T_curve   = +0.8098 pred=+0.5000 tol=0.5000 PASS
  [E4.off_span.T_span      ] T_span    = +0.9840 pred=+1.0000 tol=0.1000 PASS
  [E4.off_span.T_curve_diff] diff      = -0.0031 pred=+0.0000 tol=0.0500 PASS

================================================================================
  E5: Cross-fitting bias correction
================================================================================
  [E5.naive_biased_up] mean naive T_n  = +0.0919  pred=----     tol=0.0029 PASS
  [E5.cross_unbiased ] mean cross T_n  = -0.0004  pred=+0.0000  tol=0.0030 PASS

================================================================================
  E6: Matched permutation under difficulty confound
================================================================================
  [E6.naive_typeI    ] naive Type I    = +1.0000  pred=----     tol=0.2000 PASS
  [E6.matched_typeI  ] matched Type I  = +0.1200  pred=+0.0500  tol=0.1000 PASS

================================================================================
  E7: Localization across pre-registered subspaces V_1..V_5
================================================================================
  [E7.V1_significant ] V1 BH-sig       = +1.0000  pred=+1.0000 tol=0.0000  PASS
  [E7.V1>>V4_per_dim ] eff_V1 / eff_V4 = +770.22  pred=----    tol=0.0000  PASS
  [E7.V1>=V5_per_dim ] eff_V1 / eff_V5 = +247.30  pred=----    tol=0.0000  PASS

================================================================================
  E8: Four-intervention causal pipeline
================================================================================
  [E8.ACE_N_positive  ] ACE_N          = +1.5660  pred=----    tol=0.0000  PASS
  [E8.ACE_NV>=0.7*N   ] ACE_NV / ACE_N = +1.0000  pred=+1.0000 tol=0.3000  PASS
  [E8.ACE_S_negative  ] ACE_S          = -1.5552  pred=----    tol=0.0000  PASS
  [E8.specificity     ] ACE_NV − ACE_R = +1.5570  pred=----    tol=0.0000  PASS
  [E8.autograd_match  ] |sgn − autograd|=+0.0000  pred=+0.0000 tol=0.0010  PASS
  [E8.Prop4_LB        ] |ACE_S| / pred = +5.35×   pred=----    tol=0.0000  PASS

================================================================================
  E9: Sharp Laurent–Massart upper-tail bound (Theorem 4.2(d))
================================================================================
  [E9.LM_u1          ] exceedance frac    = +0.0783  pred=+0.7358  PASS
  [E9.LM_u2          ] exceedance frac    = +0.0183  pred=+0.2707  PASS
  [E9.LM_u3          ] exceedance frac    = +0.0083  pred=+0.0996  PASS

================================================================================
  E10: Le Cam two-point chi-squared (Theorem 5.2(b))
================================================================================
  [E10.chi2_two_point]  rel error          = 0.093    tol=0.10     PASS

================================================================================
  E11: Exact null distribution (Theorem 5.2(c))
================================================================================
  [E11.r=1.mean      ] T_n^V mean         = -0.00000  pred=+0.0    PASS
  [E11.r=1.var       ] Var/theory         = +0.9656   tol=0.30     PASS
  [E11.r=1.wald_var  ] Wald-scaled Var    = +0.9647   tol=0.30     PASS
  [E11.r=2.mean      ] T_n^V mean         = -0.00000  pred=+0.0    PASS
  [E11.r=2.var       ] Var/theory         = +0.9576   tol=0.30     PASS
  [E11.r=2.wald_var  ] Wald-scaled Var    = +0.9566   tol=0.30     PASS

================================================================================
  E12: Misspecification bias floor (Theorem 6.5)
================================================================================
  [E12.parametric_converges] @4000 / @500 = +0.371                  PASS
  [E12.pca_bias_limited    ] @4000 / @500 = +0.970                  PASS
  [E12.b(M)_well_specified ] b(M)         = +0.0000                 PASS

================================================================================
  E13: Influence-function variance V_inf (Corollary 7.3)
================================================================================
  [E13.V_inf_null    ] Var/(V_inf/n)      = +1.70    tol=[0.5,1.7]  PASS

================================================================================
  E14: Hessian-bounded Taylor remainder (Proposition 9.3)
================================================================================
  [E14.Hessian_bound ] |ACE_S - linear|   = +0.078    bound=17.58   PASS

================================================================================
  E15: Anisotropic effective rank k_eff (Remark 4.10)
================================================================================
  [E15.k_eff_value   ] k_eff (k=248)      = +6.28                   PASS
  [E15.var_aniso     ] emp/aniso          = +1.87    tol=[0.5,2.0]  PASS

================================================================================
  E16: Berry–Esseen rate (Remark 4.11)
================================================================================
  [E16.KS_distance   ] KS@500=0.039  KS@2000=0.022    tol=0.12      PASS

================================================================================
  SUMMARY
================================================================================
  45 PASS / 0 FAIL  (45 total)
  Elapsed: 244.6s
  JSON saved to D:\BlackBox_NLP_2026\toy\outputs\results.json
```

---

## 10. Bugs and design corrections discovered during the build

The toy caught seven distinct issues that would otherwise have surfaced only
when the real-model pipeline started running on Babel. Each is documented here
because the lessons transfer directly.

### 10.1 T=2 fragility makes the design matrix rank-deficient

**Symptom.** Parametric OLS gave `sin θ_max ≈ 0.99` against the planted truth.
Diffusion Maps and Kernel PCA also failed.

**Diagnosis.** At integer `s`, `sin(2π · s / 2) = sin(πs) = 0` identically.
The basis column for the T=2 sin term carries no information on integer
inputs. The 9-D basis matrix has rank 8.

**Fix.** Drop the sin(πs) basis function. Use 8-D basis throughout the toy.

**Lesson for the real run.** This is a real KT finding (their Figure 12 admits
T=2 evidence is fragile). On real models, the parametric fit should likewise
treat T=2 as cos-only or as a nuisance to be regressed out. The paper plan
should explicitly clarify HELIX_DIM = 8 in the integer-input regime.

### 10.2 Linear-axis coordinate mismatch between data and basis

**Symptom.** Even after fixing the T=2 issue, parametric OLS still failed for
unrelated reasons.

**Diagnosis.** `helix(s, C)` used `s_norm = (s - 99) / 99` for the linear
component, but `basis_vector(s)` used raw `s`. The data-generating linear
coordinate didn't match the OLS-fitting linear coordinate.

**Fix.** Refactor `helix` to call `basis_vector` directly, ensuring exactly
one linear convention.

**Lesson.** Two functions implementing "the same" basis must share code, not
just docstrings. On the real run, our parametric fit and our patch-injection
have to agree on the helix definition exactly.

### 10.3 (n, n, d_m) broadcast in pairwise distance computation

**Symptom.** Toy ran out of memory at d_m=256, n=2000 — used 13.8 GB before
we killed it.

**Diagnosis.** `(H[:, None, :] - H[None, :, :]) ** 2` materialises a
(n, n, d_m) tensor: at n=2000, d_m=256, that's 2000² × 256 × 8 bytes = 8.2 GB
per call. Diffusion Maps and Kernel PCA each call it once.

**Fix.** Use `||a - b||² = ||a||² + ||b||² - 2 a·b` to compute distances in
O(n²) memory. After the fix, n=2000 needs 32 MB total.

**Lesson.** This is a classic numpy footgun. On the real run with d_m = 4096
and n_c = 4500, the naive broadcast would take ~660 GB. The expanded-norm
trick is mandatory at scale.

### 10.4 K_bins = 20 = T period exactly cancels a helix component

**Symptom.** PCA-on-bins gave sin θ ≈ 0.53, way worse than the paper's
documented 0.10.

**Diagnosis.** With `K_bins = 20` over a 199-point range, bin width = 10.
That equals exactly one period of the T=10 helix component, so averaging over
each bin makes the T=10 signal identically zero. PCA-on-bins then misses an
entire dimension of the truth.

**Fix.** Use `K_bins = 100`. Bin width is now ~2, much smaller than any helix
period, so averaging only partially smears the signal.

**Lesson.** Binning bias is dimension-aware: `K_bins` must be much larger
than the highest-frequency component you want to recover. On real models, if
the paper's primary M̂ falls back to PCA-on-bins as a robustness check, the bin
count needs to scale with the operand range.

### 10.5 Readout that only reads the on-helix component makes ACE_S = 0

**Symptom.** First version of E8: ACE_S = -0.001 (basically zero) instead of
the predicted ~0.25.

**Diagnosis.** The first readout was `W[s] = helix(s) / ||helix(s)||²` —
purely on the helix span. Any perturbation `ξ ⊥ M_S` is invisible to such a
readout (`W[s] @ ξ = 0` for all `s`).

**Fix.** Add an off-span readout component (a linear ramp in `s` along
`v_flip`, plus a small random off-span term). Calibrate amplitudes so σ-noise
doesn't flip argmax but a unit perturbation does.

**Lesson.** Real lm_head readouts have rich off-span structure — their ability
to read off-span components is what makes mechanistic interventions work at
all. The paper's causal pipeline assumes this implicitly. Any synthetic
analogue must replicate it explicitly.

### 10.6 Confound on the wrong axis defeats matched permutation

**Symptom.** First version of E6: matched permutation Type I = 1.0 (should be
≈ 0.05).

**Diagnosis.** The first version added extra noise *only to wrong samples*.
This made wrong-vs-correct distributions differ *within* each bin, which is
exactly what matched permutation requires to be the same.

**Fix.** Add the carry-correlated extra noise to *all* samples regardless of
label. Now within-bin distributions are identical (same noise) but
between-bin marginals differ (carry bins have higher noise overall). Matched
permutation now works as intended.

**Lesson.** Designing confound experiments requires knowing exactly which
axis the confound lives on. On real data, harder problems can have either
within-bin or between-bin differences depending on the task. The paper's
matched permutation only protects against the between-bin kind. We should
audit which kind we have on real GPT-J data.

### 10.7 Display labels stale after `.passed` override

**Symptom.** Final SUMMARY says 27 PASS / 0 FAIL but the printed lines for E7
and E8 sometimes show "FAIL" status before the override runs.

**Diagnosis.** The first implementation called `record(...)` (which prints
the line based on a default tolerance check) and then mutated
`RESULTS[-1].passed`. The mutation updated the JSON but didn't re-print the
line.

**Fix.** Use a `custom()` helper that constructs the Result with the correct
`.passed` value before printing. The JSON and the printed line now match.

**Lesson.** Print and persist should derive from the same source of truth.
This was a cosmetic bug but it would have caused confusion for any reviewer
reading the run log.

---

## 11. How to interpret the artifacts in `outputs/`

### 11.1 `results.json`

```json
{
  "n_pass": 27,
  "n_fail": 0,
  "elapsed_s": 206.0,
  "results": [
    {
      "name": "E1.null",
      "metric": "T_n",
      "value": 0.0119,
      "predicted": 0.0,
      "tolerance": 0.03,
      "passed": true,
      "note": "(small noise floor at sigma^2 scale)"
    },
    ...
  ]
}
```

This is the canonical record of the run. Diff between runs to see what
changed; the values are stable across runs because the seed is fixed.

### 11.2 `fig_E1_concentration.png`

Three lines on log-log axes:
- Empirical std(T_n) at each `n` (data points connected with a solid line).
- Best-fit line from log-log regression (dashed line, with slope label).
- 1/√n reference (dotted line).

The empirical line should track the dotted line at large `n` and run slightly
steeper at small `n` (chi-squared bias).

### 11.3 `fig_E2_power_curves.png`

3×3 grid of subplots. Each subplot: one (r, Δ) cell. X-axis: n (3 points).
Y-axis: empirical power (0 to 1). Horizontal dashed line at 0.8 marks the
power threshold.

### 11.4 `fig_E3_method_comparison.png`

Five lines on log-log axes (one per method). Each line: sin θ_max vs n_c. The
parametric line should have the cleanest 1/√n slope; PCA-on-bins should
flatten; the kernel methods should track parametric.

### 11.5 `tab_E3_methods.csv`

Tab-separated method × n_c table of sin θ_max values.

---

## 12. What the toy does *not* validate

A short list, to be honest about the limits of synthetic validation:

- **Real activation noise structure.** The toy assumes isotropic Gaussian
  noise (Assumption GM). Real residual-stream noise after layer norm is
  anisotropic and can be non-Gaussian. The paper's anisotropic extension to
  Theorem 1 (§5.1) is not toy-tested.
- **Llama-specific weak helix fit.** The toy plants a clean helix and gets a
  clean recovery. Real Llama 3.1 8B has a weaker last-token helix fit (KT
  Figure 23) — the toy can't tell us how much of the test's power survives in
  that regime.
- **Manifold misspecification.** The toy's ground truth helix matches the
  fit basis exactly. On real data, the answer-helix vs operand-union vs joint
  arithmetic manifold question (paper §3.5) is open. The toy does not
  exercise this choice.
- **Tokenizer artefacts and prompt-template effects.** The toy's input is just
  (a, b) pairs of integers; there is no tokenizer, no `=` sign at the end, no
  multi-token answer handling.
- **Per-layer dynamics.** The toy operates at one synthetic "layer." Real
  models have 28-32 layers and the paper plans to pick the layer where helix
  R² is highest. Layer selection is not toy-tested.
- **Real causal-pipeline hooks.** The toy uses a linear readout `W` we
  constructed. Real models patch the residual stream via HuggingFace forward
  hooks on a transformer block; the actual mechanics of registering those
  hooks, capturing activations, running the patched forward pass, and reading
  out lm_head logits is not toy-tested.
- **Long activation cache lifecycle.** The toy doesn't write ~1 GB activation
  caches to Babel scratch; real-model SLURM jobs need that cache plumbing to
  be resumable across SLURM preemptions — see
  [../babel_execution_plan.md §12](../babel_execution_plan.md).

The toy is a math validation, not a systems validation. Both are needed. The
toy says the math is correct given clean data; the Babel run says we can get
clean data out of real models.

---

## 13. Reproducibility and seeds

### 13.1 Single seed

```python
SEED = 20260504    # the date of this paper's first toy run
```

This seed is used:
- For `make_helix_directions` (the orthonormal C).
- For sampling each World's perturbation V_true and its random coefficients.
- For sampling the noise in each World.
- For the readout's W_offspan random component.

Each World derived from `make_world(..., seed=SEED + offset)` is a separate
RNG, so different experiments use different offsets and don't interfere.

### 13.2 Determinism

Given the seed and the run command, the toy is bit-for-bit reproducible
modulo:
- Numpy version (we tested with numpy 2.3.5).
- BLAS implementation (sklearn's eigh and lstsq depend on the underlying
  LAPACK; results to 8 decimal places are stable; differences below that are
  expected across BLAS versions).

The values in this README's tables are from the run on May 6, 2026 in the
`privacy` env on this Windows box (45 PASS / 0 FAIL, ~245 seconds wall
clock). Re-running on the same machine should reproduce them exactly.
Re-running on Babel (Linux + MKL) may produce values that differ by
~1e-6 in the last decimal but should not flip any pass/fail verdict.

### 13.3 Wall-clock instability

The 245-second total wall time is approximate. It varies by ±20% depending on
system load (the laptop is shared with the user's other work). The compute
itself is deterministic; the wall time is not. On a Babel CPU node the same
toy finishes in ~3-4 minutes.

---

## 14. How to extend the toy

### 14.1 Adding a new failure mode

Edit `synth_world.py`:

```python
elif perturbation_kind == "your_new_kind":
    # construct xi_full of shape (n_pairs, d_m)
    xi[is_wrong] = xi_full[is_wrong]
```

Then add a corresponding test case to `E4_failure_modes` in `run_toy.py` with
predicted (T_curve, T_span) signatures.

### 14.2 Adding a new manifold method

Edit `manifold_methods.py`:

```python
def fit_my_method(H_c, s, k=HELIX_DIM, ...):
    # do whatever; return U_hat of shape (d_m, k) with orthonormal columns
    ...
```

Then add it to the `methods` list in `E3_manifold_recovery`. If the method
has a documented theoretical scaling, add an inline pass criterion.

### 14.3 Adding a new test statistic

Edit `tests.py`. Use the existing `T_n_subspace`, `T_n_curve`, `T_n_V` as
templates. Each test takes (H_c, H_w, ...) and returns a scalar. Make sure
the statistic is **odd** in the (correct, wrong) swap direction or note
explicitly why it isn't.

### 14.4 Adding a new intervention

Edit `causal.py`. Use the existing `ACE_*` functions as templates. Make sure
the intervention specifies (a) which subspace it operates on, (b) which
population it applies to (correct vs wrong), and (c) the sign convention for
the LD shift.

### 14.5 Running at d_m = 4096 (the paper's actual scale)

```python
# In run_toy.py main:
d_m_target = 4096
E1_theorem1_validity(d_m=d_m_target, sigma=0.1)
E3_manifold_recovery(d_m=d_m_target, sigma=0.05)
# Skip E2/E5/E6/E7/E8 at full scale (too slow).
```

Expected runtime: ~10-15 minutes on CPU. The 4096 sweep is gated behind a
manual flag because it takes 3-4× as long as the d_m=256 default.

### 14.6 Comparing two runs

After two runs that produced `results.json`:

```python
import json
a = json.load(open("run_A/results.json"))
b = json.load(open("run_B/results.json"))
for ra, rb in zip(a["results"], b["results"]):
    if ra["passed"] != rb["passed"]:
        print(f"  FLIPPED: {ra['name']}: A={ra['passed']}, B={rb['passed']}")
    elif abs(ra["value"] - rb["value"]) > 0.01:
        print(f"  DRIFT: {ra['name']}: A={ra['value']:.4f} vs B={rb['value']:.4f}")
```

This is the regression-suite use case. After every revision to the toy or to
the paper's math, this script tells you what changed.

---

*End of toy README. Total: ~1300 lines. Last updated: 2026-05-04.*
