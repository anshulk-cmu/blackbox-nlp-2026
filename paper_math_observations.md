# Observations and Open Directions — `paper_math.md` Review

*Working notes from a line-by-line re-read of [paper_math.md](paper_math.md). Not decisions — places where the current framing may be too narrow, too restrictive, or doing more implicit work than it advertises, and alternative paths worth investigating before the empirical pipeline locks in.*

*Date: 2026-05-11*
*Reader pass: Anshul, after the Riemannian / topology discussion.*

---

## 0. Reading frame

This document is a list of **observations**, **concerns**, and **directions worth exploring**. It is explicitly *not* a list of objections to the math as stated. The proofs in [paper_math.md](paper_math.md) are largely standard once their assumptions are accepted; the interesting question is whether the assumptions match what real activations will look like, and whether the deterministic Euclidean framing leaves money on the table.

Themes that keep recurring across the read:

- The ambient Euclidean metric is doing more work than the document advertises.
- The "three failure modes" decomposition may be too coarse for the helix's actual internal structure.
- Several assumptions are stated as small-constant regimes whose verifiability is left implicit.
- Topology in its strict sense is the *wrong* frame for the helix; *geometric measure* and *harmonic analysis* on the parametrization are sharper for this specific object.
- The causal step is mathematically the weakest, yet narratively the heaviest in the framing section.

Nothing here is a recommendation to change the math. The point is to log *where to look harder* before we sign off.

---

## 1. Ambient Euclidean metric vs intrinsic geometry

### 1.1 The closest-point projection is ambient

`Π_M(h) := arg min_{p ∈ M} ‖h − p‖_2` at [paper_math.md:54](paper_math.md#L54) uses the ambient Euclidean norm. Two consequences:

- Near a fold in the helix (or any high-curvature region), the *ambient-closest* point may differ from the *geodesically-closest* point. The two coincide only when the residual is normal to the manifold at the projected point — a local property within reach.
- The Taylor remainder bound `‖R_2‖_2 ≤ (κ_max/2) ‖P^N δ‖_2^2` at [paper_math.md:295](paper_math.md#L295) is the price of using ambient distance. It's a second-order correction, but it shows up as `R_1(σ, κ_max) = O(σ² κ_max²)` in the population mean of `T_n` at [paper_math.md:263](paper_math.md#L263).

**Direction worth exploring:** Compute `T_n^geodesic` using arc-length distance along `M_C` (cheap, because `g(t)` is closed-form) and compare to ambient `T_n`. If they differ meaningfully, that itself is interpretable evidence: wrong activations leave the helix in directions where curvature matters. If they agree, the second-order bound is empirically tight and the simpler statistic is fine.

### 1.2 The reach assumption is binary, not graded

REG at [paper_math.md:229](paper_math.md#L229) requires `τ(M) ≥ τ_min` uniformly. The paper notes at [paper_math.md:63](paper_math.md#L63) that the inequality `τ(M) ≥ 1/κ_max` "goes the wrong way" and that reach is governed by both local curvature and global bottlenecks (weak feature size).

For the helix with components at `T = 2, 5, 10, 100`, the high-frequency `T = 2` component oscillates rapidly along the long linear axis. The reach *near* the winding directions could be small even if the linear curve has no global bottleneck. Verifying `τ_min` empirically is described as a sanity check; it might deserve to be a first-class diagnostic, with the test downgraded (or the local-PCA fallback invoked) when reach drops below a threshold.

**Direction:** Estimate local reach `τ(p)` along the curve for each model and layer. If it varies by more than ~`2×` across the support, stratifying `T_n` by reach regime is more honest than reporting a single statistic.

### 1.3 SPD / log-Euclidean on activation covariances

A complementary lens: instead of treating each activation as a point in `ℝ^d`, represent local activation structure by its covariance `Σ(h) ∈ SPD(d)` over a neighborhood. SPD(d) has a natural Riemannian metric (affine-invariant or log-Euclidean). Wrong activations could have *correctly placed means* but *anomalous covariance structure* — higher within-class variance, off-diagonal coupling, rank collapse.

This is overkill for the headline test, but it is the right tool if the empirical residuals turn out to be small in mean and large in spread. The current `T_n` confuses these two regimes; an SPD-based statistic separates them.

**Direction:** Compute per-bin activation covariance for correct and wrong populations. Compare on the SPD manifold (Riemannian distance). Defer unless the residual mean is small but the spread is large.

### 1.4 Coordinate-free formulation

Most of Section 4 of [paper_math.md](paper_math.md) is written in coordinates after picking an orthonormal basis. Many of the bounds (Laurent–Massart, Wedin) are coordinate-free in their natural form, and re-deriving them coordinate-free might reveal that the `d`-dependence is artifactual.

**Direction:** Restate Theorem 4.2 in operator-norm / trace-norm language without committing to a basis. If the bound's `k`-dependence collapses to `k_eff` more naturally in coordinate-free form, that simplifies the anisotropic generalization (Remark 4.10).

---

## 2. Topology is the wrong frame; geometric measure is the right one

This came directly out of conversation: the helix `g(t) = u_0 + t·u_lin + Σ_T (cos/sin terms)` is **contractible** — the linear `t · u_lin` drift means the curve never closes on itself. Persistent homology gives `H_0 = 1, H_k = 0 (k ≥ 1)`: one arc, no loops, no voids. PH literally cannot diagnose anything about this geometry. Any topological invariant classifies the helix as a smooth contractible 1-manifold and stops talking.

The "voids between windings" intuition is *geometric*, not topological: `M_C` is Hausdorff-dimension-1 inside the dim-9 (or 8) span `M_S`, so it's Lebesgue-null in `M_S`. Wrong activations in `M_S \ M_C` are not in topological holes; they are in regions of `M_S` that the curve never visits at integer parameter values.

**Implications:**

- The three failure modes at [paper_math.md:152-156](paper_math.md#L152-L156) — on-curve, off-curve in-span, off-span — are a topological-flavored tripartition that misses internal structure of mode (b).
- The within-span residual `r_within = Π_{M_S}(h) − Π_{M_C}(h)` is the right *direction*, but its norm is a coarse summary statistic.

### 2.1 Arc-length residual to the integer lattice

For `h ∈ M_S`, recover the parameter `t̂(h)` via the closed-form parametrization. Define

$$d_\text{lattice}(h) := \min_{k \in \mathbb{Z} \cap [0, A]} |t̂(h) − k|.$$

This directly tests the integer-lattice hypothesis. It is sharper than `r_within(h)` because two activations with the same `‖r_within‖` could correspond to a near-miss to integer (small `d_lattice`) versus a half-integer parameter (`d_lattice ≈ 0.5`).

**Direction:** Add `d_lattice` as a secondary statistic. If wrong activations cluster at half-integer parameters, that's a specific narrative: the model is "between numbers" rather than "off the curve." If `d_lattice` is uniform on `[0, 0.5]`, the failure has no integer-lattice structure and the within-span residual is the wrong thing to report.

### 2.2 Per-period phase residuals

Each `(cos_T, sin_T)` pair carves out an `S¹` inside `M_S`. The activation's projection onto that 2D subspace defines a phase. For wrong activations, the phase might be wrong at *some* `T` but right at others.

**Direction:** Decompose the residual into a `K`-tuple of per-period failure indicators. Could replace the binary "off-curve in-span" with a *digit-scale failure profile*: did the ones-place encoder fail (`T = 10`), the tens-place (`T = 100`), or both?

### 2.3 Fourier / harmonic analysis in `t`

`T = 10` corresponds to the ones-place modulus; `T = 100` to the tens-place modulus. Failure at `T = 10` means the model lost track of the ones digit; failure at `T = 100` means the tens digit. Reporting per-`T` residual decomposition would let the paper make a *mechanistic* claim sharper than "off-manifold": *"the model fails on carries because the `T = 10` component is wrong while `T = 100` is right."*

**Direction:** This is potentially the single most interpretively valuable extension. The cost is small (the projections are already used in the parametric estimator), and the gain is a *causal narrative per failure mode*.

### 2.4 The 1D parameter `t̂` is identifiable from `M_S`

The parametrization `g(t)` is injective on `[0, A]` for any non-degenerate combination of frequencies. So `t̂(h)` is well-defined for any `h ∈ M_S` via solving `g(t̂) = Π_{M_C}(h)`. This means we have a 1D *intrinsic coordinate* along the manifold. Many of the tests above use this implicitly.

**Direction:** Treat `t̂` as a first-class observable. Plot the empirical distribution of `t̂(h_w) − t̂(h_c)` for matched pairs `(a, b)`; this is the "parameter slip" induced by being wrong. If it's centered at zero with spread, the failure is symmetric drift. If it's biased (e.g., wrong answers consistently project to *higher* `t̂`), there's a directional story.

---

## 3. Curvature and `κ_max` assumption

### 3.1 The `σ · κ_max ≤ c_0` constraint

REG at [paper_math.md:229](paper_math.md#L229) requires this. The `T = 2` and `T = 5` components have extrinsic curvature scaling as roughly `(2π/T)² · ‖u_*‖` which is largest for small `T`. If the recovered helix has nontrivial amplitude at `T = 2` (the degenerate component, see [paper_math.md:108](paper_math.md#L108)) or `T = 5`, the curvature constraint may be tight.

**Direction:** Compute `κ_max` empirically per model and layer (closed-form trig calculation from the recovered `{u_*}` directions). Check `σ̂ · κ̂_max` against `c_0 ∈ (0, 1]`. If the constraint is violated anywhere in the data support, the Theorem 1 remainder term is no longer `O(σ² κ_max²)` small; it competes with the headline signal.

### 3.2 The `m = 8` vs `m = 9` ambiguity for integer inputs

Remark 2.2 at [paper_math.md:107-110](paper_math.md#L107-L110): `sin(πa) ≡ 0` at integer `a`, so the `T = 2` sin direction is unidentifiable from integer-only data. The paper drops it. But this is *observational identifiability*, not *structural identifiability*: if any internal computation (carry propagation, e.g.) writes to the `sin(πa)` direction *off* the integer lattice, the test cannot see it on-lattice. Wrong activations that drift off-lattice into that direction would be invisible to a basis fit from on-lattice correct data.

**Direction:** Verify that the `sin(πa)` direction is not used by the model. Inject a non-integer parameter (e.g., continuous interpolation between integer prompts via embedding-space interpolation) and check whether the `T = 2` sin component activates. If so, the basis is incomplete and `m = 8` is an under-fit; the off-span residual will include an artifactual component.

### 3.3 Reach near linear-axis endpoints

The linear axis `u_lin` extends across `[0, A]`. At `t = 0` and `t = A`, the curve has open endpoints. The reach near endpoints is well-defined for an open curve, but the projection's *uniqueness* at the boundary can fail (closest point becomes ambiguous when `h` lies "past" the endpoint).

**Direction:** Check whether any correct or wrong activations project to `t̂` near `0` or `A`. If so, those samples should be excluded or treated separately, similar to how kernel methods handle boundary effects. The number of affected samples is likely small (most `a + b` lie well inside `[0, 198]`) but the affected region may be enriched in `s = 0` and `s = 198` answers, which are special cases.

### 3.4 Sectional vs extrinsic curvature

Section 1.3 of [paper_math.md](paper_math.md) is explicit at [paper_math.md:65](paper_math.md#L65) that for a 1D curve, the right object is the second fundamental form, not sectional curvature. This is correct, but Remark 3.5 at [paper_math.md:229](paper_math.md#L229) still says "maximum sectional curvature is bounded by `κ_max`." Worth re-checking that all uses of `κ_max` consistently refer to `‖II_p‖_op`, not a sectional invariant that doesn't apply.

---

## 4. Noise model concerns

### 4.1 Isotropic Gaussian is implausible after layer normalization

Layer normalization constrains activations to a `(d-1)`-sphere of radius `√d` (approximately, after the rescale). Sub-Gaussian noise around this constraint is *not* isotropic in `ℝ^d`; it lives tangent to the sphere. Assumption GM(iii) at [paper_math.md:200](paper_math.md#L200) sets `Σ_ε = σ² I_d`, the simplest case, but probably wrong for any layer immediately downstream of a normalization.

**Direction:** Theorem 4.2's effective-rank remark at [paper_math.md:452-459](paper_math.md#L452-L459) handles anisotropic noise via `k_eff = (tr P^N Σ_ε)² / tr((P^N Σ_ε)²)`. Estimate `Σ_ε` empirically per layer and report `k_eff` alongside `k`. If `k_eff ≪ k`, the test is sharper than the isotropic version suggests and the paper should claim the tighter bound. If `k_eff ≈ k`, the isotropic approximation is empirically valid.

### 4.2 Same noise across populations

Assumption GM(iii) requires `Σ_ε^c = Σ_ε^w`. Remark 4.9 at [paper_math.md:443-450](paper_math.md#L443-L450) acknowledges this. For wrong vs correct populations on the same input distribution, this is plausible. But for wrong populations enriched in high-carry / boundary-of-difficulty problems, the noise structure could systematically differ — high-carry inputs may have more attention-head activity, hence higher residual-stream variance.

**Direction:** Empirically estimate `Σ_ε^c` and `Σ_ε^w` per layer and bin. Test `Σ_ε^c = Σ_ε^w` via a Box's-M-style test stratified by bin. The matched permutation in Section 8 only controls for the *mean* of the input difficulty, not the noise structure. A mismatch here is not fatal (Remark 4.9 gives a correction term) but it does change the test's null distribution.

### 4.3 Sub-Gaussian vs sub-Weibull tails

Open question 10 at [paper_math.md:1026](paper_math.md#L1026) flags this. Real activations after softmax-attention can have heavier-than-sub-Gaussian tails, driven by a few high-attention tokens dominating. Laurent–Massart assumes Gaussian for the *exact* form; sub-Weibull would give polynomially decaying tails with weaker concentration (Kuchibhotla–Chakrabortty 2022).

**Direction:** Estimate empirical Orlicz `ψ_α`-norms of residuals per layer. If `α < 2` (heavier than sub-Gaussian), switch concentration to the corresponding sub-Weibull form. The rate degrades but stays exponential up to the data extremes. The matched permutation provides distribution-free fallback if the bound fails.

### 4.4 Within-class noise: latent variation vs deterministic

The paper acknowledges at [paper_math.md:31](paper_math.md#L31) that for a fixed model, `h(a, b)` is deterministic. The "noise" is latent variation modeled as if i.i.d. across `(a, b)` pairs. There is no within-class variation in the deterministic setting — the noise is between pairs, not between draws. Treating this as i.i.d. sub-Gaussian is a *modeling choice*, not an experimental fact.

**Direction:** Decompose empirical variance of correct activations into between-`(a, b)` variance (predictable from the helix basis) and residual variance (the "noise"). The latter is what should match Assumption GM. If residual variance is near zero — the helix basis explains everything — the test is degenerate and `T_n` becomes a property of the fitting, not the model. If it has structure, the noise model is misspecified in a specific way.

### 4.5 The prompt-template fixed component

[paper_math.md:29](paper_math.md#L29) extracts activations at the equals-sign token under a fixed prompt template. The prompt template contributes a deterministic offset that is the *same* across all `(a, b)` pairs. After mean-centering (Remark 2.4 at [paper_math.md:127](paper_math.md#L127)), this offset cancels — but only if it's exactly identical, which depends on tokenization and attention-mask details.

**Direction:** Verify the prompt-template offset is constant within a tokclass bin. If it varies (e.g., because the tokenizer treats `(a, b) = (1, 2)` differently than `(a, b) = (12, 34)`), the centering is imperfect and a residual mean-shift contaminates the off-manifold residual.

---

## 5. Assumption strain in the proofs

### 5.1 WLOG re-centering of `ξ` (Open Q1)

Lemma 3.2 at [paper_math.md:203-214](paper_math.md#L203-L214) claims the tangent component of `ξ` can be absorbed into a redefined `m_c`. The argument is first-order in curvature, with second-order error `O(κ ‖P^T ξ‖)`. Open question 1 at [paper_math.md:1008](paper_math.md#L1008) flags the exponential-map calculation as not rigorous.

**Observation:** If the wrong-population `ξ` is large (e.g., `‖ξ‖ ≈ σ`), the second-order correction is comparable to the noise floor, and the re-centering is not free. The test then mixes "off-manifold" with "shifted-along-manifold-then-re-centered" — these are different mechanisms.

**Direction:** Sketch the exponential-map calculation rigorously or state Lemma 3.2 with an explicit small-`ξ` assumption (e.g., `‖ξ‖ ≤ c · τ_min`). If `ξ` cannot be assumed small, replace the WLOG argument with a direct decomposition that reports tangent and normal components separately. The empirical work can then check which dominates.

### 5.2 Linearization-error sharpness (Open Q2)

`R_1(σ, κ_max) = O(σ² κ_max²)` at [paper_math.md:263](paper_math.md#L263). Open question 2 flags that the exact constant is not tracked. For small `σ κ_max`, this is a quadratic-in-`σ` term and is negligible; for the regime where `σ κ_max → c_0`, it is order-one and absorbs the headline signal.

**Direction:** Compute `σ κ_max` empirically and report whether the regime is "deep in the small-curvature limit" (`σ κ_max < 0.1`, say) or "marginal" (`σ κ_max ∈ [0.1, c_0]`). The paper's claims have very different strength in these two regimes, and the relevant model–layer combinations may sort into both buckets.

### 5.3 Cross-fitting under the alternative (Open Q6)

Theorem 7.2 at [paper_math.md:812-825](paper_math.md#L812-L825) gives asymptotic normality at the null. Open question 6 at [paper_math.md:1018](paper_math.md#L1018) notes the non-asymptotic version at the alternative is not done. For sample-size planning, this is what matters.

**Direction:** Either complete the calculation (Chernozhukov machinery, tedious but standard) or simulate calibrated alternatives in the toy companion to determine the empirical sample size required for the target effect sizes.

### 5.4 Minimax lower bound's chaining (Open Q4)

Theorem 5.2(b) at [paper_math.md:501-507](paper_math.md#L501-L507) claims `n ≥ C₂ r σ⁴ / Δ⁴ log(…)`. The proof at [paper_math.md:624-638](paper_math.md#L624-L638) uses two-point Le Cam plus a Fano chaining sketch. Open question 4 at [paper_math.md:1014](paper_math.md#L1014) explicitly says "this is the place where we are least confident."

**Direction:** Two options. (1) Rederive the chaining carefully — the construction of an orthogonal packing of `r` perturbations of norm `Δ/√2` in `V` needs the explicit `r σ⁴ / Δ⁴` constant from Tsybakov Theorem 2.7. (2) Downgrade Theorem 5.2(b) to a *conjecture* with the two-point bound stated as a lemma. Section 12.5 at [paper_math.md:1086](paper_math.md#L1086) already flags this as plausible.

### 5.5 The Davis–Kahan vs Wedin sharpness claim (Open Q5)

[paper_math.md:690](paper_math.md#L690) claims operator-norm Wedin is sharper than Frobenius Davis–Kahan by a factor of `√K`. Open question 5 at [paper_math.md:1016](paper_math.md#L1016) flags that the OLS Frobenius bound (Lemma 6.3) may re-introduce the `√K` that operator-norm Wedin saves.

**Direction:** Trace the `K` dependence through both forms and report which is tighter. If operator-norm Wedin is *not* sharper after the OLS step, the choice is a wash and Davis–Kahan is fine.

---

## 6. Bin / permutation issues

### 6.1 Bin sparsity at 3200 bins

Definition 8.1 at [paper_math.md:876](paper_math.md#L876) defines 3200 nominal bins, of which "approximately 200–300 are non-empty" on the 10,000-pair grid. Average bin size: 30–50 samples; rarest bins: very few. Within a bin with 5 samples, the permutation distribution has at most `5!` distinct values, which is 120 — fine for `α = 0.05` but tight.

**Observation:** Lehmann–Romano exact conditional validity (Theorem 8.3) holds for *each* bin, but the bin-weighted average statistic mixes bins of very different sizes. A small bin contributes high-variance noise but may be weighted equally.

**Direction:** Either (a) Mantel–Haenszel-style precision-weighted combination, (b) drop bins below a size threshold (e.g., `≥ 10`), or (c) coarsen the bin definition until all bins have `≥ N_min`. Each has tradeoffs:
- (a) is statistically efficient but assumes the within-bin effect is constant.
- (b) is conservative but discards data.
- (c) increases bin sizes but may merge mechanistically distinct strata.

The current paper picks one of these implicitly via "bin-weighted average" at Section 3.3 of the main paper, without making the choice explicit in the math.

### 6.2 Carry / decile / tokclass are correlated, not orthogonal

The factorization `4 × 4 × 10 × 10 × 2 = 3200` assumes the components are independent. In reality, `carry(a, b)` is highly correlated with `decile(a)` and `decile(b)`: high-decile pairs cause more carries. The *effective* dimension of the bin space is probably closer to 2–3, not 5.

**Direction:** Chi-squared test of independence on the bin components. If they're correlated (as expected), the matched permutation is technically valid but inefficient — it's conditioning on more than the data supports. A reduced bin definition (e.g., just `sumbin × tokclass`) might be statistically more powerful while still controlling for the relevant confounds.

### 6.3 Tokenization class is binary

The tokclass factor has 2 levels at [paper_math.md:879](paper_math.md#L879). But actual tokenizer behavior on numbers is more granular: BPE tokenizers may split `13` differently across leading-position vs middle-position; Llama 3.1's tokenizer treats digits idiosyncratically (per the Appendix H audit reference at [paper_math.md:47](paper_math.md#L47)).

**Direction:** Either expand tokclass to capture finer behavior or marginalize the test across tokenization regimes. The choice affects whether the matched permutation conditions on "two activations tokenized the same way" or just "both in a coarse tokclass."

### 6.4 The matched permutation tests means, not distributions

The statistic `T_n^matched` is mean-based (sum of squared residuals). The matched permutation null tests *mean equality* conditional on bin. But two distributions can have the same residual-norm mean while differing in shape (e.g., one is multimodal, the other unimodal, both with the same `E[‖r‖²]`).

**Direction:** Run a Cramér–von Mises or energy-statistic version of the permutation test, which compares full distributions. If it agrees with `T_n^matched`, the mean-based statistic is fine. If it dominates, the failure has distributional structure beyond the mean.

---

## 7. Causal proposition concerns

### 7.1 Argmax-flip non-smoothness (Open Q8)

Remark 9.4 at [paper_math.md:976-982](paper_math.md#L976-L982) introduces log-sum-exp smoothing of `LD` at temperature `τ`. The Hessian bound `L = O(1/τ)` near argmax flips, so the bound deteriorates as `τ → 0`. Open question 8 at [paper_math.md:1022](paper_math.md#L1022) explicitly says "we do not have a clean way to handle this."

**Directions worth exploring:**
- Use the *signed gap* `LD(h) − LD(h_perturbed)` as the statistic rather than `LD(h)` directly; the gap is bounded in `[−2 max|logit|, 2 max|logit|]` and the argmax flip becomes a discrete event to count rather than a smoothness obstruction.
- Stratify the causal analysis by *whether* the patch crosses an argmax boundary, and report the within-stratum effect.
- Use a smooth surrogate like `softmax(logits)[s]` directly. This is smooth everywhere and the Hessian bound is uniform.
- Use the *expected* logit-difference under temperature-`T` sampling instead of the deterministic argmax. This regularizes the boundary naturally.

### 7.2 The injection patch `δ = μ̂_ξ` is a population summary

Definition 9.2 at [paper_math.md:931](paper_math.md#L931) uses `Patch(h_c, V, 1, μ̂_ξ)`. But `μ̂_ξ` is the *average* failure direction; an individual wrong activation has its own `ξ(a, b)`. Patching with the population average doesn't reproduce any specific failure mode.

**Direction:** Paired patching. For each wrong activation `h_w` with displacement `ξ̂(a, b)`, find a *correct* activation `h_c` with similar `(a, b)` (matched permutation gives this) and patch `Patch(h_c, V, 1, ξ̂(a_w, b_w))`. The ACE estimated this way is per-problem rather than population-average. The variance decomposition tells you whether the failure is a *consistent* shift (μ̂ alone explains it) or *problem-specific* (per-pair `ξ̂` is needed).

### 7.3 Random-subspace baseline

The `ACE_R` at [paper_math.md:932](paper_math.md#L932) uses `V^rand` of the same dimension and a random target `δ^rand`. Two concerns:
- The random subspace is uniform on the Grassmannian, which is not a meaningful "null"; activations are anisotropic, so any uniform-random direction is *not* a comparable baseline.
- The norm of `δ^rand` is not specified; if it's matched to `‖μ̂_ξ‖` it's a different question than if it's matched to noise scale.

**Direction:** Use a *covariance-aware* random subspace: sample `V^rand` from the eigenstructure of the empirical activation covariance, matched in eigenvalue rank to `V`. This gives a "same-energy" null that the unconstrained Grassmannian doesn't. Report the empirical effect size under both definitions of the random baseline; if they agree, the choice doesn't matter, but if they differ, the empirical baseline is what reviewers will trust.

### 7.4 Higher-order Taylor terms (Open Q7)

Proposition 9.3's remainder is at order `L ‖μ̂_ξ‖² / 2`. Open question 7 at [paper_math.md:1020](paper_math.md#L1020) notes that the synthetic toy found a magnitude-vs-Taylor ratio of about 5× in one calibrated case, suggesting higher-order corrections or argmax-flip effects dominate.

**Direction:** Either bound `‖∇³ LD‖` and add the third-order term explicitly, or run a Taylor-residual diagnostic on the empirical data: compute `LD(h + Δh) − LD(h) − ⟨∇LD, Δh⟩` and check whether it scales as `‖Δh‖²` (Taylor regime) or differently (argmax-flip or higher-order regime). The diagnostic answers Q7 and Q8 together.

---

## 8. Estimator choice and misspecification

### 8.1 Llama misspecification

Section 6 at [paper_math.md:662](paper_math.md#L662) flags Llama 3.1 as a likely misspecification case (KT Figure 23). Theorem 6.5 handles this via an explicit bias term `b(M)`. But the composed bound at [paper_math.md:766-774](paper_math.md#L766-L774) doesn't include this bias term; it only includes the variance term.

**Direction:** Restate Theorem 6.6 to include both bias and variance components when `b(M) > 0`. The bias term is non-vanishing as `n_c → ∞`, so the test does not become exact; this should be disclosed clearly in empirical reporting. For Llama, the bound's variance shrinks with `n_c` but the bias floor is fixed — at some `n_c` the test stops improving.

### 8.2 The basis is *the helix basis*

Definition 6.1 at [paper_math.md:667](paper_math.md#L667) uses `b_j(a) = (linear, cos, sin)` from the helix. This is KT's basis, taken as given. If wrong activations use a *different* basis (e.g., a basis sensitive to carry-propagation that doesn't appear in KT's correct-activation analysis), the helix basis cannot recover the relevant manifold for wrong activations.

**Direction:** Cross-check by fitting a generic basis (Fourier in `a`, or B-splines) of comparable dimension and comparing recovered subspaces via sin-theta. If the generic basis recovers a substantially different `M̂`, the helix is not the unique scaffold and the paper's framing needs caveating ("we assume KT's basis is shared by correct and wrong populations; this is testable").

### 8.3 PCA-on-class-means as a strawman

Section 10 at [paper_math.md:992](paper_math.md#L992) dismisses PCA-on-bins via misspecification term `b(M) = O(1/K_bins)`. But for `K_bins = 200` non-empty bins, this is `b(M) ≈ 0.005` — small. The strawman argument may overstate the case.

**Direction:** Run PCA-on-bins as a baseline and report its sin-theta to `M̂_param`. If they're close, the paper's argument for the parametric estimator is weakened: the choice becomes one of convenience rather than necessity, and reviewers may push back.

### 8.4 Diffusion Maps as cross-validation only

Section 10 at [paper_math.md:996](paper_math.md#L996) uses Diffusion Maps as cross-method validation only. But Diffusion Maps is intrinsic (Laplace–Beltrami operator) and would naturally handle curved manifolds without the second-order Taylor approximation. The reason cited for not using it as primary is the difficulty of a clean composed bound for embedding-then-lift.

**Direction:** If Diffusion Maps recovers a manifold that *differs* from `M̂_param` in a structured way (not just noisy), that itself is a finding. Report sin-theta between `M̂_DM` and `M̂_param` per layer. If they disagree on curved-manifold features, the parametric estimator's linearity is biting and the test should use DM in the high-curvature regime.

---

## 9. Alternative methodological lenses worth piloting

### 9.1 Conformal prediction for null calibration

The matched permutation gives distribution-free null calibration via exchangeability. Conformal prediction gives a different distribution-free guarantee (marginal coverage), with the advantage that it doesn't require within-bin exchangeability. The test becomes: *"is `r(h_w)` a conformal outlier with respect to `{r(h_c)}` at level `α`?"*

**Direction:** Pilot conformal prediction on the residuals. Bypasses bin sparsity entirely. Gives marginal rather than conditional guarantee — a weaker but possibly more practical claim. Worth running as a sensitivity check against the headline permutation result. If they agree, the conditional bin structure is doing real work; if conformal is stronger, the bin definitions may be over-constraining.

### 9.2 Wild bootstrap

If noise is heteroskedastic across `(a, b)`, permutation can over- or under-reject. Wild bootstrap (multiply each residual by a Rademacher random variable) gives an exchangeability-free alternative.

**Direction:** Compare wild-bootstrap vs permutation `p`-values. Disagreement = heteroskedasticity is real and the permutation test should report a sensitivity bound.

### 9.3 Score-based / energy-based OOD detection

The test is fundamentally an out-of-manifold detector. OOD literature has matured (energy scores, Mahalanobis distance, gradient-based methods). Mahalanobis-on-residuals would account for anisotropic residual covariance without explicit `Σ` estimation.

**Direction:** Mahalanobis-on-residuals as a baseline. If it matches `T_n^V`, the localization story is consistent. If Mahalanobis dominates, the test is missing variance structure that anisotropic-noise treatment would catch.

### 9.4 Optimal transport between distributions

The matched permutation tests mean differences. Wasserstein-2 between correct and wrong residual distributions tests *distributional* differences. Could detect shape differences that mean-tests miss.

**Direction:** Wasserstein-2 as a secondary statistic. If `W_2` rejects but `T_n` does not, the failure has shape structure beyond the second moment.

### 9.5 Riemannian gradient on `M` for the causal step

For Proposition 9.3, instead of the Euclidean gradient `∇LD(h)`, use the Riemannian gradient on `M`: project `∇LD(h)` to `T_p M`. This decomposes "logit change due to motion along `M`" from "logit change due to motion off `M`" and gives a sharper specificity claim: the off-manifold gradient is what carries the causal signal. On-manifold gradient = "we moved to a different number"; off-manifold gradient = "we broke the representation."

**Direction:** Decompose `∇LD(h)` into tangential and normal components at each correct activation. Compute the ratio `‖P^N ∇LD‖ / ‖P^T ∇LD‖`. If it's large, the model is sensitive to off-manifold perturbations specifically, supporting the causal claim. If it's small, motion along `M` matters more, and the off-manifold story is weaker.

### 9.6 Information geometry on output distribution

Each activation `h` induces a softmax distribution over output tokens. The Fisher–Rao metric on this distribution space is natural. Off-manifold `h` could correspond to off-manifold *output distributions*, which has a clean Riemannian structure and a chi-squared concentration via Stein.

**Direction:** Explore as long-term direction, probably out of scope for BlackboxNLP. Most natural if the paper later generalizes beyond addition.

### 9.7 SAE-based subspace identification

If sparse autoencoders trained on these models have identified arithmetic-specific features, the localized subspace `V` should overlap with those features. A side-by-side comparison would give an *independent verification* of `V`.

**Direction:** Look up SAE features for GPT-J, Pythia, Llama 3.1 if available (Anthropic's SAEs, EleutherAI's). Compute overlap (cosine similarity between SAE-feature directions and `V`'s basis). If overlap is high, `V` is mechanistically grounded; if low, the geometric `V` is a different object than the SAE features, which is also informative.

---

## 10. Scope and generalization

### 10.1 Two-digit addition only

Section 12.3 at [paper_math.md:1062](paper_math.md#L1062) restricts to KT's scope. Reviewer attack 2 at [paper_math.md:1080](paper_math.md#L1080) ("wrong examples are just harder") is acknowledged. Within "two-digit addition," wrong activations may be concentrated at *specific* problem types (carries, e.g.) and the test may be entirely driven by that subpopulation.

**Direction:** Stratify the headline `T_n` by problem type (no-carry, one-carry, two-carry). If the effect is in one stratum only, report that explicitly and reframe the contribution as "carry-specific failure geometry" rather than generic "off-manifold failure." This is a sharper claim, not a weaker one, but it changes the title.

### 10.2 Equals-sign token only

[paper_math.md:29](paper_math.md#L29) extracts activations at the equals-sign token. KT argued this is where the answer-helix lives. Wrong activations may have detectable off-manifold structure at *earlier* token positions (where the carry should propagate) that is invisible at `=`.

**Direction:** Replicate the test at the digit tokens of `a` and `b`, not just `=`. If failure is upstream of `=`, the equals-sign-only analysis misses where it actually happens, and the paper's mechanistic claim is weaker than it could be.

### 10.3 Per-model `n_c` heterogeneity

Llama 3.1 8B has `n_c ≈ 98%` of 10,000, leaving `n_w ≈ 200`. GPT-J has `n_c ≈ 80%`, `n_w ≈ 2000`. The power for Llama is bottlenecked by `n_w`, not `n_c`. The paper's sample-size analysis doesn't distinguish the two regimes.

**Direction:** Recompute the power calculation at [paper_math.md:497](paper_math.md#L497) with `n = min(n_c, n_w)` substituted (the paper does this), but note that for Llama the effective power is dominated by `n_w = 200`. Minimum-detectable `Δ` for Llama is `~5×` larger than for GPT-J. Empirical claims should be per-model with explicit power.

### 10.4 Layer selection contamination

Section 3.6 of the main paper (referenced from [paper_math.md:27](paper_math.md#L27)) selects `ℓ*` by held-out `R²` on the helix basis. The selection is data-driven; the test is then run on the selected layer. This is a *post-selection* inference setup that the math does not currently account for.

**Direction:** Either (a) hold out a separate split for layer selection vs testing, (b) report `T_n` at all candidate layers and apply Bonferroni / Romano–Wolf correction, or (c) acknowledge the post-selection issue in a remark. Option (b) loses a factor `log(4)` ≈ 1.4 in power, which is cheap.

### 10.5 Greedy decoding at temperature 0

[paper_math.md:31](paper_math.md#L31) fixes temperature 0. Real model usage often involves non-zero temperature; failures under sampling may have different geometric structure than failures under greedy decoding (which are model-deterministic).

**Direction:** Out of scope for this paper but worth flagging as a follow-up: do the off-manifold residuals at temperature `T > 0` reproduce the temperature-0 story? If not, the failure-geometry framing is regime-specific.

---

## 11. Cross-cutting themes

These observations recur across multiple sections. Worth treating as priorities for the next review pass.

### 11.1 Verifiability of assumptions

GM, BD, REG are stated as if they hold; the paper notes (Section 3.2) that some will be verified empirically. A *prerequisite-check table* — for each assumption, the diagnostic that confirms or refutes it on this dataset — would make the conditional structure clear. If an assumption fails, the corresponding theorem's claim should be downgraded with the failure mode documented.

### 11.2 Per-period decomposition as a sharp interpretive lens

The most promising methodological extension is decomposing residuals by harmonic period `T`. Low cost, leverages the parametrization already in hand, converts a generic "off-manifold" claim into a *digit-scale mechanistic* claim. May be the single highest-value addition to the empirical pipeline.

### 11.3 Geodesic / Riemannian as sensitivity, not replacement

The clean chi-squared null and closed-form `p`-values from the Euclidean residual are statistically valuable. Replacing them wholesale loses the sharp inference. The right role for Riemannian methods is as a *parallel statistic* run alongside the Euclidean one, with disagreement flagged as evidence of curvature mattering.

### 11.4 The matched permutation does heavy lifting

Theorem 8.3 carries the credibility argument against the "wrong is just harder" attack. Its assumptions (bin definition, exchangeability) deserve more scrutiny than the headline theorems. Bin sparsity, dependence among bin components, and tokclass coarseness are all open. Worth a separate section in the appendix justifying the bin choice.

### 11.5 Causal proposition is the weakest link

Proposition 9.3 has the most uncertain math (argmax flip, choice of `δ`, random baseline). Yet the causal step is what the framing in Section 12.4 leans on to distinguish the paper from KT. The math here likely needs the most revision; the synthetic-toy 5× discrepancy is a warning sign.

### 11.6 Curvature is consistently the second-order story that may be first-order

Across Sections 1, 3, 5, the curvature constant `κ_max` is treated as bounded and well-behaved. For the helix's high-frequency components, this is the assumption that empirically may not hold. A `κ_max` audit per layer is the highest-leverage diagnostic.

---

## 12. What I have not investigated and might

- **Synthetic-toy reproducibility.** The paper notes (Section 12.8) that 27/27 pre-registered checks passed in the toy. I have not reviewed the toy itself; if it does not include the cases above (heavy tails, anisotropic noise, misspecification, high curvature), it gives less assurance than it appears. Worth a follow-up read of `toy/`.
- **Empirical `Σ_ε` structure.** No empirical estimate of activation noise in this document. The isotropic assumption is the highest-leverage one; an early pilot would tell us how much trouble it's in. Probably one notebook's worth of work.
- **KT Figure 23 specifically.** The Llama misspecification reference is cited but I have not read KT's Figure 23. If Llama deviates from the helix *structurally* (different basis, different period set), the parametric estimator is the wrong tool there, not just slightly biased.
- **Comparison to mechanistic-interpretability baselines.** SAE-derived subspaces, attention-head decompositions, gradient-attribution methods. Worth a side-by-side: does our `V` align with a known mechanistically-identified subspace? If yes, the paper has independent validation; if no, an explanation is owed.
- **Babel cluster availability for the per-period pilot.** [scripts/](scripts/) has Phase-2 scripts on A6000. A per-period decomposition pilot would be a small additional notebook.
- **Whether the prompt template affects helix recovery.** Different prompt templates may yield different `{u_*}` direction vectors. KT used a specific template; we inherit it. Robustness across templates is a sanity check.

---

## 13. Specific lines to revisit on the next pass

Quick index of references I want to come back to:

- [paper_math.md:27](paper_math.md#L27) — layer selection protocol; post-selection inference question
- [paper_math.md:54](paper_math.md#L54) — closest-point projection definition; ambient vs geodesic
- [paper_math.md:63](paper_math.md#L63) — reach lower-bound discussion
- [paper_math.md:107-110](paper_math.md#L107-L110) — `T = 2` degeneracy and `m = 8`/`m = 9` choice
- [paper_math.md:149](paper_math.md#L149) — within-span residual definition
- [paper_math.md:152-156](paper_math.md#L152-L156) — three failure modes (re-examine for harmonic refinement)
- [paper_math.md:200](paper_math.md#L200) — isotropic Gaussian noise assumption
- [paper_math.md:229](paper_math.md#L229) — `σ κ_max ≤ c_0` regularity assumption
- [paper_math.md:263](paper_math.md#L263) — `R_1(σ, κ_max)` linearization error
- [paper_math.md:295](paper_math.md#L295) — Taylor remainder for `Π_M`
- [paper_math.md:443-450](paper_math.md#L443-L450) — Remark 4.9, different noise across populations
- [paper_math.md:452-459](paper_math.md#L452-L459) — Remark 4.10, anisotropic effective dimension
- [paper_math.md:501-507](paper_math.md#L501-L507) — Theorem 5.2(b) minimax lower bound (downgrade candidate)
- [paper_math.md:662](paper_math.md#L662) — Llama misspecification reference
- [paper_math.md:766-774](paper_math.md#L766-L774) — composed bound (missing bias term)
- [paper_math.md:876-879](paper_math.md#L876-L879) — bin definition (3200 bins, ~200-300 non-empty)
- [paper_math.md:931-932](paper_math.md#L931-L932) — causal patches and random baseline
- [paper_math.md:976-982](paper_math.md#L976-L982) — log-sum-exp smoothing of `LD`
- [paper_math.md:1006-1027](paper_math.md#L1006-L1027) — open questions section

---

## 14. Status

None of the above are decisions. They are observations from a single pass through [paper_math.md](paper_math.md) plus the Riemannian / topology discussion. The likely next moves, in roughly decreasing leverage:

1. **Per-period (harmonic) decomposition of residuals.** High interpretive payoff, low cost. Converts "off-manifold" into a digit-scale mechanistic claim.
2. **Empirical anisotropic-noise diagnostic.** Determines whether Theorem 1's isotropic form is sharp or loose. One notebook.
3. **Verification of `σ κ_max ≤ c_0`** per (model, layer). Determines whether the linearization remainder is negligible. Closed-form calculation.
4. **Stratification of `T_n` by problem type** (carry structure). Answers the "wrong is just harder" attack at the *signal* level, not just calibration.
5. **Decision on whether to downgrade Theorem 5.2(b)** to a conjecture. Already plausible per Section 12.5 of [paper_math.md](paper_math.md).
6. **Bin definition audit:** correlated factors, sparsity, weight choice.
7. **Empirical comparison of `M̂_param` to `M̂_DM` and PCA-on-bins.** Validates or undermines the estimator choice.
8. **Causal proposition diagnostics:** Taylor-residual check, argmax-flip stratification, covariance-aware random baseline.

Open questions 1, 4, 6, 8 from Section 11 of [paper_math.md](paper_math.md) are the ones that should be Barnábás-routed before the writeup is finalized.

---

*End of observations. Re-read before the next math-revision pass.*
