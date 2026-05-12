# Observations and Open Directions — `paper_math.md` Second Pass

*Working notes from a second line-by-line read of [paper_math.md](paper_math.md), now at commit `f5f8b89`. The first pass (commit `b56397d`) reviewed an earlier draft; the math has since been rewritten substantially. Roughly 60% of the first-pass concerns are now addressed in the document. The remaining 40% stays here, joined by a smaller set of new observations the rewrite has surfaced. Line references throughout point at the current 1388-line file.*

*Date: 2026-05-11.*
*Reader pass: Anshul, after the rewrite that responds to AI-reviewer feedback (see [paper_math.md:1207-1230](paper_math.md#L1207-L1230)).*

---

## 0. Reading frame

Same posture as last pass: this is a list of observations and directions, not a list of objections. The proofs, where they exist, are largely standard once their assumptions are accepted; the interesting questions remain about whether the assumptions match real activations and whether the framing leaves the most informative signal on the table.

What changed since the first pass:

- The κ_max / σ·κ_max audit is now first-class. [Remark 3.6](paper_math.md#L255-L262) pre-registers `(τ̂, κ̂_max, σ̂_eff, η̂)` as filters before the test runs.
- Anisotropic noise has its own theorem: [Theorem 4.10](paper_math.md#L484-L525), with `k_eff` and a Hanson–Wright proof.
- The matching minimax rate is downgraded to a conjecture, with the two-point Le Cam bound presented as the only proven lower bound. The honest separation at [paper_math.md:705-737](paper_math.md#L705-L737) is the right call.
- The strict `V ⊆ ⋂_p N_p M` condition for the curve case is replaced by an explicit drift bound `η_0` ([Theorem 5.2(C)](paper_math.md#L567-L573)).
- Permutation now cites Freedman–Lane, Anderson–Robinson, Hemerik–Goeman, and states exchangeability as `h ⊥ y | φ(a, b)` ([§8.2](paper_math.md#L1017-L1049)).
- Cross-fitting Neyman-orthogonality is verified by an explicit Gateaux calculation at [paper_math.md:957-998](paper_math.md#L957-L998).

What did not change, and now stands out as the central remaining gap, is the per-period harmonic decomposition. The three failure modes ([Definition 2.6](paper_math.md#L138-L158)) are still defined topologically (on-curve, in-span off-curve, off-span), with no mechanism in the math for separating "the model lost the ones place" from "the model lost the tens place." Section 1 below makes the case for closing this gap, since it is both the highest-leverage methodological extension and the cheapest one to add.

A second theme is the rewrite's strategic posture. [Section 12](paper_math.md#L1233-L1320) explicitly accepts a workshop-grade scope: BlackboxNLP target, no main-conference push, the matching minimax bound and the cross-domain generalization both deferred. That posture is defensible, but it changes what counts as "still live." A few items below would matter for a main-conference submission and are correspondingly low priority for the current target.

---

## 1. The per-period harmonic decomposition is still missing, and it is the single most informative addition

The helix curve is parameterized by frequencies `T ∈ {2, 5, 10, 100}` ([Definition 2.1](paper_math.md#L91-L107)). Each `(cos_T, sin_T)` pair carves out a circle inside the helix span. A wrong activation can sit off-helix in any of these circles independently: the ones-place encoder (`T = 10`) can fail while the tens-place encoder (`T = 100`) succeeds, or vice versa.

The current statistic `T_n^V` collapses this into a single scalar. The math at [paper_math.md:551-558](paper_math.md#L551-L558) projects the residual onto a single subspace `V` and reports its squared norm, averaged. Two activations with the same `||P_V r||²` could correspond to a clean failure in `T = 10` only, or to a diffuse failure spread across `T = 5, 10, 100`. The headline statistic does not distinguish them.

What the per-period decomposition would buy:

1. A **digit-scale failure profile** per wrong activation. For each `T`, compute `||P_{V_T} r(h_w)||²` where `V_T = span(u_cos^T, u_sin^T)`. The 4-tuple is the failure signature. Reporting it converts "off-manifold" into a mechanistic claim about which numerical place broke.
2. A **stratified causal narrative**. If wrong activations cluster on `T = 10` failures for high-carry inputs and `T = 100` failures for high-magnitude inputs, that is the kind of finding that makes a workshop reviewer remember the paper. It is also exactly the structure KT's representation predicts but does not test.
3. **A sharper test under cancellation**. If a model's wrong activations have small mean residual on aggregate but large spread across `T`, the scalar statistic loses power while the per-`T` decomposition picks it up. The matched permutation framework adapts cleanly: run the test once per `T`, apply a step-down correction across the four tests.

The mathematical cost is small. The per-`T` projections are already in use by the parametric estimator ([Definition 6.1](paper_math.md#L765-L774)). The chi-squared null calibration carries through directly with `r` replaced by `r_T = 2`. A new Theorem could state per-period validity in two pages.

The interpretive cost of *not* adding it is asymmetric. If the empirical pipeline runs scalar `T_n^V` and finds a positive effect, a reviewer will ask: "But which digit-place broke?" If the pipeline runs the per-`T` decomposition first and reports the profile, the answer is in the table.

**Direction.** Add a short Section 5.8 stating per-period validity. Run the per-`T` decomposition as part of the headline empirical pipeline, not as an appendix. If carries map to `T = 10` failures specifically, lead the empirical paper with that finding.

---

## 2. The causal section remains the weakest part of the math

Three concerns from the first pass are still live.

### 2.1 The injection patch is a population summary

[Definition 9.2](paper_math.md#L1071-L1092) constructs `ACE_S` using `Patch(h_c, V, 1, μ̂_ξ)`. The patch direction `μ̂_ξ` is the average wrong-population displacement. An individual wrong activation has its own `ξ̂(a, b)`. Patching with the average reproduces the central-tendency failure but not the variation around it.

The cleaner experiment is paired patching. For each wrong activation `h_w(a, b)`, find a correct activation `h_c(a', b')` with `φ(a, b) = φ(a', b')` (the matched permutation framework provides this) and patch `Patch(h_c(a', b'), V, 1, ξ̂(a, b))`. The ACE estimated this way is per-problem, not population-average. The decomposition into "consistent population shift" versus "problem-specific failure" is then directly readable from the variance of the per-pair effects.

**Direction.** Add a paired-patching variant `ACE_S^paired` to [Definition 9.2](paper_math.md#L1070-L1092). Report both. If they agree, the population summary is fine; if they disagree, the per-problem story is the one to report.

### 2.2 The Alignment Assumption ALN has an ad-hoc threshold

The rewrite adds (ALN) at [paper_math.md:1136-1143](paper_math.md#L1136-L1143), with calibration constant `c_0 = 0.3` declared as the threshold below which the magnitude prediction is "inconclusive." The number 0.3 is unjustified in the doc. Any reviewer will ask: why not 0.2, why not 0.5?

Two possible defenses:

1. Pre-register `c_0` based on the synthetic toy. If the toy gives `cos θ` distributions for clean alignment and adversarial misalignment that are well-separated, the threshold falls out of that calibration.
2. Drop the threshold and report `cos θ` as a continuous quantity, with the magnitude prediction graded by it. The reviewer-facing claim is then "alignment is `cos θ = X`, predicted ACE is `Y`, observed ACE is `Z`; the prediction holds at the X-th percentile of the noise floor."

The second is cleaner. Putting a single number in front of an inequality looks like p-hacking even when it is not.

**Direction.** Replace the binary "above/below 0.3" framing with a continuous calibration plot (predicted vs observed ACE, colored by `cos θ`) in the empirical paper. The math doc can drop the threshold entirely.

### 2.3 The argmax-flip surrogate is acknowledged but not solved

[Remark 9.4](paper_math.md#L1154-L1161) introduces `LD_τ` via log-sum-exp at temperature `τ`, and acknowledges the Hessian bound `L = O(1/τ)` near argmax flips. This is open question 9 at [paper_math.md:1203](paper_math.md#L1203). The rewrite is honest about the obstruction but offers no resolution.

Three options worth piloting empirically:

1. Use the **signed gap** `LD(h) − LD(h')` as the statistic. The argmax flip becomes a discrete event to count, not a smoothness obstruction.
2. Use **softmax probability** at the answer token, `softmax(logits)[s]`. Smooth everywhere, uniformly bounded Hessian, and arguably the more natural quantity since the model's behavior is determined by the probability distribution, not the logit difference.
3. Stratify the causal analysis by whether the patch crosses an argmax boundary, and report the within-stratum effect separately.

Option 2 is the simplest. It also has the advantage that "the patch increased the answer probability" is a more reviewer-friendly claim than "the patch increased the logit difference."

**Direction.** Run the synthetic toy with `softmax[s]` as the target functional. If the magnitude-vs-Taylor ratio is closer to 1× than the current 5× discrepancy ([Q8 at paper_math.md:1201](paper_math.md#L1201)), the argmax-flip story is the dominant source of the toy discrepancy and the smooth-probability variant is the right thing to use in the empirical paper.

---

## 3. The Llama bias term is in Theorem 6.5 but not propagated to Theorem 6.6

[Theorem 6.5](paper_math.md#L857-L868) gives the misspecification-aware sin-theta bound, with explicit `bias + variance` decomposition. Good. But [Theorem 6.6](paper_math.md#L872-L897), the composition with Theorem 1, only carries forward the `sin θ_max` term. There is no `bias(M)` term in the composed bound's right-hand side.

For Llama 3.1 8B, the rewrite acknowledges this is the realistic setting (KT Figure 23, cited at [paper_math.md:761](paper_math.md#L761)). With `b(M) > 0`, the test does not become exact as `n_c → ∞`; the variance term shrinks but the bias floor is fixed. At some sample size, the test stops improving. This should be visible in the bound and in the empirical reporting.

**Direction.** Add a Theorem 6.6' that includes both bias and variance components. State explicitly: when `b(M) > 0`, the test asymptotic value is `||μ_ξ||² + tr(Σ_ξ) + O(b(M))`. Pre-register a misspecification check on Llama: fit the parametric model, compute the residual `||m̂(a, b) − h_c(a, b)||` per pair, and report its mean as `b̂(M)`. If `b̂(M)` is comparable to the headline `T_n` magnitude, the Llama result is a Llama-specific story (the helix is wrong for that model), not a falsification of the framework.

---

## 4. Layer selection has no post-selection inference correction

The layer-selection protocol at [paper_math.md:29](paper_math.md#L29) chooses `ℓ*` by held-out `R²` on the helix basis, from a candidate set of four layers per model. The test then runs at `ℓ*`. This is a data-driven selection followed by inference on the selected object, and the math currently does nothing to account for the selection.

[Remark 5.7](paper_math.md#L754-L755) discusses Romano–Wolf for adaptive `V` selection. That is the right machinery, applied to the wrong choice variable. The selection that matters more for inference is `ℓ*`, not `V`.

The cost of correcting is small. With four candidates, Bonferroni gives a factor of `log 4 ≈ 1.4` in the threshold. Romano–Wolf gives less. Either is cheap relative to the loss of reviewer credibility from "we picked the layer that worked best, then ran the test."

**Direction.** Either (a) hold out a separate split for layer selection, run the test on a fresh split; or (b) report `T_n` at all four candidate layers with Romano–Wolf step-down across them; or (c) explicitly state in a remark that the test is conditional on a separately-justified `ℓ*` and not corrected for selection. Option (a) is the cleanest. Option (b) costs a factor of 1.4 in power. Option (c) is the minimum the math owes the reviewer.

---

## 5. The bin definition has three small but accumulating issues

[Definition 8.1](paper_math.md#L1009-L1015) factorizes bins as `4 × 4 × 10 × 10 × 2 = 3200`. [Remark 8.4](paper_math.md#L1047) and [Remark 8.5](paper_math.md#L1049) acknowledge sparse bins and exchangeability diagnostics. These are good additions. Three issues remain.

**Correlated bin factors.** `carry(a, b)`, `decile(a)`, `decile(b)`, and `sumbin(a + b)` are not independent. High-decile pairs cause more carries; large `sumbin` is mechanically tied to high deciles. The effective dimension of the bin space is closer to two or three, not five. Conditioning on a five-factor product when the data only support a two-factor structure is statistically wasteful: it shrinks the within-bin sample size for no real exchangeability gain.

**Tokclass is binary.** Real tokenizer behavior on numbers is more granular than two classes. Llama 3.1's tokenizer treats digits idiosyncratically (Appendix H of the main paper, referenced at [paper_math.md:47](paper_math.md#L47)). If two activations with different tokenization patterns sit in the same tokclass bin, the bin is conditioning on less than it appears to.

**The matched test is mean-based.** `T_n^matched` averages squared residual norms. The matched permutation null tests mean equality conditional on bin. Two distributions can have the same mean residual norm while differing in shape: one multimodal, one unimodal, both with the same `E[||r||²]`. The mean test misses shape.

**Directions.** (a) Run a chi-squared independence test on the five bin factors. If they are heavily correlated (likely), pilot a coarsened bin function `φ_2 = (sumbin, tokclass)` and report `T_n^matched` under both. (b) Replace the binary tokclass with the categorical Llama-tokenization-pattern variable from the audit. (c) Add an energy-statistic or Cramér–von Mises version of the permutation test as a sensitivity check; if it agrees with the mean test, the mean test is fine.

---

## 6. The drift bound η_0 is testable but the rewrite leaves its scale unspecified

[Theorem 5.2(C)](paper_math.md#L567-L573) replaces the strict common-normal assumption by an empirical drift constant `η_0`. This is a real improvement. But the doc does not say what value of `η_0` is small enough for the bound to be useful, or how `η_0` scales with the helix's geometry.

[Q5 at paper_math.md:1195](paper_math.md#L1195) notes the rewrite expects `η_0 = O(κ_max · diam(M))` for short curves, `O(1)` for long curves where the tangent rotates many times. For our setting (`A = 198`, four frequency components including `T = 2`), the tangent rotates roughly `198 / 2 = 99` times around the `T = 2` circle alone. The "long curve" regime is the realistic one. `η_0 = O(1)` means the additive error term in [Theorem 5.2(C)](paper_math.md#L571), which is `O(η_0 D_max²)`, is the same order as the headline signal `||μ_ξ||²` whenever `D_max² ≈ ||μ_ξ||²`.

**Direction.** Either compute `η_0` analytically for the helix curve as a function of the recovered amplitudes, or pre-register an empirical estimate as part of [Remark 3.6](paper_math.md#L255-L262)'s diagnostic table. If `η̂_0` is `O(1)` on real models, restrict `V` to subspaces where the drift is genuinely small (the high-frequency `T ∈ {2, 5}` directions, after centering); the curve case (C) is then either the right tool or the wrong tool depending on the layer, not always either.

---

## 7. The noise model: anisotropy is in, but two adjacent concerns are not

[Theorem 4.10](paper_math.md#L484-L525) handles anisotropic `Σ_ε` via `k_eff`. This addresses the layer-norm concern from the first pass. Two adjacent issues remain.

**Sub-Weibull tails.** [Q10 at paper_math.md:1205](paper_math.md#L1205) commits to empirical diagnosis of tail heaviness. Real activations after softmax-attention can have polynomial tails when a few attention heads dominate. The Hanson–Wright inequality used in Theorem 4.10's proof at [paper_math.md:505-513](paper_math.md#L505-L513) assumes sub-Gaussian; sub-Weibull would degrade the rate from exponential to polynomial. The fallback is the matched permutation, which is distribution-free, so the test does not break, but the closed-form `p`-values from the chi-squared null become unreliable.

**Within-class noise as a modeling choice.** The rewrite acknowledges at [paper_math.md:33](paper_math.md#L33) that for a fixed model `h(a, b)` is deterministic. The "noise" is between-pair variation modeled as i.i.d. sub-Gaussian. There is no within-class variation in the deterministic setting. Treating between-pair variation as i.i.d. is convenient but is a modeling decision, not an experimental fact.

**Direction.** As part of the synthetic toy validation, run the test under both Gaussian and `t(ν=4)` noise (heavy-tailed). Compare permutation `p`-values and chi-squared `p`-values. If they agree to the second decimal, sub-Gaussian is fine in practice. If chi-squared is consistently too liberal, the empirical paper should use permutation as primary and report chi-squared as a sensitivity.

---

## 8. The estimator scope is parametric-only, with no cross-validation against a generic basis

[Section 6](paper_math.md#L759-L902) builds the recovery theory around `M̂_param` exclusively. [Section 10](paper_math.md#L1167-L1179) treats the four other estimators (diffusion maps, kernel PCA, local PCA, PCA-on-bins) as cross-method validation only. The first-pass concern was that the helix basis is KT's, taken as given, and a wrong-population basis might be different.

The rewrite does not engage with this. There is no protocol for fitting a generic basis (Fourier in `a`, B-splines, or a local low-rank approximation) and comparing the recovered subspace to `M̂_param`. If the wrong population uses representational structure that is *not* in the helix basis (a carry-propagation direction that does not appear in correct-activation analysis), the parametric estimator cannot recover it, and `T_n` measures the wrong thing.

This is a low-cost addition. Fit a B-spline basis of comparable dimension to the helix (`K = 8`), recover its column span, compute `sin θ_max` between the two recovered subspaces. If they agree (`sin θ < 0.1`), the helix is the unique scaffold up to noise, and the parametric framing is justified. If they disagree, the empirical paper has a real follow-up question.

**Direction.** Add a B-spline-basis comparison to the synthetic toy first; if it works, fold into the empirical pipeline as a one-page appendix. The math doc can stay as-is; this is a methodology check, not a theoretical extension.

---

## 9. Scope: equals-token-only and per-model n_w heterogeneity are unchanged

Two scope decisions from the first pass remain.

**Equals-sign-only.** Activations are extracted at the equals-sign token ([paper_math.md:31](paper_math.md#L31)). KT's argument is that the answer-helix lives there. The failure may live upstream: the carry-propagation step happens at the digit tokens of `a` and `b`, before the `=`. If the off-manifold structure is at the digit positions and is integrated away by the time the model writes the answer, the equals-token analysis sees the symptom, not the source.

**Per-model `n_w`.** Llama 3.1 has `n_c ≈ 9800` and `n_w ≈ 200`. GPT-J has `n_c ≈ 8050` and `n_w ≈ 1950`. The power for Llama is bottlenecked by `n_w`, and the minimum-detectable `Δ` for Llama is roughly `√(1950 / 200) ≈ 3.1×` larger than for GPT-J. The math at [paper_math.md:580](paper_math.md#L580) substitutes `n = min(n_c, n_w)`, which is technically right but understates the asymmetry: per-model power calculations should be presented separately, not aggregated.

Both are noted at [Section 12.3](paper_math.md#L1262-L1267) as out of scope. That is defensible for a workshop submission. The equals-sign-only scope in particular is worth one sentence in the empirical paper acknowledging that an upstream-token analysis is the natural follow-up.

---

## 10. Methods worth piloting in parallel with the headline pipeline

These are not blockers. They are sensitivity checks and alternative lenses that, run cheaply alongside the main test, would either confirm the headline finding or sharpen it.

**Geodesic vs ambient distance.** The closest-point projection at [paper_math.md:56](paper_math.md#L56) uses ambient Euclidean norm. For a 1D curve with closed-form parameterization, geodesic distance is also closed-form. Compute both versions of `T_n` on the toy. If they agree, the linearization regime holds; if they disagree, curvature matters at the data scale and the parametric estimator's linearity is biting.

**Lattice distance `d_lattice`.** For `h ∈ M_S`, recover the parameter `t̂(h)` via the closed-form parameterization, then compute `d_lattice(h) = min_{k ∈ ℤ ∩ [0, A]} |t̂ − k|`. This directly tests the integer-lattice hypothesis. Two activations with the same `||r_within||²` could correspond to a near-miss to integer (small `d_lattice`) versus a half-integer parameter (`d_lattice ≈ 0.5`). Reporting `d_lattice` distinguishes these.

**Mahalanobis-on-residuals.** As an OOD baseline. Computes `r(h)^T Σ̂_r^{-1} r(h)` where `Σ̂_r` is the empirical correct-population residual covariance. Naturally handles anisotropic structure without fitting `Σ_ε` directly. If it matches `T_n^V`, the localization story is consistent. If Mahalanobis dominates, the test is missing variance structure that an anisotropic-noise treatment would catch.

**Wasserstein-2 between residual distributions.** As a distributional sensitivity check. If `W_2` rejects but `T_n^matched` does not, the failure has shape structure beyond the second moment.

These are one notebook each. The right time to run them is during the synthetic toy validation, not after the empirical results land.

---

## 11. What the rewrite addressed (so we do not re-flag)

Pulled from the AI-reviewer-feedback log at [paper_math.md:1207-1230](paper_math.md#L1207-L1230) and verified against the current draft:

- **κ_max diagnostics** are now first-class via [Remark 3.6](paper_math.md#L255-L262). The first-pass concern about verifiability is closed.
- **Anisotropic noise** has [Theorem 4.10](paper_math.md#L484-L525) with `k_eff` and Hanson–Wright concentration. The first-pass concern about layer-norm artifacts is closed.
- **Minimax matching rate (5.2b)** is downgraded to a conjecture, exactly as the first pass recommended. The two-point Le Cam bound is presented as the only proven lower bound.
- **Curve case (C)** replaces the strict `V ⊆ ⋂_p N_p M` by an explicit drift bound `η_0`, addressing the first-pass concern about that assumption being unrealistic (though see Section 6 above for what is still open about `η_0`).
- **Permutation framework** now cites Freedman–Lane / Anderson–Robinson / Hemerik–Goeman, states exchangeability as `h ⊥ y | φ(a, b)`, and adds [Remark 8.4](paper_math.md#L1047) on sparse bins and [Remark 8.5](paper_math.md#L1049) on exchangeability diagnostics. The first-pass concern about the bin/permutation argument being under-specified is closed.
- **Cross-fitting Neyman-orthogonality** is verified by the explicit Gateaux calculation at [paper_math.md:957-998](paper_math.md#L957-L998). The first-pass concern about `√n`-consistency is closed at the null. (The alternative-side argument is still flagged at [Q7](paper_math.md#L1199); see Section 7 above.)
- **`D_max²` looseness** in the composed bound is addressed by [Remark 6.7](paper_math.md#L899-L902), which distinguishes uncentered from centered widths and notes the unstable-subspace sharpening.
- **Sectional-vs-extrinsic curvature** terminology is fixed at [paper_math.md:67](paper_math.md#L67) (the new "Note on curvature terminology" paragraph).

Eight items closed; seven still live, distributed across Sections 1-9 above. The hit rate on the first pass was higher than I expected.

---

## 12. Specific lines to revisit on the next pass

A short index for the next read-through, by section:

- [paper_math.md:33](paper_math.md#L33): the deterministic-model framing for noise. Worth checking whether Section 4's i.i.d. sub-Gaussian framing reads consistently with this.
- [paper_math.md:91-110](paper_math.md#L91-L110): helix definition and `T = 2` degeneracy. Center of the per-period decomposition argument (Section 1 above).
- [paper_math.md:138-181](paper_math.md#L138-L181): three failure modes and Pythagorean decomposition. Where the per-period extension lives if added.
- [paper_math.md:255-262](paper_math.md#L255-L262): Remark 3.6 diagnostic table. Confirm `η̂` is included; it is, but the scaling concern in Section 6 above is not addressed in the remark itself.
- [paper_math.md:484-525](paper_math.md#L484-L525): Theorem 4.10 anisotropic version. Hanson–Wright proof outline; check `k_eff` estimator concentration claim at [paper_math.md:523](paper_math.md#L523).
- [paper_math.md:567-573](paper_math.md#L567-L573): Theorem 5.2(C) drift bound. The `η_0` scaling story is at [Q5 / paper_math.md:1195](paper_math.md#L1195).
- [paper_math.md:597-599](paper_math.md#L597-L599): conjectured matching rate (5.2b). The Ingster pointer is correct; the question is whether the workshop audience cares about the rigorous version.
- [paper_math.md:761-868](paper_math.md#L761-L868): Theorem 6.5 misspecification. Where the bias term is defined; check that Theorem 6.6 either propagates it or explicitly notes the omission.
- [paper_math.md:957-998](paper_math.md#L957-L998): Neyman-orthogonality calculation. The alternative-side argument's threshold needs sharpening.
- [paper_math.md:1009-1015](paper_math.md#L1009-L1015): bin definition. The five-factor independence assumption is implicit; Section 5 above.
- [paper_math.md:1136-1143](paper_math.md#L1136-L1143): Alignment Assumption (ALN). The `c_0 = 0.3` threshold needs justification or removal.
- [paper_math.md:1154-1161](paper_math.md#L1154-L1161): Remark 9.4 log-sum-exp smoothing. The `O(1/τ)` Hessian behavior is acknowledged but unresolved.
- [paper_math.md:1184-1205](paper_math.md#L1184-L1205): open questions for Barnábás. Q1, Q5, Q7, Q8, Q9, Q10 all map to live concerns above.

---

## 13. Priority ordering for the next math-revision pass

Roughly in decreasing leverage, with the workshop submission as the target audience:

1. **Per-period harmonic decomposition** (Section 1). The single biggest interpretive payoff for the smallest mathematical addition. If only one item from this list lands before the empirical pipeline runs, this is it.
2. **Llama bias term in the composed bound** (Section 3). Cheap to add, prevents a reviewer from claiming the framework is silently wrong on the realistic case.
3. **Layer-selection post-selection inference** (Section 4). A one-paragraph remark or a one-table extension is enough; pretending the selection does not affect inference is a known foot-gun.
4. **Drop or justify the `c_0 = 0.3` threshold** in (ALN) (Section 2.2). One unjustified number in a math paper is one too many.
5. **Paired-patching `ACE_S`** (Section 2.1). Adds two lines to Definition 9.2 and substantially sharpens the causal claim.
6. **Drift constant `η_0` scaling** (Section 6). Either an a-priori bound or pre-registered empirical reporting; without it, Theorem 5.2(C)'s usefulness depends on a constant whose typical value is not characterized.
7. **Bin-factor independence audit** (Section 5). One chi-squared test, decides whether the five-factor permutation is doing real work or just shrinking sample sizes.
8. **Per-period failure profile in the empirical paper** (Section 1, applied). Even if the math doc keeps `T_n^V` as headline, the empirical paper should report the per-`T` decomposition.

Items 1, 2, 3 are the "cannot be omitted" tier. Items 4-8 are the "would substantially improve the paper" tier. Methods worth piloting in parallel (Section 10) are below this in priority but should run during the toy validation rather than waiting for the empirical results.

What I am deferring entirely for the workshop scope: the matching minimax bound (5.2b), generalization beyond two-digit addition, equals-token-only relaxation, alternative basis comparisons (kept as a sensitivity check, not a re-derivation). The rewrite's Section 12 makes the case for these being out of scope; I find that case persuasive given the BlackboxNLP timeline.

---

## 14. What I have not investigated and might

- **The synthetic toy in detail.** The rewrite reports 27 of 27 pre-registered checks passed ([paper_math.md:1320](paper_math.md#L1320)) and a 5× magnitude-vs-Taylor discrepancy in one case ([Q8 / paper_math.md:1201](paper_math.md#L1201)). I have not read [toy/](toy/) closely enough to know whether the toy includes heavy-tailed noise, anisotropic noise, or curvature regimes that would stress-test the bounds in the way the empirical pipeline will.
- **KT Figure 23 specifically.** Cited at [paper_math.md:761](paper_math.md#L761) as the Llama misspecification reference. If Llama deviates from the helix structurally (different basis, different period set, different dimension) rather than just noisily, the bias term in Theorem 6.5 is the wrong shape and the realistic story is "Llama needs its own representation" rather than "Llama is the parametric estimator's hard case."
- **The `babel_execution_plan.md` and `full_paper_plan.md` rewrites.** Commit `f5f8b89` rewrote both substantially. I have read enough of the math doc to know what is in it; I have not done a similar pass on the plans. If the plans now over-specify items that the math has dropped or vice versa, the inconsistency would be worth catching before the pipeline runs.
- **Comparison to mechanistic-interpretability baselines.** SAE-derived subspaces, attention-head decompositions, gradient-attribution methods. A side-by-side overlap test (cosine between `V` and known SAE features) would give an independent validation of the localized subspace. Probably out of scope for the workshop, but cheap to run.

---

*End of second-pass observations. The first-pass file at commit `b56397d` is superseded by this one. Re-read before the next math-revision pass.*
