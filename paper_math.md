# Off-Manifold Failure: A Geometric Test for Localized Computational Errors in Language Models

*Mathematical Proposal — Draft for Review*

**Authors.** Anshul Kumar and Barnábás Póczos

## Abstract

This document is a mathematical proposal. It lays out the problem we want to study, the test statistic we propose, and the theorems we can prove. Each theorem is stated as a target with a proof outline; the full proofs are work in progress.

We are sharing this in advance of running the empirical pipeline on GPT-J 6B, Pythia 6.9B, and Llama 3.1 8B (target venue: BlackboxNLP 2026, direct submission deadline July 17, 2026), so that the math can be reviewed and tightened in parallel with the experimental work rather than after. We flag every place where we are uncertain about the statement, the proof structure, or the constants. Section 11 collects these into a list of open questions we would like your input on.

The technical content we propose: (1) a non-asymptotic concentration bound for an off-manifold residual statistic, with anisotropic-noise generalization (Theorem 4.10); (2) a localized version with a power upper bound and a *two-point Le Cam* lower bound — the matching $r\sigma^4/\Delta^4$ minimax rate is downgraded to a conjecture pending an Ingster-style chi-squared mixture argument; (3) a finite-sample manifold-recovery bound for the parametric estimator with operator-norm Wedin and an explicit misspecification term; (4) a cross-fitting protocol with explicit Neyman-orthogonal influence function and Gateaux-derivative verification; (5) exact conditional validity of a stratified permutation null in the Lehmann–Romano / Freedman–Lane / Anderson–Robinson framework; (6) a causal proposition with Hessian-controlled second-order Taylor remainder.

A note on scope: an earlier draft attempted a Fano-chaining proof of the matching minimax lower bound; this draft removes that argument as incoherent and presents only the two-point bound as proven, with the matching rate flagged as conjectural. The localized-test assumption $V \subseteq \bigcap_p N_p M$, which is unrealistic for a curved manifold, is also relaxed in this draft to an explicit normal-frame drift bound $\eta_0$ that is empirically verifiable.

We would value your feedback on whether the proof structures are sound, where the constants can be tightened, and where the assumptions can be relaxed. We expect to revise this document substantially based on your comments before any of it appears in the BlackboxNLP submission.

---

## 1. Setup: notation and the data we will work with

We work in a finite-dimensional Euclidean space $\mathbb{R}^{d}$ throughout, where $d$ is the residual-stream width of the language model under analysis. For the three models we plan to analyze empirically (GPT-J 6B, Pythia 6.9B, Llama 3.1 8B), $d = 4096$ in all three cases. We keep $d$ as a free parameter to emphasize that the proposed theorems should be model-agnostic.

This section is descriptive: we are setting up the objects, not yet proving anything. The proposed theorems begin in Section 4.

### 1.1 Activations as random vectors in $\mathbb{R}^d$

Fix a transformer language model $f$ with $L$ blocks. Let $\ell \in \{0, 1, \dots, L\}$ index the residual-stream layer, with $\ell = 0$ being the embedding output and $\ell = L$ being the final residual stream before the unembedding map. We are interested in a fixed analysis layer $\ell^{*}$, chosen by the layer-selection protocol of the main paper (Section 3.6). The candidate set per model is $\{L/4, L/2, 3L/4, L-1\}$ (Llama 3.1 8B, $L=32$: $\{8, 16, 24, 31\}$; GPT-J 6B, $L=28$: $\{7, 14, 21, 27\}$; Pythia 6.9B, $L=32$: $\{8, 16, 24, 31\}$). We select $\ell^{*}$ as the candidate with highest held-out $R^{2}$ on the helix basis, with fallback to Diffusion Maps if no candidate achieves $R^{2} \geq 0.9$.

For an integer addition problem $(a, b)$ with $a, b \in \{0, 1, \dots, 99\}$ and ground-truth answer $s = a + b \in \{0, 1, \dots, 198\}$, we run the model on a fixed prompt template (specified in the main paper, Section 3.2) and extract the residual-stream activation at the equals-sign token at layer $\ell^{*}$. We denote this activation as $h \in \mathbb{R}^{d}$. When we want to emphasize the dependence on the problem, we write $h(a, b)$.

For a fixed model, $h(a, b)$ is a deterministic function of $(a, b)$ because the model's weights and the tokenizer are deterministic and we decode greedily at temperature $0$. The randomness in our analysis comes from the choice of which $(a, b)$ pairs we sample and from how we model within-class variation (which we treat as latent noise; see Assumption GM below).

### 1.2 Correct and wrong populations

Define the correctness label. Let $\tau: \{0, 1, \dots, 198\} \to \mathcal{V}$ be the (fixed, model-specific) tokenization map sending an integer answer $s$ to the token id used by the model to represent the decimal-string of $s$ as the immediate next token after the equals sign (under our prompt template; see Appendix H of the main paper for tokenizer-audit details and the handful of multi-token cases we exclude). Let $\mathrm{logit}^{\mathrm{final}}_{t}(a, b)$ denote the logit assigned to vocabulary token $t \in \mathcal{V}$ at the answer position by the *final-layer* output of the model (i.e., after the unembedding map applied to the last residual stream), not by a logit-lens at the analysis layer $\ell^*$. Then

$$
y(a, b) =
\begin{cases}
1 & \text{if } \arg\max_{t \in \mathcal{V}} \mathrm{logit}^{\mathrm{final}}_{t}(a, b) = \tau(s), \\
0 & \text{otherwise.}
\end{cases}
$$

We refer to the set $\{(a, b) : y(a, b) = 1\}$ as the *correct population* and to $\{(a, b) : y(a, b) = 0\}$ as the *wrong population*. The correctness label is determined by the model's actual greedy decoding behavior at temperature $0$, not by an intermediate-layer probe; the analysis layer $\ell^*$ is only the layer at which we extract activations $h$ for residual computation.

We collect $n_c$ activations from the correct population into a matrix $H_c \in \mathbb{R}^{n_c \times d}$ (one activation per row) and $n_w$ activations from the wrong population into a matrix $H_w \in \mathbb{R}^{n_w \times d}$. The integer $n_c$ depends on the model (approximately $80.5\%$ of $10{,}000$ ordered pairs for GPT-J, $77.2\%$ for Pythia, $98\%$ for Llama, per Kantamneni–Tegmark 2025 Table 2; the per-model retained counts after tokenizer audit are reported empirically in Appendix H of the main paper).

### 1.3 The manifold $M$ and its tangent–normal decomposition

We assume there exists a smooth submanifold $M \subset \mathbb{R}^{d}$ on which correct-population activations live (in the noise-free limit). We do not yet specify $M$'s parametric form; that is the subject of Section 2. For now $M$ is a set with the following structure:

- $M$ is a smooth $m$-dimensional embedded submanifold of $\mathbb{R}^{d}$, with $m \ll d$. We write $q := d - m$ for the codimension. For Kantamneni–Tegmark's helix-span at the answer position, $m = 9$ in the continuous-input ideal model and $m = 8$ in the integer-only identifiable model (see Remark 2.2).
- $M$ has finite *reach* $\tau(M) > 0$ in the sense of Federer [federer1959]. Concretely, for every $x \in \mathbb{R}^{d}$ with $d(x, M) < \tau(M)$, the closest-point projection $\Pi_{M}(x) := \arg\min_{p \in M} \|x - p\|_{2}$ exists and is unique.
- For every $p \in M$, the tangent space $T_{p} M \subset \mathbb{R}^{d}$ is an $m$-dimensional linear subspace, and the normal space $N_{p} M = (T_{p} M)^{\perp}$ is a $q$-dimensional linear subspace. Let $P^T_{p}, P^N_{p}: \mathbb{R}^{d} \to \mathbb{R}^{d}$ denote the orthogonal projections onto $T_{p} M$ and $N_{p} M$ respectively, so $P^T_{p} + P^N_{p} = I_{d}$ and $P^T_{p} P^N_{p} = 0$.

The reach $\tau(M)$ controls how far from $M$ the closest-point projection remains well-defined and globally injective. Following the modern reach literature, reach is controlled by both *local* curvature and *global* bottlenecks:

$$
\tau(M) \;=\; \min\bigl\{\tau_{\mathrm{local}}(M),\,\tfrac{1}{2}\,\mathrm{wfs}(M)\bigr\},
$$

where $\tau_{\mathrm{local}}(M) \leq 1/\kappa_{\max}$ is bounded above by the inverse of the maximum operator norm of the second fundamental form, and $\mathrm{wfs}(M)$ is the weak feature size capturing the narrowest near-self-intersection bottleneck. We do *not* attempt to derive $\tau(M) \geq 1/\kappa_{\max}$ from curvature alone (the inequality goes the wrong way: curvature gives an upper bound on the local reach, not a lower bound on the global reach). We simply assume below that $\tau(M)$ is bounded below by some positive $\tau_{\min}$ uniformly. Verifying this empirically per (model, layer) pair is a sanity check we plan to run before computing $T_{n}$.

**Note on curvature terminology.** For a 1-dimensional curve $M$ embedded in $\mathbb{R}^{d}$, we use *extrinsic curvature*, equivalently the operator norm of the *second fundamental form* $\|II_{p}\|_{\mathrm{op}} \leq \kappa_{\max}$, not "sectional curvature." Sectional curvature is an intrinsic Riemannian-geometric invariant of 2-planes, which does not apply to a 1-dimensional curve. For a linear subspace, the second fundamental form vanishes identically.

### 1.4 The off-manifold residual

For any $h \in \mathbb{R}^{d}$ with $d(h, M) < \tau(M)$, define the *off-manifold residual*

$$
r(h) := h - \Pi_{M}(h). \tag{1.1}
$$

By construction, $r(h) \in N_{\Pi_{M}(h)} M$; the residual lies in the normal bundle of $M$ at the closest point.

When $M$ is itself a linear subspace with orthonormal basis $U \in \mathbb{R}^{d \times k}$ (so $U^{\top} U = I_{k}$), then $\Pi_{M}(h) = U U^{\top} h$ and $r(h) = (I - U U^{\top}) h$. We will use both the linear-subspace case (helix span $M_{S}$) and the smooth-manifold case (helix curve $M_{C}$) in what follows; they are treated in parallel.

---

## 2. The two manifold objects we propose: curve and span

We follow Kantamneni and Tegmark (2025), who establish empirically that in their three target models the integer-encoding helix is both a one-dimensional curve in residual-stream space (one parameter, the integer value) and the embedding of that curve into a nine-dimensional ambient subspace (the column span of one linear axis plus four cosine and four sine directions). These are different geometric objects with different test statistics, and we believe both are interpretively informative. We define both formally.

The proposal in this section is to track the curve and the span as distinct objects throughout, so that the three failure modes (on-curve, off-curve in-span, off-span) can be cleanly distinguished. Whether the data actually exhibit all three modes is an empirical question we plan to address experimentally; the math here just sets up the framework.

### 2.1 The helix curve $M_{C}$

**Definition 2.1 (Helix curve — continuous).**
Let $A \in \{99, 198\}$ be the upper endpoint of the integer label range (we use $A = 99$ for operand encoding and $A = 198$ for answer encoding). Define the continuous helix function $g: [0, A] \to \mathbb{R}^{d}$ by

$$
g(t) := u_{0} + t \cdot u_{\mathrm{lin}}
+ \sum_{T \in \{2, 5, 10, 100\}}
\bigl(\cos(2\pi t / T) \cdot u_{\cos}^{T}
+ \sin(2\pi t / T) \cdot u_{\sin}^{T}\bigr)
$$

where $u_{0}, u_{\mathrm{lin}}, u_{\cos}^{T}, u_{\sin}^{T} \in \mathbb{R}^{d}$ are fixed direction vectors recovered from data. The *helix curve* is the continuous 1-dimensional submanifold

$$
M_{C} := \{g(t) : t \in [0, A]\} \subset \mathbb{R}^{d}.
$$

Integer-labeled samples $h(a)$ correspond to evaluations of $g$ at integer $t = a$; we treat them as discrete observations on the continuous curve, not as defining the curve. Throughout the paper, when we speak of tangent spaces, projection linearization, or reach, we are working on the continuous curve $M_{C}$, not on a finite point cloud.

**Remark 2.2 (The $T = 2$ degeneracy at integer inputs).**
Note that $\sin(2\pi a / 2) = \sin(\pi a) = 0$ for all integer $a$. Hence the basis function $\sin(2\pi a / 2)$ vanishes identically on the support of integer-labeled data, and the corresponding column of any design matrix built from these basis functions on integer inputs is the zero column. Kantamneni and Tegmark discuss this fragility in their Figure 12.

For *integer-only* regression, the identifiable basis has $K = 8$ non-degenerate functions (one linear, one $\cos(\pi a)$, then three $\cos/\sin$ pairs for $T \in \{5, 10, 100\}$); the helix-span dimension recoverable from data is correspondingly $m = 8$. We retain the 9-vector notation for direction vectors $\{u_{*}\}$ as elements of $\mathbb{R}^{d}$ when discussing the ideal continuous-input model, but explicitly note that $\dim(M_{S}^{\mathrm{eff}}) = 8$ throughout the empirical setting we plan to analyze.

### 2.2 The helix span $M_{S}$

**Definition 2.3 (Helix span — affine).**
The helix span is the *affine subspace*

$$
M_{S} := u_{0} \;+\; \mathrm{span}\bigl(
\{u_{\mathrm{lin}}\} \cup
\{u_{\cos}^{T}, u_{\sin}^{T} : T \in \{2, 5, 10, 100\}\}
\bigr) \subset \mathbb{R}^{d}.
$$

We let $S := \mathrm{span}(\cdots)$ denote the linear part and $U_{S} \in \mathbb{R}^{d \times m}$ an orthonormal basis for $S$, with $m = \dim(S) \in \{8, 9\}$ depending on whether the data are integer ($m = 8$) or continuous ($m = 9$). Throughout the paper we assume all activations are mean-centered before fitting — equivalently we treat the offset $u_{0}$ as a separate intercept term to be estimated in regression. After centering, the relevant projection in the test statistic is onto the linear subspace $S$, not the affine $M_{S}$.

**Remark 2.4 (Linear vs affine).**
The distinction between the affine $M_{S}$ and its linear part $S$ matters. Real residual-stream activations are not guaranteed to lie in a linear subspace through the origin, and ignoring the offset would make the off-manifold residual dominated by a mean-shift artifact. We explicitly center activations (subtract the empirical mean of $H_{c}$) before computing $T_{n}$ in all empirical work; under this centering, $M_{S}$ becomes the linear $S$.

**Remark 2.5.**
$M_{C} \subset M_{S}$. Every point on the helix curve is a linear combination of the nine direction vectors, hence lies in their span. The converse fails: most points in $M_{S}$ are not on $M_{C}$ because they correspond to non-integer values of $a$ or to integer values plus span-internal noise.

### 2.3 The three failure modes

The curve–span distinction induces an orthogonal decomposition of the residual.

**Definition 2.6 (Failure modes).**
For an activation $h \in \mathbb{R}^{d}$, define

$$
\begin{aligned}
r_{S}(h) &:= h - \Pi_{M_{S}}(h) = (I - U_{S} U_{S}^{\top}) h, \\
r_{C}(h) &:= h - \Pi_{M_{C}}(h).
\end{aligned}
$$

The within-span residual is

$$
r_{\mathrm{within}}(h) := r_{C}(h) - r_{S}(h) = \Pi_{M_{S}}(h) - \Pi_{M_{C}}(h).
$$

The three failure modes are characterized by the relative magnitudes of $\|r_{S}(h)\|$ and $\|r_{\mathrm{within}}(h)\|$:

(a) *On-curve, in-span*: both small. $h$ lies near $M_{C}$, projecting close to a valid integer-labeled point $g(t^*)$ on the curve.
(b) *In-span but off-curve*: $\|r_{\mathrm{within}}(h)\|$ large, $\|r_{S}(h)\|$ small. $h$ lies (approximately) in $M_{S}$ but its helix coordinates do not match $g(t^*)$ for any $t^* \in [0, A]$ — that is, $h$ uses linear combinations of the helix-basis directions that are inconsistent with the parametric curve. (Equivalently: there is no $t^*$ such that $h \approx g(t^*)$, even though $h$ projects cleanly onto $S$.)
(c) *Off-span*: $\|r_{S}(h)\|$ large. $h$ has departed $M_{S}$.

**Lemma 2.7 (Pythagorean residual decomposition).**
For any $h \in \mathbb{R}^{d}$ with $d(h, M_{C}) < \tau(M_{C})$,

$$
\|r_{C}(h)\|^{2} = \|r_{S}(h)\|^{2} + \|r_{\mathrm{within}}(h)\|^{2}.
$$

*Proof outline.* Since $M_{C} \subset M_{S}$, we have $\Pi_{M_{C}}(h) \in M_{S}$, so

$$
r_{S}(h) = h - \Pi_{M_{S}}(h) \in M_{S}^{\perp},
$$

while

$$
r_{\mathrm{within}}(h) = \Pi_{M_{S}}(h) - \Pi_{M_{C}}(h) \in M_{S}.
$$

The two are therefore orthogonal in $\mathbb{R}^{d}$, and Pythagoras gives the claim. ∎

We will define test statistics $T_{n}^{S}$ and $T_{n}^{C}$ corresponding to these two residuals and report both. Throughout, when we write $T_{n}$ without qualification, we mean $T_{n}^{S}$, the linear-subspace version.

---

## 3. The generative-model assumption we propose

The proposed theorems will be stated under a generative model that captures the simplest non-trivial alternative to the null "correct and wrong activations are identically distributed." The assumption is restrictive — real activations almost certainly violate parts of it — but it is the level at which we believe clean theorems can be stated. We discuss relaxations in remarks throughout.

**Assumption 3.1 (Generative model, GM — proposed).**
For each population $p \in \{c, w\}$, we observe activations of the form

$$
h = m_{p}(a, b) + \varepsilon_{p}, \qquad
m_{p}: \{0, \dots, 99\}^{2} \to \mathbb{R}^{d}, \qquad
\varepsilon_{p} \in \mathbb{R}^{d},
$$

where:

(i) $m_{c}(a, b) \in M$ for every $(a, b)$ in the correct population;
(ii) $m_{w}(a, b) = m_{c}(a, b) + \xi(a, b)$ where, after the tangent–normal re-centering of Lemma 3.2 below, $\xi(a, b) \in N_{m_{c}(a, b)} M$;
(iii) $\{\varepsilon_{p}^{(i)}\}_{i}$ are independent zero-mean sub-Gaussian random vectors with covariance $\Sigma_{\varepsilon}$ and Orlicz $\psi_{2}$-norm at most $\sigma$: $\|\langle u, \varepsilon\rangle\|_{\psi_{2}} \leq \sigma \|u\|_{2}$ for all $u \in \mathbb{R}^{d}$;
(iv) $\{\xi(a, b)\}$ are independent of $\varepsilon$, with mean $\mu_{\xi} \in \mathbb{R}^{d}$ and covariance $\Sigma_{\xi}$, and i.i.d. across $(a, b)$ in the wrong population.

**Lemma 3.2 (Without-loss-of-generality re-centering of $\xi$ via the exponential map).**
Let $p := m_{c}(a, b) \in M$ and $\xi(a, b) \in \mathbb{R}^{d}$ be any perturbation with $\|\xi\|_2 < \tau(M)/2$. Decompose $\xi = P^T_p \xi + P^N_p \xi$ and define the re-centered base point and re-centered residual via the exponential map of $M$ at $p$:

$$
\widetilde{m}_{c}(a, b) := \exp_p\bigl(P^T_p \xi(a, b)\bigr) \in M,
\quad
\widetilde{\xi}(a, b) := \xi(a, b) - \bigl(\widetilde{m}_c(a, b) - p\bigr).
$$

(Here $\exp_p: T_p M \to M$ is the Riemannian exponential map of the embedded submanifold $M$, which sends a tangent vector $v \in T_p M$ to the point on $M$ obtained by following the geodesic from $p$ in direction $v$ for unit time. For a 1D curve $M_C$ with arc-length parameterization, $\exp_{g(t)}(c \cdot g'(t)/\|g'(t)\|) = g(t + c)$.)

Then $\widetilde{m}_{c}(a, b) \in M$ exactly (by definition of the exponential map), and

$$
\bigl\|\widetilde{\xi}(a, b) - P^N_p \xi(a, b)\bigr\|_2 \;\leq\; \tfrac{1}{2}\,\kappa_{\max}\, \|P^T_p \xi\|_2^2.
$$

In particular, $\widetilde{\xi}(a, b)$ differs from the normal component $P^N_p \xi$ by a curvature-controlled second-order term, and $\widetilde{\xi}(a, b) \in N_{\widetilde{m}_c(a,b)} M$ to first order in $\kappa_{\max} \|P^T_p \xi\|$.

*Proof outline.* The exponential map satisfies $\exp_p(v) = p + v + \tfrac{1}{2} II_p(v, v) + O(\kappa^2 \|v\|^3)$ where $II_p$ is the second fundamental form taking values in $N_p M$ (Lee, *Introduction to Riemannian Manifolds* 2018, Proposition 5.19). Substituting $v = P^T_p \xi$ and using $\|II_p\|_{\mathrm{op}} \leq \kappa_{\max}$ gives

$$
\widetilde{m}_c - p = P^T_p \xi + \tfrac{1}{2} II_p(P^T_p \xi, P^T_p \xi) + O(\kappa_{\max}^2 \|P^T_p \xi\|^3).
$$

Then $\widetilde{\xi} = \xi - (\widetilde{m}_c - p) = P^N_p \xi - \tfrac{1}{2} II_p(P^T_p\xi, P^T_p\xi) + O(\kappa_{\max}^2 \|P^T_p \xi\|^3)$. The second fundamental form value $II_p(\cdot, \cdot)$ lies in $N_p M$, so $\widetilde{\xi}$ is a normal-bundle vector to first order; the remaining $O(\kappa_{\max}^2 \|P^T_p \xi\|^3)$ correction is absorbed into the linearization error of §4.3. ∎

**Remark 3.3 (Same noise across populations).**
Assumption GM(iii) requires $\Sigma_{\varepsilon}$ to be the same for correct and wrong populations. This is realistic for activation noise driven by within-prompt token variation, but may fail if the wrong population is drawn from a categorically different prompt distribution. We address relaxation of this assumption in Section 4 Remark 4.9.

### 3.1 Boundedness assumption

**Assumption 3.4 (BD, boundedness).**
There exists $D_{\max} < \infty$ (depending on the model and layer) such that, with probability at least $1 - \delta$ over the noise, $\|h_{c}^{(i)}\|_{2} \leq D_{\max}$ for every correct activation in the analysis sample, and similarly for wrong activations.

For sub-Gaussian noise with parameter $\sigma$ and a clean signal $m_{p}(a, b)$ uniformly bounded on a compact $M$ by $\|m_p\|_\infty$, the standard sub-Gaussian maximal inequality over $n$ samples gives, with probability at least $1 - \delta$,

$$
D_{\max} \;=\; \|m_p\|_\infty + \sigma\bigl(\sqrt{d} + \sqrt{\log(n/\delta)}\bigr).
$$

The two terms have different origins: $\sigma \sqrt{d}$ is the typical Euclidean norm of an isotropic $d$-dimensional sub-Gaussian vector (from $\mathbb{E}\|\varepsilon\|^2 = O(d \sigma^2)$), and $\sigma \sqrt{\log(n/\delta)}$ is the maximum-over-$n$-samples penalty (sub-Gaussian maximal inequality). Combining them under a single square root, as in $\sigma \sqrt{d + \log(1/\delta)}$, conflates the two regimes and is wrong when $n \gg 1$. For activations after layer normalization, $\|m_p\|_\infty = O(\sqrt{d})$ typically, giving $D_{\max} = O(\sqrt{d} + \sigma \sqrt{\log(n/\delta)})$.

### 3.2 Regularity assumption on the manifold

**Assumption 3.5 (REG, regularity).**
There exist $\tau_{\min} > 0$ and $\kappa_{\max} < \infty$ such that $\tau(M) \geq \tau_{\min}$ uniformly and the maximum extrinsic curvature (operator norm of the second fundamental form $\|II_p\|_{\mathrm{op}}$) of $M$ is bounded by $\kappa_{\max}$ at every point of the data support. Moreover $\sigma \cdot \kappa_{\max} \leq c_{0}$ for a small absolute constant $c_{0} \in (0, 1]$.

The condition $\sigma \kappa_{\max} \leq c_{0}$ ensures that the typical noise displacement is small relative to the manifold's curvature scale, so the linearization of Section 4.3 is valid. We verify this empirically per model and layer before running the main experiments.

**Remark 3.6 (Empirical diagnostics for REG and the common-normal-bundle assumption).**
Both the reach lower bound $\tau(M) \geq \tau_{\min}$ and the curvature upper bound $\|II_p\|_{\mathrm{op}} \leq \kappa_{\max}$ are not directly observable but can be estimated from the correct-population activations:

- *Reach lower bound:* compute $\widehat\tau \geq \min_{i \neq j} \tfrac{\|h_c^{(i)} - h_c^{(j)}\|^2}{2 \cdot d(h_c^{(i)} - h_c^{(j)}, T_{p_j} M)}$ over a sample of pairs $(i, j)$, where $T_{p_j} M$ is estimated from local PCA at $p_j = h_c^{(j)}$. This is the Aamari–Levrard [aamari2019] reach estimator, consistent up to lower-order terms in $n_c$.
- *Curvature upper bound:* compute $\widehat\kappa_{\max}$ as the maximum operator norm of the local Hessian of the parametric helix curve, $\widehat{II}_p \approx -g''(t)$ at integer $t$ where $g(t)$ is from Definition 2.1. Closed form: $\|II\|_{\mathrm{op}} = \max_t \sup_{u \perp g'(t), \|u\|=1} |\langle u, g''(t)\rangle|$.
- *Common-normal-bundle drift $\eta$ for Theorem 5.2(C):* sample $\{p_k\}_{k=1}^K \subset M$, estimate $P^N_{p_k}$ from local PCA, and compute $\widehat\eta = \max_{k} \|P_V (P^N_{p_k} - P^N_{p_0}) P_V\|_{\mathrm{op}}$ for the chosen $V$ and a reference $p_0$.

We pre-register reporting $\widehat\tau, \widehat\kappa_{\max}, \widehat\sigma_{\mathrm{eff}}, \widehat\eta$ for each (model, layer) pair before computing the test statistic, and treat the test as conditional on the joint event that $\widehat\sigma_{\mathrm{eff}} \widehat\kappa_{\max} \leq c_0$ (the small-noise regime) and $\widehat\eta \leq \eta_0$. Layers failing these diagnostics are excluded from the analysis.

---

## 4. Proposed Theorem 1: validity of the test

We propose to prove the following theorem. The statement and proof outline below are what we have worked out so far; we expect the specific constants, and possibly some of the assumptions, to change during review.

### 4.1 Definition of the test statistic

**Definition 4.1 (The off-manifold test statistic).**
Given correct activations $H_c = \{h_{c}^{(i)}\}_{i=1}^{n_c}$ and wrong activations $H_w = \{h_{w}^{(j)}\}_{j=1}^{n_w}$, we propose the test statistic

$$
T_{n} := \frac{1}{n_w} \sum_{j=1}^{n_w} \|r(h_{w}^{(j)})\|_{2}^{2}
- \frac{1}{n_c} \sum_{i=1}^{n_c} \|r(h_{c}^{(i)})\|_{2}^{2},
$$

where $r(h) = h - \Pi_{M}(h)$ uses the *true* manifold $M$. The sample-estimated version $T_{n}^{\widehat{M}}$ replaces $M$ by an estimator $\widehat{M}$; this is the subject of Section 6.

### 4.2 Theorem 1 (proposed): validity of the test

**Theorem 4.2 (Proposed: validity and concentration of $T_{n}$).**
Suppose Assumption 3.1 (GM) holds with isotropic Gaussian noise $\Sigma_{\varepsilon} = \sigma^{2} I_{d}$, Assumption 3.4 (BD), and Assumption 3.5 (REG). Let $k := d - \dim(M)$ be the dimension of the residual (normal-bundle) subspace, and let $n := \min(n_c, n_w)$. We propose to prove that:

(a) **Population mean (alternative).** Under $H_{1}: \mu_{\xi} \neq 0 \text{ or } \Sigma_{\xi} \neq 0$,

$$
\mathbb{E}[T_{n}] = \|\mu_{\xi}\|_{2}^{2} + \mathrm{tr}(\Sigma_{\xi})
+ R_{1}(\sigma, \kappa_{\max}),
$$

where $R_{1}(\sigma, \kappa_{\max}) = O(\sigma^{2} \kappa_{\max}^{2})$ is a linearization-error term that vanishes when $M$ is linear. The mean of $T_n$ depends only on the perturbation $\xi$, not on the noise $\varepsilon$: under Assumption GM(iii) the two populations share $\Sigma_\varepsilon$, so the noise traces $\mathrm{tr}(P^N \Sigma_\varepsilon P^N)$ cancel exactly across the difference $\bar{Y}_w - \bar{Y}_c$ (see Corollary 4.6 below; Remark 4.9 handles the case $\Sigma_\varepsilon^c \neq \Sigma_\varepsilon^w$).

(b) **Null behavior.** Define $H_{0}: \mu_{\xi} = 0$ and $\Sigma_{\xi} = 0$ — equivalently, $\xi(a, b) \equiv 0$, so the wrong-population signal coincides with the correct-population signal in distribution. Under $H_0$, $\mathbb{E}[T_{n}] = R_{1}(\sigma, \kappa_{\max})$ which is identically zero when $M$ is linear and $O(\sigma^{2} \kappa_{\max}^{2})$ in general. We emphasize that the null is a statement about the perturbation $\xi$ being zero, not about the noise $\varepsilon$: the ε-contribution to $\mathbb{E}[T_n]$ cancels regardless of whether $\xi$ vanishes.

(c) **Non-asymptotic concentration.** For every $t > 0$,

$$
\mathbb{P}\bigl[\,\bigl|T_{n} - \mathbb{E}[T_{n}]\bigr| > t\,\bigr]
\;\leq\; 4 \exp\!\Bigl(- c \cdot n \cdot \min\!\bigl(\tfrac{t^{2}}{k \sigma^{4}},\, \tfrac{t}{\sigma^{2}}\bigr)\Bigr), \tag{4.1}
$$

with absolute constant $c \geq 1/8$ (the Laurent–Massart sub-exponential chi-square constant).

(d) **Sharp Laurent–Massart form.** For any $u > 0$ and $n_{c} = n_{w} = n$,

$$
\mathbb{P}\!\left[\,T_{n} - \mathbb{E}[T_{n}] \geq 2 \sigma^{2} \sqrt{\tfrac{2 k u}{n}} + \tfrac{2 \sigma^{2} u}{n}\,\right]
\;\leq\; 2 e^{-u}. \tag{4.2}
$$

The same bound holds for the lower tail (replace $\geq$ by $\leq -\cdots$).

The remainder of this section gives our proposed proof outline for Theorem 4.2. The outline has four steps: (1) linearize $\Pi_{M}$ around clean points, (2) compute the population mean, (3) derive the variance, (4) apply Laurent–Massart to get the sharp tail bound. We have worked through each step at the level of detail below; we believe the structure is correct but expect the constants in Step 4 in particular to need tightening.

### 4.3 Step 1: linearization of $\Pi_{M}$

**Lemma 4.3 (Tubular-neighborhood projection — proposed, explicit form).**
Let $M \subset \mathbb{R}^d$ be a smooth submanifold with reach $\tau(M) \geq \tau_{\min} > 0$ and second-fundamental-form bound $\sup_{p \in M} \|II_p\|_{\mathrm{op}} \leq \kappa_{\max}$. Let $p \in M$ and $\delta \in \mathbb{R}^{d}$ with $\|\delta\|_{2} < \tau_{\min} / 2$. Decompose $\delta = P^T_p \delta + P^N_p \delta$ (tangent and normal components). Then

$$
\Pi_{M}(p + \delta) = p + P^T_{p} \delta + R_{2}(\delta, p),
\quad
\|R_{2}(\delta, p)\|_{2}
\;\leq\;
\frac{\kappa_{\max}}{2(1 - \kappa_{\max}\|P^N_p \delta\|_2)} \|P^N_{p} \delta\|_{2}^{2}, \tag{4.3a}
$$

provided $\kappa_{\max} \|P^N_p \delta\|_2 < 1$ (which holds whenever $\|\delta\|_2 \leq \tau_{\min}/2$ since $\tau_{\min} \leq 1/\kappa_{\max}$ locally). Equivalently, in the small-noise regime $\sigma \kappa_{\max} \leq c_0 < 1$ of Assumption REG,

$$
\|R_{2}(\delta, p)\|_2 \;\leq\; \frac{\kappa_{\max}}{2(1 - c_0)} \|P^N_p \delta\|_2^2. \tag{4.3b}
$$

The factor $\|P^N_p \delta\|_2^2$ rather than $\|\delta\|_2^2$ is essential: the tangent component $P^T_p \delta$ is a first-order motion along $M$ and contributes only $O(\kappa_{\max}^2 \|P^T_p \delta\|^3)$ to the second-order remainder, which is one order of magnitude smaller than the normal-component contribution.

Consequently the residual at $h = p + \delta$ should satisfy

$$
r(h) = P^N_{p} \delta - R_{2}(\delta, p),
\quad
\|r(h) - P^N_p \delta\|_2 \;\leq\; \tfrac{\kappa_{\max}}{2(1 - c_0)} \|P^N_p \delta\|_2^2. \tag{4.3c}
$$

*Proof outline.* This is the standard tubular-neighborhood estimate for embedded $C^2$ submanifolds with positive reach. Federer [federer1959] §4.18 establishes that $\Pi_M$ is $C^{1, 1}$ on the open tube $\{x : d(x, M) < \tau(M)\}$. The derivative at $p$ is $D \Pi_M(p) = P^T_p$ and the Hessian (in directions normal to $M$) is bounded by the second fundamental form. Niyogi–Smale–Weinberger [niyogi2008] Lemma 4.1 gives the precise Taylor remainder for normal perturbations:

$$
\Pi_M(p + \delta_N) - p - P^T_p \delta_N \cdot 0 = -\tfrac{1}{2} II_p(\delta_N, \delta_N) + O(\kappa_{\max}^2 \|\delta_N\|^3),
$$

where $\delta_N := P^N_p \delta$ and $II_p(\cdot, \cdot)$ is the second fundamental form, an $\mathbb{R}^d$-valued symmetric bilinear form on $T_p M$ whose values lie in $N_p M$. The operator-norm bound $\|II_p\|_{\mathrm{op}} \leq \kappa_{\max}$ then gives the leading-order $\tfrac{1}{2}\kappa_{\max} \|\delta_N\|^2$ bound. The denominator $(1 - c_0)$ in (4.3b) is the standard refinement that absorbs the higher-order terms inside the tube; see Aamari–Levrard [aamari2019] Proposition A.1 for an explicit modern statement. The tangent component contributes no second-order term because $\Pi_M(p + P^T_p \delta) = p + P^T_p \delta + O(\kappa_{\max}^2 \|P^T_p\delta\|^3)$ — it stays on the manifold to within third order. ∎

**Corollary 4.4 (Proposed: residual under Assumption GM).**
Under Assumption 3.1 with $h_{c} = m_{c}(a, b) + \varepsilon_{c}$,

$$
r(h_{c}) = P^N_{m_{c}(a, b)} \varepsilon_{c} - R_{2}(\varepsilon_{c}, m_{c}(a, b)),
$$

and similarly for wrong activations $h_{w} = m_{c}(a, b) + \xi(a, b) + \varepsilon_{w}$,

$$
r(h_{w}) = \xi(a, b) + P^N_{m_{c}(a, b)} \varepsilon_{w} - R_{2}(\xi + \varepsilon_{w}, m_{c}(a, b)).
$$

In what follows we write the leading-order expressions and absorb $R_{2}$ into the error term $R_{1}$ of Theorem 4.2. The bound $\|R_{1}\| = O(\sigma^{2} \kappa_{\max}^{2})$ comes from $\mathbb{E}[\|R_{2}(\varepsilon, p)\|_{2}^{2}] = O(\sigma^{4} \kappa_{\max}^{2})$ and Cauchy–Schwarz applied to the cross-term in the squared residual expansion.

### 4.4 Step 2: the population mean of $T_n$

**Lemma 4.5 (Mean of squared residual — proposed).**
Under Assumptions GM, BD, REG and the linearization of Lemma 4.3, conditional on $a, b$ we propose:

$$
\begin{aligned}
\mathbb{E}[\|r(h_{c})\|_{2}^{2} \mid a, b] &= \mathrm{tr}(P^N_{m_{c}(a, b)} \Sigma_{\varepsilon} P^N_{m_{c}(a, b)}) + O(\sigma^{2} \kappa_{\max}^{2}), \\
\mathbb{E}[\|r(h_{w})\|_{2}^{2} \mid a, b] &= \|\mu_{\xi}\|_{2}^{2} + \mathrm{tr}(\Sigma_{\xi}) + \mathrm{tr}(P^N_{m_{c}(a, b)} \Sigma_{\varepsilon} P^N_{m_{c}(a, b)}) + O(\sigma^{2} \kappa_{\max}^{2}).
\end{aligned}
$$

*Proof outline.* For the correct case, expand

$$
\mathbb{E}[\|P^N \varepsilon_{c}\|_{2}^{2}] = \mathbb{E}[\varepsilon_{c}^{\top} P^N \varepsilon_{c}]
= \mathrm{tr}(P^N \mathbb{E}[\varepsilon_{c} \varepsilon_{c}^{\top}]) = \mathrm{tr}(P^N \Sigma_{\varepsilon}).
$$

Using $P^N = (P^N)^{\top} = (P^N)^{2}$, this equals $\mathrm{tr}(P^N \Sigma_{\varepsilon} P^N)$. The remainder term comes from Corollary 4.4.

For the wrong case, write $r(h_{w}) = \xi + P^N \varepsilon_{w}$ to leading order (Corollary 4.4). Then

$$
\|r(h_{w})\|_{2}^{2}
= \|\xi\|_{2}^{2} + 2 \langle \xi, P^N \varepsilon_{w}\rangle + \|P^N \varepsilon_{w}\|_{2}^{2}.
$$

Take expectations. The cross term should vanish because $\mathbb{E}[P^N \varepsilon_{w}] = P^N \mathbb{E}[\varepsilon_{w}] = 0$ and $\xi \perp \varepsilon_{w}$ by Assumption GM(iv). For the squared-norm term,

$$
\mathbb{E}[\|\xi\|_{2}^{2}] = \|\mathbb{E}[\xi]\|_{2}^{2} + \mathrm{tr}(\mathrm{Cov}(\xi))
= \|\mu_{\xi}\|_{2}^{2} + \mathrm{tr}(\Sigma_{\xi}).
$$

Combining gives the stated expression. ∎

**Corollary 4.6 (Proposed: mean of $T_{n}$).**
Under Assumptions GM, BD, REG with isotropic noise $\Sigma_{\varepsilon} = \sigma^{2} I$,

$$
\mathbb{E}[T_{n}] = \|\mu_{\xi}\|_{2}^{2} + \mathrm{tr}(\Sigma_{\xi}) + O(\sigma^{2} \kappa_{\max}^{2}).
$$

For anisotropic noise, the trace terms cancel exactly across populations provided the same $\Sigma_{\varepsilon}$ is used (Assumption GM(iii)), and the formula is unchanged.

*Proof outline.* The trace terms $\mathrm{tr}(P^N \Sigma_{\varepsilon} P^N)$ are identical for correct and wrong populations under Assumption GM(iii) and cancel in the difference $T_{n}$. The remaining terms give the stated expression. ∎

### 4.5 Step 3: variance of $T_n$ via chi-squared analysis

**Lemma 4.7 (Sub-exponential parameter of squared residual — standard).**
Under Assumption GM with isotropic Gaussian noise $\Sigma_{\varepsilon} = \sigma^{2} I$, the centered squared residual norm $X_{i} := \|r(h_{i})\|_{2}^{2} - \mathbb{E}[\|r(h_{i})\|_{2}^{2}]$ satisfies, for both populations,

$$
\mathbb{P}[\,X_{i} \geq 2 \sigma^{2} \sqrt{k u} + 2 \sigma^{2} u\,] \leq e^{-u},
\quad
\mathbb{P}[\,X_{i} \leq -2 \sigma^{2} \sqrt{k u}\,] \leq e^{-u}, \tag{4.4}
$$

for every $u > 0$. In particular $X_{i}$ is sub-exponential with parameters $(\nu, \alpha) = (2 \sigma^{2} \sqrt{k}, 2 \sigma^{2})$.

*Proof outline.* For correct activations, $P^N_{p} \varepsilon \sim \mathcal{N}(0, \sigma^{2} P^N_{p})$ in $\mathbb{R}^{d}$. Since $P^N_{p}$ is the orthogonal projection onto a $k$-dimensional subspace, $\|P^N_{p} \varepsilon\|_{2}^{2} \sim \sigma^{2} \chi^{2}_{k}$. Laurent–Massart [laurent2000] Lemma 1 states that for $Z \sim \chi^{2}_{k}$ and $u > 0$:

$$
\mathbb{P}[Z - k \geq 2 \sqrt{k u} + 2 u] \leq e^{-u},
\quad
\mathbb{P}[Z - k \leq -2 \sqrt{k u}] \leq e^{-u}. \tag{4.5}
$$

Multiplying through by $\sigma^{2}$ gives the stated bound for the correct case. For wrong activations, $\|r(h_{w})\|_{2}^{2}$ has the same chi-squared structure plus a deterministic shift $\|\xi\|^{2}$ plus a Gaussian cross-term; the centered version inherits sub-exponential tails with the same parameters up to a constant absorbed into $c$. ∎

**Lemma 4.8 (Concentration of empirical mean — standard).**
Let $X_{1}, \dots, X_{n}$ be i.i.d. centered sub-exponential random variables satisfying (4.4). Then for any $u > 0$,

$$
\mathbb{P}\!\left[\,\bar{X}_{n} \geq 2 \sigma^{2} \sqrt{\tfrac{k u}{n}} + \tfrac{2 \sigma^{2} u}{n}\,\right] \leq e^{-u}. \tag{4.6}
$$

*Proof outline.* The sum $S_{n} = \sum_{i} X_{i}$ has the moment-generating-function bound, valid for $|\lambda| < (2 \sigma^{2})^{-1}$:

$$
\mathbb{E}[e^{\lambda S_{n}}] = \prod_{i=1}^{n} \mathbb{E}[e^{\lambda X_{i}}]
\leq \exp\!\bigl(\tfrac{\lambda^{2} \cdot n \cdot 4 k \sigma^{4}}{2 (1 - 2 \sigma^{2} |\lambda|)}\bigr).
$$

This is the standard Bernstein-MGF for sub-exponential variables (Wainwright [wainwright2019] Proposition 2.9). Apply Markov to $\lambda \bar{X}_{n}$ and optimize over $\lambda$: choose

$$
\lambda^{*} = \frac{u}{u \cdot 2 \sigma^{2} + n \cdot 2 \sigma^{2} \sqrt{k u / n}}
$$

which yields the stated bound after simplification. ∎

### 4.6 Step 4: union bound and Theorem 1 conclusion

*Proposed proof of Theorem 4.2.* Write $T_{n} = \bar{Y}_{w} - \bar{Y}_{c}$ where $\bar{Y}_{p} = \frac{1}{n_{p}} \sum_{i} \|r(h_{i,p})\|_{2}^{2}$. By Lemmas 4.5 and 4.8,

$$
\mathbb{P}\!\left[\,\bar{Y}_{w} - \mathbb{E}[\bar{Y}_{w}] \geq 2 \sigma^{2} \sqrt{\tfrac{k u}{n_w}} + \tfrac{2 \sigma^{2} u}{n_w}\,\right] \leq e^{-u},
$$

and the same with $w \to c$ and $n_w \to n_c$. Apply the union bound to the four events (upper and lower tails, two populations) at level $u$ each, with total tail $4 e^{-u}$. Choose $u = \log(4/\delta)$.

For the symmetric form (4.1), set $u = c n \min(t^{2}/(k \sigma^{4}), t/\sigma^{2})$ in (4.6); this yields the deviation bound $|T_{n} - \mathbb{E}[T_{n}]| > t$ with the stated exponential rate. The constant $c \geq 1/8$ comes from the Laurent–Massart proof's specific moment-generating function expansion with universal numerical constants. ∎

**Remark 4.9 (Different noise across populations).**
If $\Sigma_{\varepsilon}^{c} \neq \Sigma_{\varepsilon}^{w}$, Lemma 4.5 still holds with $\Sigma_{\varepsilon}$ replaced by $\Sigma_{\varepsilon}^{p}$ in each line. The trace terms no longer cancel in $T_{n}$, and we instead obtain

$$
\mathbb{E}[T_{n}] = \|\mu_{\xi}\|^{2} + \mathrm{tr}(\Sigma_{\xi}) + \mathrm{tr}(P^N (\Sigma_{\varepsilon}^{w} - \Sigma_{\varepsilon}^{c}) P^N).
$$

The last term is a noise-distribution mismatch and can be bounded by $k \cdot \|\Sigma_{\varepsilon}^{w} - \Sigma_{\varepsilon}^{c}\|_{\mathrm{op}}$. We test for this empirically by comparing residual covariance matrices between correct and wrong populations and reporting the bound.

**Theorem 4.10 (Proposed: anisotropic version of Theorem 4.2).**
Replace the isotropic-noise hypothesis $\Sigma_\varepsilon = \sigma^2 I$ in Theorem 4.2 by an arbitrary positive-semidefinite $\Sigma_\varepsilon$ with $\|\Sigma_\varepsilon\|_{\mathrm{op}} \leq \sigma_{\mathrm{op}}^2$. Define the *effective normal-bundle rank* and *effective normal-bundle scale*:

$$
k_{\mathrm{eff}} := \frac{(\mathrm{tr}(P^N \Sigma_{\varepsilon}))^{2}}{\mathrm{tr}((P^N \Sigma_{\varepsilon})^{2})},
\quad
\sigma_{\mathrm{eff}}^2 := \frac{\mathrm{tr}(P^N \Sigma_\varepsilon)}{k_{\mathrm{eff}}} = \frac{\mathrm{tr}((P^N \Sigma_\varepsilon)^2)}{\mathrm{tr}(P^N \Sigma_\varepsilon)}.
$$

For isotropic $\Sigma_{\varepsilon} = \sigma^{2} I$, $k_{\mathrm{eff}} = k$ and $\sigma_{\mathrm{eff}}^2 = \sigma^2$. Then we propose:

(a') **Population mean.** Same as Theorem 4.2(a), since the noise traces $\mathrm{tr}(P^N \Sigma_\varepsilon P^N)$ cancel exactly across populations under GM(iii) regardless of anisotropy.

(c') **Non-asymptotic concentration.** For every $u > 0$ and $n_c = n_w = n$,

$$
\mathbb{P}\!\left[T_n - \mathbb{E}[T_n] \geq 2 \sigma_{\mathrm{eff}}^2 \sqrt{\tfrac{2 k_{\mathrm{eff}} u}{n}} + \tfrac{2 \sigma_{\mathrm{op}}^2 u}{n}\right] \leq 2 e^{-u}, \tag{4.7}
$$

with the same bound for the lower tail.

*Proof outline.* Apply the Hanson–Wright inequality (Vershynin [vershynin2018] Theorem 6.2.1) directly to the centered quadratic form $\|P^N \varepsilon\|_2^2 - \mathrm{tr}(P^N \Sigma_\varepsilon)$ with $\varepsilon$ sub-Gaussian: there exist absolute constants $c_1, c_2$ such that

$$
\mathbb{P}\!\bigl[\bigl|\|P^N \varepsilon\|_2^2 - \mathrm{tr}(P^N \Sigma_\varepsilon)\bigr| > t\bigr]
\;\leq\;
2 \exp\!\Bigl(- c_1 \min\!\bigl(\tfrac{t^2}{\mathrm{tr}((P^N \Sigma_\varepsilon)^2)}, \tfrac{t}{\|P^N \Sigma_\varepsilon\|_{\mathrm{op}}}\bigr)\Bigr).
$$

Equivalently, the centered statistic is sub-exponential with parameters $(\nu, \alpha) = (2\sqrt{\mathrm{tr}((P^N \Sigma_\varepsilon)^2)}, 2 \|P^N \Sigma_\varepsilon\|_{\mathrm{op}})$ — no $\Sigma_\varepsilon^{-1/2}$ rescaling is needed (and would not be defined when $\Sigma_\varepsilon$ is singular). Substituting $\nu^2 = 4 k_{\mathrm{eff}} \sigma_{\mathrm{eff}}^4$ and $\alpha = 2 \sigma_{\mathrm{op}}^2$ where $\sigma_{\mathrm{op}}^2 := \|P^N \Sigma_\varepsilon\|_{\mathrm{op}}$, then averaging over $n$ samples and applying Bernstein-MGF as in Lemma 4.8, gives (4.7).

**Estimating $k_{\mathrm{eff}}$ from data.** A consistent estimator from the correct population alone is

$$
\widehat{k}_{\mathrm{eff}} := \frac{(\mathrm{tr}(\widehat{\Sigma}_r))^2}{\mathrm{tr}(\widehat{\Sigma}_r^2)},
\quad
\widehat{\Sigma}_r := \frac{1}{n_c} \sum_{i=1}^{n_c} r(h_c^{(i)}) r(h_c^{(i)})^\top,
$$

the empirical normal-bundle residual covariance restricted to $V$ (or to the full normal bundle for the global test). Concentration of $\widehat{k}_{\mathrm{eff}}$ to $k_{\mathrm{eff}}$ at rate $O(\sqrt{k/n_c})$ follows from standard sample-covariance concentration (Vershynin [vershynin2018] Section 4.7).

**Remark.** Real activations after layer normalization are unlikely to be isotropic. We will report $\widehat{k}_{\mathrm{eff}}$ alongside the test statistic and use Theorem 4.10 rather than Theorem 4.2 for calibration. Theorem 5.2's localized bounds extend analogously with $r$ replaced by $r_{\mathrm{eff}} := (\mathrm{tr}(P_V P^N \Sigma_\varepsilon))^2 / \mathrm{tr}((P_V P^N \Sigma_\varepsilon)^2)$.

**Remark 4.11 (Berry–Esseen rate to normality).**
By the Berry–Esseen theorem with the modern Esseen constant [tyurin2010],

$$
\sup_{t \in \mathbb{R}} \left|\mathbb{P}\!\left[\tfrac{T_{n} - \mathbb{E}[T_{n}]}{\sqrt{\mathrm{Var}(T_{n})}} \leq t\right] - \Phi(t)\right|
\;\leq\;
\frac{0.4748 \cdot \mathbb{E}|X_{1} - \mathbb{E}[X_{1}]|^{3}}{\sqrt{n} \cdot (\mathrm{Var}(X_{1}))^{3/2}}.
$$

For a chi-squared random variable $Z \sim \chi^2_k$ scaled by $\sigma^2$, the third central moment is $8 k \sigma^6$ and the variance is $2 k \sigma^4$, so the Lyapunov ratio is $8k\sigma^6 / (2k\sigma^4)^{3/2} = 8k / (2\sqrt 2 \cdot k^{3/2} \sigma^0) = \tfrac{2\sqrt{2}}{\sqrt{k}}$. Substituting:

$$
\sup_{t} |\cdots| \;\leq\; \frac{0.4748 \cdot 2\sqrt{2}}{\sqrt{n k}} \;=\; O\!\left(\tfrac{1}{\sqrt{n k}}\right).
$$

The rate *improves* with $k$ rather than degrading: a high-dimensional residual gives a sharper Gaussian approximation, not a coarser one (the chi-squared distribution is closer to Gaussian when $k$ is large, by CLT applied internally to $\chi^2_k = \sum_i Z_i^2$). For fixed $k$, the rate is the classical $O(n^{-1/2})$. An earlier draft incorrectly stated $O(\sqrt{k/n})$, which inverted the scaling.

---

## 5. Proposed Theorem 2: localization power

We now propose a localized version of the test. The headline mathematical content of this section — and the most novel part of the proposal — is a minimax lower bound matching the achievable rate, which would show that the localized test cannot be improved beyond constants. We have worked out the lower-bound argument by Le Cam two-point reduction plus Fano chaining at the level below; it is the place where we most need Barnábás's input on the chaining step.

### 5.1 The localized test statistic

**Definition 5.1 (Localized test statistic).**
Let $V \subset \mathbb{R}^{d}$ be an $r$-dimensional linear subspace with orthonormal basis $U_{V} \in \mathbb{R}^{d \times r}$, and let $P_V = U_{V} U_{V}^{\top}$ be the orthogonal projection onto $V$. The localized test statistic is

$$
T_{n}^{V} := \frac{1}{n_w} \sum_{j=1}^{n_w} \|P_V r(h_{w}^{(j)})\|_{2}^{2}
- \frac{1}{n_c} \sum_{i=1}^{n_c} \|P_V r(h_{c}^{(i)})\|_{2}^{2}.
$$

### 5.2 Theorem 2 (proposed): power and minimax optimality

**Theorem 5.2 (Proposed: localization power; lower bound conjectural).**
Suppose Assumptions GM, BD, REG hold with isotropic Gaussian noise. Suppose $\xi(a, b) \in V$ almost surely (so $\mu_{\xi} \in V$ and $\Sigma_{\xi}$ has support in $V$), where $V$ is a fixed $r$-dimensional subspace, and *one of the following holds*:

- **(L) Linear-span case.** $M = M_S$ is the linear (helix) span of Definition 2.3 (after centering). Then $N_p M = M_S^\perp$ is constant over $p \in M$, the common normal subspace assumption $V \subseteq \bigcap_p N_p M = M_S^\perp$ is automatic, and the theorem holds as stated below.

- **(C) Curve case with bounded normal-frame drift.** $M = M_C$ is the helix curve and $V \subseteq N_{p_0} M$ for a fixed reference base point $p_0$, with the *normal-frame drift constant*
$$
\eta := \sup_{p \in M\,:\,\Pi_M^{-1}(p) \cap \mathrm{supp}(h_w) \neq \emptyset} \bigl\|P_V (P^N_p - P^N_{p_0}) P_V\bigr\|_{\mathrm{op}}
$$
satisfying $\eta \leq \eta_0$ for a small constant $\eta_0$. Then the conclusions below hold with an additional additive error $O(\eta_0 D_{\max}^2)$ in $\mathbb{E}[T_n^V]$ and an additional $O(\eta_0 r \sigma^2)$ in the variance.

The strict assumption $V \subseteq \bigcap_p N_p M$ used in earlier drafts holds in case (L) trivially, and in case (C) only with $\eta = 0$ — which fails for a curved 1D manifold whose normal space rotates along the curve. Case (C) replaces it by an explicit, empirically verifiable drift bound.

We propose:

(a) **Achievability (upper bound).** For the test that rejects $H_{0}$ when $T_{n}^{V} > c_{\alpha}$ with $c_{\alpha}$ calibrated to level $\alpha$, the test should achieve power at least $1 - \beta$ provided

$$
n \;\geq\; C_{1} \cdot \frac{r \sigma^{4}}{\Delta^{4}} \cdot \log\!\bigl(\tfrac{1}{\beta}\bigr), \tag{5.1}
$$

where $\Delta := \|\mu_{\xi}\|_{2}$ and $C_{1}$ is an absolute constant.

(b) **Two-point Le Cam lower bound (proven).** Let $\mathcal{P}_{\Delta}$ denote the class of distributions in Assumption GM with $\|\mu_{\xi}\|_{2} \geq \Delta$ and $\xi \in V$. Any test $\psi$ with size at most $\alpha$ that achieves uniform power $\inf_{P \in \mathcal{P}_{\Delta}} \mathbb{P}_{P}[\psi = 1] \geq 1 - \beta$ requires

$$
n \;\geq\; C_{2}^{\mathrm{LC}} \cdot \frac{\sigma^{2}}{\Delta^{2}} \cdot \log\!\bigl(\tfrac{1}{\beta(1 - \alpha)}\bigr), \tag{5.2a}
$$

for an absolute constant $C_{2}^{\mathrm{LC}}$. The proof is a direct two-point Le Cam reduction (Lemmas 5.5–5.6 below). This rate matches the upper bound in $\sigma^{2}/\Delta^{2}$ but does *not* match the upper bound in $r$ or in the exponent of $\Delta$.

(b') **Conjectured matching minimax rate.** We conjecture, and do not prove in this document, that the rate (5.1) is minimax-optimal:

$$
n \;\geq\; C_{2} \cdot \frac{r \sigma^{4}}{\Delta^{4}} \cdot \log\!\bigl(\tfrac{1}{\beta(1 - \alpha)}\bigr). \tag{5.2b}
$$

The rate (5.2b) is the standard minimax separation rate for testing $\mathcal{N}(0, \sigma^2 I_r)$ against $\bigcup_{\|\mu\|=\Delta} \mathcal{N}(\mu, \sigma^2 I_r)$, established by Ingster [ingster1993, ingster2003] for the Gaussian sequence model and refined non-asymptotically by Baraud [baraud2002] and Collier–Comminges–Tsybakov [collier2017]. The argument in those works combines Le Cam with a chi-squared mixture over a packing of $\mathbb{S}^{r-1}(\Delta)$, not the simple Fano-over-orthogonal-directions argument we sketched in earlier drafts (which gives $r \sigma^{2}/\Delta^{2} \log r$, not the desired $r \sigma^{4}/\Delta^{4}$). Translating those results to our setting requires checking that the wrong-population conditional distribution under GM matches the Gaussian sequence model after the projection of Lemma 5.3, which we believe is straightforward but have not verified. We treat (5.2b) as a target for follow-up work.

(c) **Exact null distribution.** Under $H_{0}$ with isotropic Gaussian noise and oracle $M$, $T_{n}^{V}$ is distributed exactly as a difference of two scaled chi-squared random variables; $\bar{Y}_{w} = \frac{1}{n_w}\sum_j \|P_V r(h_w^{(j)})\|^2 \sim \frac{\sigma^2}{n_w}\chi^2_{n_w r}$ and similarly for $\bar{Y}_c$, with $T_n^V = \bar{Y}_w - \bar{Y}_c$. The asymptotic Wald form is

$$
\sqrt{\tfrac{n_c n_w}{n_c + n_w}} \cdot \frac{T_{n}^{V}}{\sigma^{2} \sqrt{2 r}}
\;\xrightarrow{d}\; \mathcal{N}(0, 1),
$$

so $T_n^V$ has mean $0$ and variance $2 r \sigma^4 (n_c + n_w)/(n_c n_w)$ to leading order, giving closed-form $p$-values without permutation.

The remainder of this section proves Theorem 5.2.

### 5.3 Step 1: reduction to projected space

**Lemma 5.3 (Proposed: projected residual under Assumption GM).**
Under the hypotheses of Theorem 5.2,

$$
\begin{aligned}
P_V r(h_{c}) &= P_V \varepsilon_{c} + e_c, \\
P_V r(h_{w}) &= \xi + P_V \varepsilon_{w} + e_w,
\end{aligned}
$$

with remainder bounds:

- *Linear-span case (L):* $\|e_c\|, \|e_w\| = O_{\mathbb{P}}(\sigma^{2} \kappa_{\max})$ from the linearization remainder of Lemma 4.3 only (κ_max = 0 for a linear M, so the remainder vanishes exactly).
- *Curve case (C):* $\|e_c\|, \|e_w\| = O_{\mathbb{P}}(\sigma^{2} \kappa_{\max}) + O_{\mathbb{P}}(\eta_0 \cdot (\sigma + \|\xi\|))$ where the second term is the normal-frame drift error.

*Proof outline.* By Corollary 4.4, $r(h_{c}) = P^N_{p} \varepsilon_{c} + R_2$ where $\|R_2\| \leq \tfrac{1}{2}\kappa_{\max} \|P^N \varepsilon_c\|^2$. Apply $P_V$:

$$
P_V r(h_{c}) = P_V P^N_{p} \varepsilon_{c} + P_V R_2.
$$

In case (L), $P^N_p = I - U_S U_S^\top$ is constant in $p$ and $V \subseteq M_S^\perp$, so $P_V P^N_p = P_V$ exactly and the only error is the linearization remainder $\|P_V R_2\| \leq \tfrac{1}{2}\kappa_{\max} \|P^N \varepsilon\|^2$, which is $O_\mathbb{P}(\sigma^2 \kappa_{\max})$ in expectation; for a linear span $\kappa_{\max} = 0$ and this vanishes.

In case (C), $V \subseteq N_{p_0} M$ at the reference point but $V \not\subseteq N_p M$ at general $p \in M$. Decompose $P_V P^N_p = P_V P^N_{p_0} + P_V (P^N_p - P^N_{p_0}) = P_V + P_V (P^N_p - P^N_{p_0})$, the second term having operator norm at most $\eta$ on $V$ by the drift definition. Hence

$$
\|P_V P^N_p \varepsilon - P_V \varepsilon\|_{2} \leq \eta \|\varepsilon\|_2,
$$

contributing an $O_\mathbb{P}(\eta_0 \sigma)$ term. For $r(h_{w}) = \xi + P^N \varepsilon_{w} + R_2$: $P_V \xi = \xi$ since $\xi \in V$, and the same drift analysis gives an additional $O_\mathbb{P}(\eta_0 \|\xi\|)$ term from $P_V P^N_p \varepsilon_w$. Combining gives the claim. ∎

The drift error $\eta_0$ is the *replacement* for the strict $V \subseteq \bigcap_p N_p M$ condition. It is empirically verifiable: estimate $P^N_p$ at a finite sample of base points along the curve and compute the maximum operator-norm drift restricted to $V$. In the helix-curve case, the tangent direction rotates at a rate set by $\|u_{\mathrm{lin}}\|$ relative to the cosine/sine amplitudes; we expect $\eta_0$ to be small after centering and after restricting $V$ to the high-frequency $T \in \{2,5\}$ components, and we plan to report $\widehat{\eta}_0$ alongside the test statistic.

### 5.4 Step 2: variance in projected space

**Lemma 5.4 (Proposed: variance reduction in projected space).**
Under the hypotheses of Theorem 5.2 and isotropic Gaussian noise,

$$
\begin{aligned}
\mathrm{Var}(\|P_V r(h_{c})\|_{2}^{2}) &= 2 r \sigma^{4}, \\
\mathrm{Var}(\|P_V r(h_{w})\|_{2}^{2}) &= 2 r \sigma^{4} + 4 \|\mu_{\xi}\|_{2}^{2} \sigma^{2} + O(\|\Sigma_{\xi}\|^{2}).
\end{aligned}
$$

*Proof outline.* For the correct case, $P_V \varepsilon_{c} \sim \mathcal{N}(0, \sigma^{2} P_V)$ in $V$ which is $r$-dimensional, so $\|P_V \varepsilon_{c}\|^{2} \sim \sigma^{2} \chi^{2}_{r}$. The variance of $\sigma^{2} \chi^{2}_{r}$ is $2 r \sigma^{4}$.

For the wrong case, write $W := P_V r(h_{w}) = \xi + P_V \varepsilon_{w}$ to leading order. Then

$$
\|W\|^{2} = \|\xi\|^{2} + 2 \langle \xi, P_V \varepsilon_{w}\rangle + \|P_V \varepsilon_{w}\|^{2}.
$$

The variance decomposes (using independence of $\xi$ and $\varepsilon_{w}$):

$$
\begin{aligned}
\mathrm{Var}(\|W\|^{2})
&= \mathrm{Var}(\|\xi\|^{2}) + \mathrm{Var}(2 \langle \xi, P_V \varepsilon_{w}\rangle) + \mathrm{Var}(\|P_V \varepsilon_{w}\|^{2}) + 2\,\mathrm{Cov}\text{-terms} \\
&= O(\|\Sigma_{\xi}\|^{2}) + 4 \|\mu_{\xi}\|^{2} \sigma^{2} + 2 r \sigma^{4} + (\text{lower order}).
\end{aligned}
$$

The middle Gaussian-cross variance equals $4 \mathbb{E}[\langle \xi, P_V \varepsilon_{w}\rangle^{2}] = 4 \|\mathbb{E}[\xi]\|^{2} \sigma^{2} + O(\sigma^{2} \mathrm{tr}\, \Sigma_{\xi})$; the dominant term in the typical hardness regime $\|\mu_{\xi}\|^{2} > \mathrm{tr}\, \Sigma_{\xi}$ is $4 \|\mu_{\xi}\|^{2} \sigma^{2}$. ∎

### 5.5 Step 3: achievability bound

*Proposed proof of Theorem 5.2(a).* By Lemma 4.5 restricted to $V$, $\mathbb{E}[T_{n}^{V}] = \|\mu_{\xi}\|^{2} + \mathrm{tr}(\Sigma_{\xi}|_{V})$. By Lemma 5.4, $\mathrm{Var}(T_{n}^{V}) \leq 2 r \sigma^{4}/n + 4 \Delta^{2} \sigma^{2} / n$. The second term dominates the first only when $\Delta^{2} \geq r \sigma^{2} / 2$, which is the easy regime; the hard regime (where the lower bound matters) is $\Delta \ll \sigma$.

Apply Lemma 4.8 adapted to $r$-dimensional projected chi-square: with probability $1 - \beta/2$,

$$
T_{n}^{V} \geq \mathbb{E}[T_{n}^{V}] - 2 \sigma^{2} \sqrt{\tfrac{r \log(2/\beta)}{n}} - \tfrac{2 \sigma^{2} \log(2/\beta)}{n}.
$$

Under $H_{0}$, $T_{n}^{V}$ is centered at zero, so the level-$\alpha$ threshold is

$$
c_{\alpha} = 2 \sigma^{2} \sqrt{\tfrac{r \log(2/\alpha)}{n}} + \tfrac{2 \sigma^{2} \log(2/\alpha)}{n}.
$$

For the level-$\alpha$ test to reject under $H_{1}$ with probability $\geq 1 - \beta$, we need $\mathbb{E}[T_{n}^{V}] \geq c_{\alpha} + 2 \sigma^{2} \sqrt{r \log(2/\beta)/n} + 2 \sigma^{2} \log(2/\beta)/n$. Substituting $\mathbb{E}[T_{n}^{V}] \geq \Delta^{2}$ and rearranging gives

$$
n \geq C_{1} \cdot \frac{r \sigma^{4}}{\Delta^{4}} \cdot \log(1/\beta)
$$

for an explicit constant $C_{1}$ depending on $\alpha, \beta$. ∎

### 5.6 Step 4: lower bound via two-point Le Cam, and the conjectural matching rate

We prove the two-point Le Cam lower bound (5.2a), which gives the $\sigma^2/\Delta^2$ rate but not the matching $r$-dependence. We then state the conjectural full minimax rate (5.2b) with a detailed pointer to the Ingster-style argument that would establish it. Earlier drafts of this section sketched a "Fano chaining" argument that would give the $\Delta^4$ rate; on closer reading the sketch is incoherent (it derives three different rates in three paragraphs), so we have removed it and replaced it with an honest separation between what we prove and what we conjecture.

**Lemma 5.5 (Le Cam two-point lemma).**
Let $\mathcal{P}_{0}, \mathcal{P}_{1}$ be two distributions on a sample space, and let $\psi$ be any test rejecting $\mathcal{P}_{0}$ with size $\alpha$ and power $1 - \beta$ on $\mathcal{P}_{1}$. Then the total variation distance $\mathrm{TV}(\mathcal{P}_{0}, \mathcal{P}_{1}) \geq 1 - \alpha - \beta$, equivalently the chi-squared divergence $\chi^{2}(\mathcal{P}_{1} \| \mathcal{P}_{0}) \geq (1 - \alpha - \beta)^{2}$ when the latter is small.

(Standard; see Tsybakov [tsybakov2009] Theorem 2.2.)

**Lemma 5.6 (Chi-squared divergence between two normal-bundle perturbations).**
Let $\mathcal{P}_{0}$ be the joint distribution under Assumption GM with $\xi \equiv 0$, and let $\mathcal{P}_{1}$ be the same with $\xi \equiv v \in V$, $\|v\| = \Delta$. Then for $n$ i.i.d. samples from each population (correct and wrong),

$$
\chi^{2}(\mathcal{P}_{1}^{\otimes n} \| \mathcal{P}_{0}^{\otimes n})
\;\leq\; \exp\!\bigl(\tfrac{n \Delta^{2}}{\sigma^{2}}\bigr) - 1. \tag{5.3}
$$

*Proof outline.* Restricted to wrong samples (where the distributions differ), $\mathcal{P}_{0} = \mathcal{N}(m_{c}(a, b), \sigma^{2} I)$ and $\mathcal{P}_{1} = \mathcal{N}(m_{c}(a, b) + v, \sigma^{2} I)$. The chi-squared divergence between two Gaussians with shared covariance is

$$
\chi^{2}(\mathcal{P}_{1} \| \mathcal{P}_{0}) = \exp\!\bigl(\tfrac{\|v\|^{2}}{\sigma^{2}}\bigr) - 1
= \exp\!\bigl(\tfrac{\Delta^{2}}{\sigma^{2}}\bigr) - 1.
$$

Tensorizing across $n$ i.i.d. wrong samples uses the fact that $1 + \chi^2$ is multiplicative under products of independent measures: $(1 + \chi^2(\mathcal{P}_1^{\otimes n} \| \mathcal{P}_0^{\otimes n})) = (1 + \chi^2(\mathcal{P}_1 \| \mathcal{P}_0))^n$. The correct samples contribute a factor of $1$ to this product since their distribution is unchanged across $\mathcal{P}_0$ and $\mathcal{P}_1$. ∎

*Proposed proof of Theorem 5.2(b), part (5.2a).* By Lemma 5.5, any test with size $\alpha$ and uniform power $1 - \beta$ over $\mathcal{P}_{\Delta}$ requires $\chi^{2}(\mathcal{P}_{1}^{\otimes n} \| \mathcal{P}_{0}^{\otimes n}) \geq (1 - \alpha - \beta)^{2}$. Combine with (5.3):

$$
\exp(n \Delta^{2} / \sigma^{2}) - 1 \geq (1 - \alpha - \beta)^{2},
$$

so $n \Delta^{2} / \sigma^{2} \geq \log\bigl(1 + (1 - \alpha - \beta)^{2}\bigr)$. This gives $n \geq C_2^{\mathrm{LC}} \sigma^{2} / \Delta^{2} \cdot \log\!\bigl(\tfrac{1}{\beta(1-\alpha)}\bigr)$ after rearrangement. ∎

**Toward the conjectural matching rate (5.2b).** The two-point Le Cam bound (5.2a) does not depend on the dimension $r$ of $V$, because it considers a *single* alternative direction. The achievable upper bound (5.1), $n \geq C_1 r \sigma^4 / \Delta^4$, is harder than (5.2a) by a factor of $r \sigma^2 / \Delta^2$. Closing this gap is the content of the chi-squared mixture argument of Ingster [ingster1993, ingster2003]: one constructs a uniform mixture over an approximately-$\Delta$-norm packing of $\mathbb{S}^{r-1}$ inside $V$ and bounds the chi-squared between the mixture and $\mathcal{P}_0$ by a quadratic form. Baraud [baraud2002] and Collier–Comminges–Tsybakov [collier2017] give finite-sample versions with explicit constants. The standard outcome is that for the Gaussian sequence model, the minimax separation rate is $\Delta^2 \asymp \sigma^2 \sqrt{r/n}$, i.e. $n \asymp r \sigma^4 / \Delta^4$. We expect the same rate to apply here after Lemma 5.3 reduces our problem to that model on the projected residuals, but we have not pushed the argument through. *We do not claim (5.2b) as proven; we record it as a target.*

### 5.7 Step 5: exact null distribution

*Proposed proof of Theorem 5.2(c).* Under $H_{0}$ and Lemma 5.3, $P_V r(h_{c})$ and $P_V r(h_{w})$ are i.i.d. $\mathcal{N}(0, \sigma^{2} P_V)$ in $V$. Hence

$$
\frac{1}{n_{p}} \sum_{i=1}^{n_{p}} \|P_V r(h_{i, p})\|^{2}
\sim \frac{\sigma^{2}}{n_{p}} \chi^{2}_{n_{p} r}.
$$

The difference $T_{n}^{V}$ is the difference of two scaled chi-squared random variables. By the standard $F$-distribution identity, the ratio $\bar{Y}_{w} / \bar{Y}_{c}$ follows an $F$-distribution with $(n_w r, n_c r)$ degrees of freedom under $H_{0}$.

For the difference, the central-limit theorem gives the asymptotic form: $\sqrt{n} (T_{n}^{V} / \sigma^{2}) / \sqrt{2 r}$ converges to $\mathcal{N}(0, 1)$. The non-asymptotic exact distribution is a difference of two independent scaled chi-squared random variables, which has known closed form (the generalized chi-squared distribution; see Davies [davies1980]).

For computational purposes we use the asymptotic normal approximation together with the Berry–Esseen rate $O(\sqrt{r/n})$ from Tyurin 2010 [tyurin2010], giving closed-form $p$-values that match permutation calibration to leading order. ∎

**Remark 5.7 (Adaptive $V$ via Romano–Wolf).**
If $V$ is selected from a pre-specified family $\mathcal{V} = \{V_{1}, \dots, V_{K}\}$ rather than being fixed, control of joint Type I error follows from Romano–Wolf [romano2005] step-down with studentized maxT statistic $\max_{k} T_{n}^{V_{k}} / \widehat{\sigma}^{2}$. This preserves the Theorem 5.2(a) rate up to a factor of $\log K$.

---

## 6. Proposed Theorem 3: manifold recovery from data

In practice we will not know $M$; we will estimate it from the correct population. The proposed Theorem 3 below bounds the error in this estimation. We propose to prove the bound for the parametric estimator, where the analysis is cleanest, and to handle the other four estimators (diffusion maps, kernel PCA, local PCA, PCA-on-bins) empirically only. The misspecification term in Section 6.4 below is what handles the case where the parametric class does not exactly contain $M$, which we expect on Llama 3.1 8B per Kantamneni–Tegmark Figure 23.

### 6.1 The parametric estimator

**Definition 6.1 (Parametric estimator $\widehat{M}_{\mathrm{param}}$).**
Let $b_{1}, \dots, b_{K}: \mathbb{Z} \to \mathbb{R}$ be a fixed system of basis functions of the integer label $a$ (for instance, the helix basis with the $\sin(\pi a)$ column dropped per Remark 2.2, $K = 8$). Let $B \in \mathbb{R}^{n_c \times K}$ be the design matrix with $B_{i, j} = b_{j}(a_{i})$, and let $H_c \in \mathbb{R}^{n_c \times d}$ be the matrix of correct activations. Define

$$
\widehat{C} := (B^{\top} B)^{-1} B^{\top} H_c \in \mathbb{R}^{K \times d},
\quad
\widehat{M}_{\mathrm{param}} := \mathrm{col\text{-}span}(\widehat{C}^{\top}) \subset \mathbb{R}^{d}.
$$

The estimator $\widehat{M}_{\mathrm{param}}$ is the column space of the $d \times K$ matrix $\widehat{C}^{\top}$ whose columns are the per-basis-function direction vectors. We orthonormalize via QR decomposition for use in projections.

### 6.2 Theorem 3 (proposed): finite-sample concentration

**Theorem 6.2 (Proposed: manifold recovery).**
Let Assumption GM hold with sub-Gaussian noise $\|\varepsilon\|_{\psi_{2}} \leq \sigma$. Let $\lambda_{\min}^{B} := \lambda_{\min}(\mathbb{E}[b b^{\top}])$ and $\lambda_{\max}^{B} := \lambda_{\max}(\mathbb{E}[b b^{\top}])$ where $b = (b_{1}(a), \dots, b_{K}(a))^{\top}$ and the expectation is over the empirical distribution of integer labels in the data. Let $\kappa^B := \lambda_{\max}^B / \lambda_{\min}^B$ denote the design condition number. Let $\sigma_{K}(C^*)$ denote the smallest singular value of the population coefficient matrix $C^* \in \mathbb{R}^{K \times d}$. Suppose the parametric class is well-specified, i.e. $\mathbb{E}[H_c \mid B] = B (C^*)^{\top}$ exactly. We propose to prove that with probability at least $1 - \delta$,

$$
\sin \theta_{\max}\bigl(\widehat{M}_{\mathrm{param}}, M\bigr)
\;\leq\;
C_{3} \cdot \frac{\sigma\, \sqrt{\kappa^B}}{\sqrt{\lambda_{\min}^{B}} \cdot \sigma_{K}(C^*)} \cdot \sqrt{\frac{d \log(d / \delta)}{n_c}}, \tag{6.1}
$$

for an absolute constant $C_{3}$. When the helix design is well-conditioned ($\kappa^B = O(1)$, which holds for the orthonormalized helix basis after centering), this simplifies to

$$
\sin \theta_{\max} \;\leq\; C_3' \cdot \frac{\sigma}{\sqrt{\lambda_{\min}^B} \cdot \sigma_K(C^*)} \cdot \sqrt{\tfrac{d \log(d/\delta)}{n_c}}.
$$

The denominator carries $\sqrt{\lambda_{\min}^B}$ rather than $\lambda_{\min}^B$: in the OLS Frobenius bound below, $(B^\top B/n_c)^{-1}$ contributes $1/\lambda_{\min}^B$, but $\|B^\top E\|_{\mathrm{op}} \lesssim \sigma \sqrt{n_c \lambda_{\max}^B (d+K)}$ contributes a $\sqrt{\lambda_{\max}^B}$, and the two combine to $\sqrt{\lambda_{\max}^B}/\lambda_{\min}^B = \sqrt{\kappa^B}/\sqrt{\lambda_{\min}^B}$.

Our proposed proof has two steps: (1) operator-norm bound on $\widehat{C}$ via sub-Gaussian matrix concentration, (2) Wedin's perturbation theorem to convert operator-norm to sin-theta. The second step uses the operator-norm form of Wedin rather than the Frobenius Davis–Kahan: we believe this avoids an extra $\sqrt{K}$ factor.

### 6.3 Step 1: Frobenius bound on $\widehat{C}$

**Lemma 6.3 (OLS estimator concentration).**
Under the hypotheses of Theorem 6.2, with probability at least $1 - \delta$,

$$
\|\widehat{C} - C^*\|_{\mathrm{op}}
\;\leq\;
C \cdot \sigma \cdot \frac{\sqrt{\lambda_{\max}^B}}{\lambda_{\min}^B} \cdot \sqrt{\frac{d + K \log(d / \delta)}{n_c}}
\;=\;
C \cdot \sigma \cdot \frac{\sqrt{\kappa^B}}{\sqrt{\lambda_{\min}^B}} \cdot \sqrt{\frac{d + K \log(d / \delta)}{n_c}}. \tag{6.2}
$$

Converting to Frobenius via $\|\cdot\|_{F} \leq \sqrt{K} \|\cdot\|_{\mathrm{op}}$:

$$
\|\widehat{C} - C^*\|_{F}
\;\leq\;
C \cdot \sigma \cdot \frac{\sqrt{\kappa^B}}{\sqrt{\lambda_{\min}^B}} \cdot \sqrt{\frac{K(d + K \log(d / \delta))}{n_c}}. \tag{6.3}
$$

*Proof outline.* Write $H_c = B (C^*)^{\top} + E$ with $E \in \mathbb{R}^{n_c \times d}$ a noise matrix with i.i.d. sub-Gaussian rows of parameter $\sigma$. The OLS residual is

$$
\widehat{C} - C^* = (B^{\top} B)^{-1} B^{\top} E.
$$

Apply the matrix concentration bound for sub-Gaussian matrices (Vershynin [vershynin2018] Section 4.7): $\|B^{\top} E\|_{\mathrm{op}} \leq C \sigma \sqrt{n_c \lambda_{\max}^B (d + K \log(d/\delta))}$ with the stated probability. The inverse satisfies $\|(B^{\top} B)^{-1}\|_{\mathrm{op}} \leq 1/(n_c \lambda_{\min}^B (1 - o(1)))$ by matrix Bernstein (Tropp [tropp2015]), valid for $n_c \gg K \log K$. Combining: $\|\widehat C - C^*\|_{\mathrm{op}} \leq \sigma \sqrt{\lambda_{\max}^B (d + K\log(d/\delta))/n_c} / \lambda_{\min}^B = \sigma \sqrt{\kappa^B (d + K\log(d/\delta))/n_c} / \sqrt{\lambda_{\min}^B}$. ∎

### 6.4 Step 2: from Frobenius to sin-theta

**Lemma 6.4 (Wedin's perturbation theorem, operator-norm form).**
Let $A, A + E \in \mathbb{R}^{d \times K}$ have full column rank, with smallest singular value $\sigma_{K}(A) > 0$. Let $V_{A}, V_{A+E}$ denote their column spaces. Then

$$
\sin \theta_{\max}(V_{A}, V_{A+E})
\;\leq\;
\frac{\|E\|_{\mathrm{op}}}{\sigma_{K}(A) - \|E\|_{\mathrm{op}}}
\;\leq\;
\frac{2 \|E\|_{\mathrm{op}}}{\sigma_{K}(A)}, \tag{6.4}
$$

provided $\|E\|_{\mathrm{op}} \leq \sigma_{K}(A) / 2$.

This is Wedin [wedin1972]; see Stewart–Sun [stewart1990] Theorem 3.6 for the modern statement.

*Proposed proof of Theorem 6.2.* Apply Lemma 6.4 with $A = (C^*)^{\top}$ and $E = (\widehat{C} - C^*)^{\top}$:

$$
\sin \theta_{\max}\bigl(\mathrm{col\text{-}span}((C^*)^{\top}), \mathrm{col\text{-}span}(\widehat{C}^{\top})\bigr)
\;\leq\;
\frac{2 \|\widehat{C} - C^*\|_{\mathrm{op}}}{\sigma_{K}((C^*)^{\top})}
=
\frac{2 \|\widehat{C} - C^*\|_{\mathrm{op}}}{\sigma_{K}(C^*)}.
$$

Substitute (6.2) and absorb constants into $C_{3}$. ∎

### 6.5 Misspecification accounting

**Theorem 6.5 (Proposed: misspecification term).**
Suppose the true manifold $M$ is approximated by the parametric class but not exactly contained in it: $\inf_{C} \mathbb{E}[\|m_{c}(a, b) - B(a) C^{\top}\|_{2}^{2}] =: b^{2}(M) > 0$. Then with probability at least $1 - \delta$,

$$
\sin \theta_{\max}(\widehat{M}_{\mathrm{param}}, M)
\;\leq\;
\underbrace{\frac{b(M)}{\sigma_{K}(C^*)}}_{\text{bias}}
\;+\;
\underbrace{C_{3} \cdot \frac{\sigma \sqrt{\kappa^B}}{\sqrt{\lambda_{\min}^{B}}\, \sigma_{K}(C^*)} \sqrt{\frac{d \log(d / \delta)}{n_c}}}_{\text{variance}}. \tag{6.5}
$$

*Proof outline.* Define $C^*$ as the population OLS minimizer rather than the true generative parameter. Then $\mathbb{E}[H_c] = B (C^*)^{\top} + R$ where $\|R\|_{F} \leq b(M) \sqrt{n_c}$ is the misspecification residual. The Frobenius bound (6.3) now picks up an additional $b(M) \sqrt{n_c} / n_c = b(M) / \sqrt{n_c}$ term, but this is dominated by the deterministic bias $b(M)$ in the operator-norm bound. Apply Wedin as before; the bias and variance terms combine additively. ∎

### 6.6 Composition with Theorem 1

**Theorem 6.6 (Proposed: composed finite-sample bound).**
Substituting $\widehat{M}_{\mathrm{param}}$ for $M$ in $T_{n}$ produces a test statistic $T_{n}^{\widehat{M}}$ satisfying, with probability at least $1 - 2 \delta$ under the hypotheses of Theorems 4.2 and 6.2,

$$
\bigl|T_{n}^{\widehat{M}} - (\|\mu_{\xi}\|^{2} + \mathrm{tr}(\Sigma_{\xi}))\bigr|
\;\leq\;
\underbrace{2 \sigma^{2} \sqrt{\tfrac{2 k \log(2/\delta)}{n}} + \tfrac{2 \sigma^{2} \log(2/\delta)}{n}}_{\text{Theorem 1 noise term}}
\;+\;
\underbrace{C_{4} \cdot D_{\max}^{2} \cdot \sin \theta_{\max}(\widehat{M}, M)}_{\text{Theorem 3 manifold-error term}} \tag{6.6}
$$

where $D_{\max}$ is the boundedness constant from Assumption BD.

*Proof outline.* The residual perturbation under projection-error $\theta := \theta_{\max}(\widehat{M}, M)$ is bounded by Wedin in operator norm: $\|\Pi_{M} - \Pi_{\widehat{M}}\|_{\mathrm{op}} = \sin \theta$ (Stewart–Sun [stewart1990] Theorem 3.6). Hence for any $h$ with $\|h\| \leq D_{\max}$,

$$
\|r_{\widehat{M}}(h)^{2} - r_{M}(h)^{2}\|
= \bigl|\|r_{\widehat{M}}(h)\|^{2} - \|r_{M}(h)\|^{2}\bigr|
\leq 2 \|r_{M}(h)\| \cdot \|r_{\widehat{M}}(h) - r_{M}(h)\|
+ \|r_{\widehat{M}}(h) - r_{M}(h)\|^{2}
\leq C_{4} D_{\max}^{2} \sin \theta.
$$

Averaging across populations and combining with Theorem 4.2's noise concentration via union bound at $\delta/2$ each gives the stated result. ∎

The composed bound vanishes at rate $1/\sqrt{n}$ in dominant terms; the manifold-error term is the bottleneck when $\sigma \sqrt{d \log d}/\sqrt{n_c} > \sigma^{2} \sqrt{k}/\sqrt{n}$, which holds typically.

**Remark 6.7 (Tightening $D_{\max}^2$ via centering).**
The $D_{\max}^2$ factor in (6.6) is potentially loose: it bounds the squared norm of the activation, including the center-of-mass component which is annihilated by the projection difference $\Pi_M - \Pi_{\widehat M}$ when both pass through the same affine offset. After mean-centering activations (Remark 2.4), the relevant quantity is the *centered* width $D_{\max}^{\mathrm{ctr}} := \max_i \|h_c^{(i)} - \bar h_c\|$, which is typically much smaller than the uncentered $\|h_c^{(i)}\|$ at deeper layers (where activations have large mean component due to the residual stream's accumulation). Empirically we expect $D_{\max}^{\mathrm{ctr}} / D_{\max}$ in the range $0.1$–$0.3$. We will report both the uncentered Theorem 6.6 bound and the centered version with $D_{\max}^{\mathrm{ctr}}$ replacing $D_{\max}$.

A second sharpening: when $\widehat M, M$ share their first $m_0$ basis directions (typical for the helix span where the linear and large-period $T \in \{50, 100\}$ components are recovered with high accuracy and the bulk of the error is in high-frequency $T \in \{2, 5\}$ components), the relevant operator-norm difference is $\|\Pi_M - \Pi_{\widehat M}\|_{\mathrm{op}}$ restricted to the unstable subspace, often substantially smaller than the global $\sin\theta_{\max}$.

---

## 7. Proposed cross-fitting protocol and influence function

In practice, fitting $\widehat{M}$ on the same correct samples whose residuals we measure will introduce an upward bias in $T_{n}$ (the correct-residual baseline is artificially small). We propose K-fold cross-fitting to remove this bias. The cross-fitted estimator is Neyman-orthogonal in the sense of Chernozhukov et al. 2018, which should give $\sqrt{n}$-consistency without rate-loss.

### 7.1 The cross-fitting estimator

**Definition 7.1 ($K$-fold cross-fitted statistic).**
Partition the index set $\{1, \dots, n_c\}$ into $K$ folds $\mathcal{F}_{1}, \dots, \mathcal{F}_{K}$. For each fold $k$, fit $\widehat{M}^{(-k)}$ on the correct samples *not* in $\mathcal{F}_{k}$. Define the cross-fitted statistic

$$
T_{n}^{\mathrm{cross}} :=
\frac{1}{n_w} \sum_{j=1}^{n_w} \frac{1}{K} \sum_{k=1}^{K} \|r_{\widehat{M}^{(-k)}}(h_{w}^{(j)})\|^{2}
\;-\;
\frac{1}{K} \sum_{k=1}^{K} \frac{1}{|\mathcal{F}_{k}|} \sum_{i \in \mathcal{F}_{k}} \|r_{\widehat{M}^{(-k)}}(h_{c}^{(i)})\|^{2}.
$$

### 7.2 Asymptotic normality and influence function

**Theorem 7.2 (Proposed: asymptotic normality of $T_{n}^{\mathrm{cross}}$).**
Under Assumptions GM, BD, REG, the cross-fitting protocol of Definition 7.1, and $K \geq 2$ fixed, the cross-fitted statistic is unbiased for $\mathbb{E}[T_{n}]$ to first order:

$$
\mathbb{E}[T_{n}^{\mathrm{cross}}] = \|\mu_{\xi}\|^{2} + \mathrm{tr}(\Sigma_{\xi}) + O_{\mathbb{P}}(\sin^{2} \theta_{\max}(\widehat{M}, M)).
$$

Moreover, as $n \to \infty$,

$$
\sqrt{n} \bigl(T_{n}^{\mathrm{cross}} - \mathbb{E}[T_{n}]\bigr) \xrightarrow{d} \mathcal{N}(0, V_{\infty}),
$$

where $V_{\infty}$ is the asymptotic variance computable in closed form (Section 7.3 below).

### 7.3 Influence function calculation

By the von Mises calculus / influence-function approach (Chernozhukov et al. [chernozhukov2018]), the cross-fitted statistic admits the expansion

$$
T_{n}^{\mathrm{cross}} - \mathbb{E}[T_{n}]
= \frac{1}{n_{w}} \sum_{j} \phi_{w}(h_{w}^{(j)}) - \frac{1}{n_{c}} \sum_{i} \phi_{c}(h_{c}^{(i)}) + o_{\mathbb{P}}(1/\sqrt{n}),
$$

where the influence functions are

$$
\begin{aligned}
\phi_{w}(h) &= \|r_{M}(h)\|^{2} - \mathbb{E}[\|r_{M}(h_{w})\|^{2}], \\
\phi_{c}(h) &= \|r_{M}(h)\|^{2} - \mathbb{E}[\|r_{M}(h_{c})\|^{2}].
\end{aligned}
$$

**Neyman-orthogonality verification.** Following Chernozhukov et al. [chernozhukov2018] §1, our parameter of interest is $\theta_0 := \mathbb{E}[T_n] = \mathbb{E}[\|r_M(h_w)\|^2] - \mathbb{E}[\|r_M(h_c)\|^2]$, with nuisance parameter $\Pi_M$ (or, parameterizing in the linear-span case, $U_S$ — the orthonormal basis of $S$). Define the moment function

$$
\psi(h_c, h_w; \theta, \Pi) := (\|h_w - \Pi h_w\|^2 - \|h_c - \Pi h_c\|^2) - \theta.
$$

Neyman-orthogonality requires the Gateaux derivative of $\mathbb{E}[\psi]$ with respect to $\Pi$ at the truth to vanish:

$$
\partial_t \, \mathbb{E}[\psi(h_c, h_w; \theta_0, \Pi_M + t \Delta_\Pi)]\big|_{t=0} = 0
\quad \text{for all admissible } \Delta_\Pi.
$$

For the linear-span parameterization, write $\Pi = U U^\top$ with $U \in \mathbb{R}^{d \times m}$ orthonormal, and let $\Delta_U$ be a tangent direction satisfying $U^\top \Delta_U + \Delta_U^\top U = 0$ (the Stiefel-manifold tangent condition). Then $\partial_t (U + t\Delta_U)(U + t\Delta_U)^\top|_{t=0} = U \Delta_U^\top + \Delta_U U^\top$, and

$$
\partial_t \mathbb{E}[\|h - (U + t\Delta_U)(U+t\Delta_U)^\top h\|^2]\big|_{t=0}
= -2 \mathbb{E}[h^\top (U \Delta_U^\top + \Delta_U U^\top)(I - UU^\top) h].
$$

Under Assumption GM, $\mathbb{E}[h_p] \in M_S = \mathrm{col}(U)$ for both $p \in \{c, w\}$ in the *centered* model where $\mathbb{E}[\xi] = 0$ (the $\mu_\xi \neq 0$ case is the alternative we are testing for, not a nuisance). Hence $(I - UU^\top)\mathbb{E}[h_p] = 0$ to leading order under both populations, and the cross-term vanishes from population means. The remaining variance contribution is the same across populations under GM(iii) and cancels in the difference $\mathbb{E}[\psi]$. Therefore the Gateaux derivative vanishes, establishing Neyman-orthogonality. In the alternative $\mu_\xi \neq 0$ case the orthogonality holds at first order in the manifold-estimation error $\|\widehat{U} - U\|_{\mathrm{op}}$, with a residual bias of order $\|\widehat{U} - U\|_{\mathrm{op}} \cdot \|\mu_\xi\| = o_\mathbb{P}(\|\mu_\xi\|^2)$ when $\|\widehat{U} - U\|_{\mathrm{op}} = o_\mathbb{P}(\|\mu_\xi\|)$ — which holds whenever Theorem 6.2 gives $\sin\theta_{\max} = o_\mathbb{P}(\Delta)$.

This Neyman-orthogonality is what makes the cross-fitted estimator $\sqrt{n}$-consistent without requiring $\sqrt{n}$-consistency of the manifold estimator: the leading bias term cancels at first order. The variance is

$$
V_{\infty} = \frac{\mathrm{Var}(\phi_{w}(h_{w}))}{n_{w}/n} + \frac{\mathrm{Var}(\phi_{c}(h_{c}))}{n_{c}/n}.
$$

By Lemma 5.4 restricted to the full residual space, $\mathrm{Var}(\phi_{c}(h_{c})) = 2 k \sigma^{4} + \text{(higher)}$ and $\mathrm{Var}(\phi_{w}(h_{w})) = 2 k \sigma^{4} + 4 \|\mu_{\xi}\|^{2} \sigma^{2} + O$.

**Corollary 7.3 (Proposed: closed-form asymptotic CI).**
For $n_c = n_w = n/2$,

$$
T_{n}^{\mathrm{cross}} \pm z_{\alpha/2} \sqrt{V_{\infty} / n}
\quad\text{with}\quad
V_{\infty} = 4 k \sigma^{4} + 8 \|\mu_{\xi}\|^{2} \sigma^{2}
$$

gives a valid $(1 - \alpha)$-asymptotic confidence interval. We estimate $V_{\infty}$ from the cross-fitted residuals.

---

## 8. Proposed: matched permutation with exact conditional validity

Wrong samples are systematically harder than correct samples (more carries, larger sums). A naive permutation null would reject for *any* between-population difference, including pure input-difficulty differences that have nothing to do with manifold geometry. We propose stratified within-bin permutation as a fix, building on the established literature on stratified and regression-adjusted permutation tests (Lehmann–Romano [lehmann2005] §15, Freedman–Lane [freedmanlane1983], Anderson–Robinson [andersonrobinson2001], Hemerik–Goeman [hemerik2018]). The construction below gives exact conditional Type I control via the Lehmann–Romano permutation-group argument.

### 8.1 The bin partition and exchangeability

**Data convention for §8.** The matched permutation test runs on the *pooled* data $\mathcal{D}_n = \{(a_i, b_i, h_i, y_i)\}_{i=1}^n$ where $i$ indexes the union of the correct and wrong populations and $y_i \in \{0, 1\}$ is the correctness label. We do *not* permute within $H_c$ alone (which would give single-label bins where permutation is meaningless); we permute the *labels* $y_i$ within bins of pooled $(h_i, y_i)$ pairs. This applies throughout §8 unless otherwise stated.

**Definition 8.1 (Coarse bin function).**
Define $\phi: \{0, \dots, 99\}^{2} \to \mathcal{B}$, where $\mathcal{B}$ is a finite set of bins, by

$$
\phi(a, b) := \bigl(\mathrm{sumbin}(a + b),\, \mathrm{carry}(a, b),\, \mathrm{decile}(a),\, \mathrm{decile}(b),\, \mathrm{tokclass}(a, b)\bigr).
$$

Each component takes finitely many values; in our pre-registration the total number of bins is $|\mathcal{B}| = 4 \times 4 \times 10 \times 10 \times 2 = 3200$, of which approximately $200$–$300$ are non-empty on the 10{,}000-pair grid.

### 8.2 Conditional null and exchangeability

**Definition 8.2 (Conditional null — explicit exchangeability conditions).**
Let $\mathcal{D}_n = \{(a_i, b_i, h_i, y_i)\}_{i=1}^n$ denote the data, where $y_i = y(a_i, b_i)$ is the correctness label and $h_i = h(a_i, b_i)$ is the residual-stream activation. The conditional null hypothesis is

$$
H_{0}^{\mathrm{cond}}: \quad \text{for every bin } B \in \mathcal{B} \text{ and every permutation } \pi \in \mathfrak{S}(\phi^{-1}(B)),
$$

$$
\bigl\{(h_{i}, y_{i})\bigr\}_{i \in \phi^{-1}(B)}
\overset{d}{=}
\bigl\{(h_{\pi(i)}, y_{\pi(i)})\bigr\}_{i \in \phi^{-1}(B)},
$$

where $\overset{d}{=}$ denotes equality in joint distribution conditional on the bin assignments $\{\phi(a_i, b_i)\}_{i=1}^n$. Concretely, $H_0^{\mathrm{cond}}$ says: *within each bin, knowing the correctness label tells you nothing additional about the activation* — equivalently, $h \perp y \mid \phi(a, b)$.

This is the appropriate null for our science question: we want to detect activation-level differences between correct and wrong populations *that are not explained by input difficulty* (sum size, carries, operand magnitude, tokenization). A naive between-population test rejects $h \perp y$ marginally, conflating input-difficulty differences with representational differences. The within-bin null $h \perp y \mid \phi$ controls for these by stratification.

**Connection to Freedman–Lane / Anderson–Robinson.** Our setup is the "permutation under reduced model" scheme of Freedman–Lane [freedmanlane1983], specialized to the case where the nuisance covariate $\phi$ is discrete and the regression on $\phi$ is the bin-mean. Anderson–Robinson [andersonrobinson2001] Table 1 catalogs the resulting permutation methods; ours is the one labeled "permutation of raw data within strata," which they show preserves exact validity when within-stratum exchangeability holds (their condition R5). Hemerik–Goeman [hemerik2018] §4 give modern conditions for exactness of permutation tests under sub-group invariance, of which ours is a special case.

**Theorem 8.3 (Proposed: exact conditional validity).**
Let $T_{n}^{\mathrm{matched}}$ be the bin-weighted average statistic defined in the main paper Section 3.3, and let $c_{\alpha}$ be its $(1 - \alpha)$-quantile under the bin-restricted permutation distribution (the permutations are drawn uniformly from $\prod_{B} \mathfrak{S}(\phi^{-1}(B))$). Then under $H_{0}^{\mathrm{cond}}$,

$$
\mathbb{P}[T_{n}^{\mathrm{matched}} > c_{\alpha} \mid \{\phi(a_{i}, b_{i})\}_{i}] \leq \alpha.
$$

*Proof outline.* Conditional on bin assignments, $H_0^{\mathrm{cond}}$ implies that the within-bin labels $\{y_{i}\}_{i \in \phi^{-1}(B)}$ are exchangeable with respect to the activations $\{h_i\}_{i \in \phi^{-1}(B)}$ for each $B$. The product group $G := \prod_{B \in \mathcal{B}} \mathfrak{S}(\phi^{-1}(B))$ acts on the data by permuting indices within each bin. The conditional null is invariant under this action: for any $g \in G$, the joint distribution of the bin-permuted data equals the joint distribution of the original data. Lehmann–Romano [lehmann2005] Theorem 15.2.1 (the randomization hypothesis for general invariance groups) then gives $\mathbb{P}[T_n^{\mathrm{matched}} > c_\alpha] \leq \alpha$. ∎

**Remark 8.4 (Empty / sparse bins).** Bins with $\leq 1$ sample contribute nothing to the permutation distribution (any permutation is trivial). Bins with all-correct or all-wrong labels also contribute trivially. We aggregate over non-trivial bins only when computing $T_n^{\mathrm{matched}}$, weighting each bin by $|\phi^{-1}(B)| \cdot p_B (1 - p_B)$ where $p_B = $ fraction-correct in bin $B$, following Anderson–Robinson's [andersonrobinson2001] efficient-stratified-test construction.

**Remark 8.5 (Diagnosing exchangeability violations).** Exchangeability could fail if (a) the bin function $\phi$ is too coarse and residual difficulty variation remains within bins, or (b) there is autocorrelation across $(a, b)$ pairs from the prompt template. We diagnose (a) by post-hoc regression of $\|r(h)\|^2$ on continuous within-bin difficulty features (digit-product, operand sum modulo small primes); a significant within-bin slope indicates the bin function is too coarse. We diagnose (b) by shuffling the prompt order and checking that the test statistic distribution is unchanged. If diagnostics flag either issue, fall back to the Freedman–Lane regression-adjusted residual permutation: regress $\|r(h)\|^2$ on continuous difficulty features within each bin first, then permute the regression residuals.

---

## 9. Proposed causal pipeline and Proposition 4

The proposed Theorems 1, 2, 3 establish a *correlational* claim: the wrong-population residual differs from the correct-population residual in a localized subspace. We propose four interventions to upgrade this to a *mechanistic* claim, in the sense that the localized subspace is functionally responsible for the failure rather than just associated with it. Proposition 4 below ties the magnitude of the causal effect to the size of the localized perturbation quantitatively, which is the central new content of this section.

### 9.1 The patch operation

**Definition 9.1 (Patch).**
Given an activation $h \in \mathbb{R}^{d}$, an orthonormal basis $U_{V} \in \mathbb{R}^{d \times r}$, a strength $\alpha \in [0, 1]$, and a target $\delta \in \mathbb{R}^{r}$,

$$
\mathrm{Patch}(h, V, \alpha, \delta) := h + \alpha (U_{V} \delta - P_V h).
$$

At $\alpha = 1$ and $\delta = 0$, this is $h - P_V h = (I - P_V) h$, the removal patch that zeros out the $V$-component. At $\alpha = 1$ and $\delta \neq 0$, this is $(I - P_V) h + U_{V} \delta$, the injection patch that replaces the $V$-component by $\delta$.

### 9.2 The four interventions and their ACEs

**Definition 9.2 (Interventions and average causal effects).**
Let $\widehat{M} \subset \mathbb{R}^{d}$ denote the estimated manifold, $V \subset \widehat{M}^{\perp}$ the localized failure subspace, $V^{\mathrm{rand}}$ a random subspace of the same dimension as $V$, and $\widehat{\mu}_{\xi}$ the cross-fitted estimate of $\mu_{\xi}$ from wrong-population residuals projected onto $V$.

**Logit-difference as a downstream forward-pass functional.** For $h \in \mathbb{R}^d$ representing a candidate residual-stream activation at the analysis layer $\ell^*$ on a problem $(a, b)$ with answer token $\tau(s)$, define $\mathrm{forward}_{\ell^* \to L}(h; a, b)$ as the model's downstream forward pass: the function that takes $h$ at layer $\ell^*$ in place of the model's natural activation $h(a, b)$ and propagates through layers $\ell^* + 1, \dots, L$ and the unembedding map, producing logits at the answer position. Then

$$
\mathrm{LD}(h; a, b)
\;:=\;
\mathrm{logit}^{\mathrm{final}}_{\tau(s)}(\mathrm{forward}_{\ell^* \to L}(h; a, b))
\;-\;
\max_{t \neq \tau(s)} \mathrm{logit}^{\mathrm{final}}_{t}(\mathrm{forward}_{\ell^* \to L}(h; a, b)).
$$

This is the *downstream* logit-difference: it is a functional of the residual stream at $\ell^*$ holding the rest of the network fixed, in the sense of activation patching (Wang–Variengien [wang2022]; Conmy et al. [conmy2023]). Critically, $\mathrm{LD}(h; a, b)$ is well-defined for any $h \in \mathbb{R}^d$, not only for the natural activation $h(a, b)$. We write $\mathrm{LD}(h)$ when the dependence on $(a, b)$ is clear from context. Then:

$$
\begin{aligned}
\mathrm{ACE}_{N} &:= \mathbb{E}_{h_{w}}\!\bigl[ \mathrm{LD}(\Pi_{\widehat{M}}(h_{w})) - \mathrm{LD}(h_{w}) \bigr], \\
\mathrm{ACE}_{NV} &:= \mathbb{E}_{h_{w}}\!\bigl[ \mathrm{LD}(\mathrm{Patch}(h_{w}, V, 1, 0)) - \mathrm{LD}(h_{w}) \bigr], \\
\mathrm{ACE}_{S} &:= \mathbb{E}_{h_{c}}\!\bigl[ \mathrm{LD}(\mathrm{Patch}(h_{c}, V, 1, \widehat{\mu}_{\xi})) - \mathrm{LD}(h_{c}) \bigr], \\
\mathrm{ACE}_{R} &:= \mathbb{E}_{h_{c}}\!\bigl[ \mathrm{LD}(\mathrm{Patch}(h_{c}, V^{\mathrm{rand}}, 1, \delta^{\mathrm{rand}})) - \mathrm{LD}(h_{c}) \bigr].
\end{aligned}
$$

### 9.3 Proposition 4 with Hessian remainder

**Proposition 9.3 (Proposed: causal sufficiency, second-order).**
Assume the model's logit-difference $\mathrm{LD}: \mathbb{R}^{d} \to \mathbb{R}$ (in the downstream-forward-pass sense of Definition 9.2) is twice continuously differentiable, with operator-norm-bounded Hessian: $\sup_{h \in B(h_{c}, \rho)} \|\nabla^{2} \mathrm{LD}(h)\|_{\mathrm{op}} \leq L$ for $\rho > 2 (\|\widehat{\mu}_{\xi}\| + D_{\max}^{V})$, where $D_{\max}^V := \sup_{i} \|P_V h_c^{(i)}\|$ is the maximum norm of the V-projection of correct activations. The patch operation gives a per-sample displacement

$$
\Delta h_c \;:=\; \mathrm{Patch}(h_c, V, 1, \widehat{\mu}_\xi) - h_c \;=\; U_V \widehat{\mu}_\xi - P_V h_c.
$$

Note that $\Delta h_c \neq U_V \widehat\mu_\xi$ in general: the patch *replaces* the V-component of $h_c$ by $U_V \widehat\mu_\xi$, so the displacement subtracts off the original V-component $P_V h_c$. Then

$$
\bigl|\mathrm{ACE}_{S} - \mathrm{ACE}_{S}^{\mathrm{linear}}\bigr|
\;\leq\;
\frac{L}{2} \mathbb{E}_{h_c}\bigl[\|\Delta h_c\|_2^2\bigr]
\;=\;
\frac{L}{2}\bigl(\|\widehat{\mu}_{\xi}\|^{2} + \mathbb{E}\|P_V h_c\|^2 - 2 \widehat\mu_\xi^\top U_V^\top \mathbb{E}[P_V h_c]\bigr), \tag{9.1}
$$

where the first-order linear prediction is

$$
\mathrm{ACE}_{S}^{\mathrm{linear}} := \mathbb{E}_{h_{c}}\!\bigl[ \langle \nabla \mathrm{LD}(h_{c}), \Delta h_c \rangle \bigr]
= \widehat{\mu}_{\xi}^{\top} U_{V}^{\top} \mathbb{E}_{h_{c}}[\nabla \mathrm{LD}(h_{c})] - \mathbb{E}_{h_c}[h_c^\top P_V \nabla \mathrm{LD}(h_c)].
$$

**Simplification under linear-span GM.** In the linear-span case (M = M_S, V ⊥ M_S), Assumption GM gives $h_c = m_c + \varepsilon_c$ with $m_c \in M_S$, so $P_V m_c = 0$ and $\mathbb{E}[P_V h_c] = \mathbb{E}[P_V \varepsilon_c] = 0$ (zero-mean noise, GM(iii)). The second cross-term in (9.1) vanishes in expectation, leaving

$$
\mathbb{E}\|\Delta h_c\|^2 = \|\widehat{\mu}_\xi\|^2 + \mathbb{E}\|P_V \varepsilon_c\|^2 = \|\widehat\mu_\xi\|^2 + r \sigma^2 \quad \text{(under isotropic noise)}.
$$

The leading-order linear prediction simplifies to $\mathrm{ACE}_S^{\mathrm{linear}} = \widehat\mu_\xi^\top U_V^\top \mathbb{E}[\nabla \mathrm{LD}(h_c)]$ when $\nabla \mathrm{LD}(h_c)$ is uncorrelated with $\varepsilon_c$ (which holds when $\nabla \mathrm{LD}$ is approximately constant on the noise scale, i.e., when $L \cdot \sigma \sqrt{r} \ll \|\nabla \mathrm{LD}\|$).

**Magnitude lower bound (Cauchy–Schwarz, with alignment assumption).** Cauchy–Schwarz gives the *equality*

$$
\bigl|\mathrm{ACE}_{S}^{\mathrm{linear}}\bigr|_{\mathrm{leading}}
\;=\;
\|\widehat{\mu}_{\xi}\| \cdot \|U_{V}^{\top} \mathbb{E}[\nabla \mathrm{LD}(h_{c})]\| \cdot |\cos \theta_{\xi, \nabla}|, \tag{9.2}
$$

where $\theta_{\xi, \nabla}$ is the angle between $\widehat{\mu}_{\xi}$ and $U_{V}^{\top} \mathbb{E}[\nabla \mathrm{LD}(h_{c})]$. This is *not a lower bound* by itself — Cauchy–Schwarz is an equality with $|\cos\theta|$. To turn (9.2) into a useful lower bound on the magnitude, we additionally assume $|\cos \theta_{\xi, \nabla}| \geq c_0 > 0$ for some calibration constant $c_0$:

$$
\text{(Alignment Assumption ALN):} \quad
\bigl|\langle \widehat{\mu}_\xi,\, U_V^\top \mathbb{E}[\nabla \mathrm{LD}(h_c)]\rangle\bigr| \;\geq\; c_0 \cdot \|\widehat\mu_\xi\| \cdot \|U_V^\top \mathbb{E}[\nabla\mathrm{LD}(h_c)]\|. \tag{9.3}
$$

Under (ALN), (9.2) becomes the lower bound $|\mathrm{ACE}_S^{\mathrm{linear}}|_{\mathrm{leading}} \geq c_0 \cdot \|\widehat\mu_\xi\| \cdot \|U_V^\top \mathbb{E}[\nabla \mathrm{LD}(h_c)]\|$. (ALN) is testable empirically: estimate both vectors and compute the cosine. We pre-register reporting $\widehat{\cos\theta_{\xi,\nabla}}$ alongside the test, with the threshold $c_0 = 0.3$ as the alignment criterion below which we treat the magnitude prediction as inconclusive.

*Proof outline.* Apply Taylor's theorem with integral remainder to $\mathrm{LD}(h + \Delta h) - \mathrm{LD}(h)$:

$$
\mathrm{LD}(h + \Delta h) - \mathrm{LD}(h)
= \langle \nabla \mathrm{LD}(h), \Delta h \rangle + \int_{0}^{1} (1 - t) \langle \nabla^{2} \mathrm{LD}(h + t \Delta h) \Delta h, \Delta h \rangle \, dt.
$$

With $\Delta h = U_V \widehat\mu_\xi - P_V h$ as derived above, $\|\Delta h\|^2 = \|\widehat\mu_\xi\|^2 + \|P_V h\|^2 - 2 \widehat\mu_\xi^\top U_V^\top P_V h$ since $U_V^\top P_V = U_V^\top$. The remainder is bounded by $\tfrac{1}{2} L \|\Delta h\|^2$ pointwise. Take expectations over $h_c$ to get (9.1). The magnitude relation (9.2) is Cauchy–Schwarz applied to the leading-order term $\widehat\mu_\xi^\top U_V^\top \mathbb{E}[\nabla\mathrm{LD}(h_c)]$. ∎

**Remark 9.4 (Smooth approximation of $\mathrm{LD}$ at argmax flips).**
$\mathrm{LD}(h) = \mathrm{logit}_{s}(h) - \max_{t \neq s} \mathrm{logit}_{t}(h)$ is non-smooth at points where the argmax over $t \neq s$ changes. Replace the max by log-sum-exp at temperature $\tau$:

$$
\mathrm{LD}_{\tau}(h) := \mathrm{logit}_{s}(h) - \tau \log \sum_{t \neq s} \exp(\mathrm{logit}_{t}(h) / \tau).
$$

$\mathrm{LD}_{\tau}$ is $C^{\infty}$, and $\mathrm{LD}_{\tau} \to \mathrm{LD}$ uniformly as $\tau \to 0^{+}$ on any compact set. The Hessian-bound constant $L$ in Proposition 9.3 for $\mathrm{LD}_{\tau}$ scales as $1/\tau$ near argmax flips. We apply Proposition 9.3 to $\mathrm{LD}_{\tau}$ for some $\tau > 0$ and report sensitivity to $\tau$ in the appendix.

---

## 10. Discussion of method comparison

The five manifold-recovery methods we consider are listed in the main proposal document (Section 2.6). The mathematical basis for selecting the parametric estimator as primary, rather than one of the kernel methods, is:

- Theorem 6.2 (proposed) for the parametric estimator $\widehat{M}_{\mathrm{param}}$, which gives the cleanest finite-sample guarantee of any of the five methods.

- For PCA on class means, we expect the misspecification term $b(M) = O(1/K_{\mathrm{bins}})$ from Theorem 6.5 to dominate at large $n_c$, so the estimator does not converge to $M$ even with infinite data when the bin width fails to resolve the manifold's highest-frequency component.

- For Local PCA we cite Singer–Wu [singer2012] convergence under a small-curvature assumption $\kappa \cdot h \leq c$ for kernel bandwidth $h$. We expect this to fail in the high-curvature regime of the helix, leading to the empirical breakdown observed in the toy companion document.

- For Diffusion Maps we cite Coifman–Lafon [coifman2006] operator-norm convergence to the Laplace–Beltrami operator at rate $O(\sigma_{DM}^{2} + n^{-1/2} \sigma_{DM}^{-d/2})$ plus the regression-lift step. We do not propose to prove a clean composed bound for the embedding-then-lift pipeline; we use Diffusion Maps as cross-method validation rather than as a primary estimator.

- For Kernel PCA the same applies, citing Schölkopf–Smola–Müller [scholkopf1998] consistency.

We propose to recommend the parametric estimator on the combined basis of (a) the cleanest theoretical guarantee, (b) best empirical performance in the toy companion document, and (c) direct match to KT's own protocol so that comparisons are unambiguous.

---

## 11. Open questions for Barnábás

This document is a proposal, not a finished result. Several issues identified in earlier drafts have been addressed in this revision (the AI-reviewer feedback log is summarized at the end of this section). The remaining open questions, in order of how blocking they are for the empirical pipeline:

1. **The WLOG re-centering of $\xi$.** Lemma 3.2 claims that any tangent component of $\xi$ can be absorbed into a different choice of $m_{c}(a, b) \in M$, leaving the remaining $\widetilde{\xi}$ orthogonal to the tangent space. We have written this as a first-order claim; the second-order error comes from the curvature of $M$ at $m_{c}$. We have not written out the exponential-map calculation rigorously and would value confirmation that the rate $O(\kappa \cdot \|P^T \xi\|)$ is correct.

2. **Sharpness of the linearization-error term $R_{1}$.** Theorem 4.2's remainder $R_{1}(\sigma, \kappa_{\max})$ is bounded by $O(\sigma^{2} \kappa_{\max}^{2})$. After the Lemma 4.3 revision (with the Niyogi–Smale–Weinberger–style bound and the $(1 - c_0)$ denominator), we believe the constant in front is now correct up to absolute factors, but we have not propagated this all the way through the squared-norm expansion of $\mathbb{E}[\|r\|^2]$ to track the constant in front of $R_1$.

3. **Exact constants $C_{1}, C_{2}^{\mathrm{LC}}, C_{3}, C_{4}$.** The absolute constants in Theorems 5.2(a), 5.2(b)/(5.2a), 6.2, and 6.6 are stated abstractly. Tracking them through the proofs is mechanical but would yield explicit numerical bounds. We have not done this and would welcome guidance on which constants matter most for the application.

4. **The conjectured minimax matching rate (5.2b).** Section 5.6 now records the rate $n \geq C_2 r \sigma^4 / \Delta^4$ as a *conjecture*, not a theorem, with a pointer to Ingster's chi-squared mixture argument as the canonical template. We would value Barnábás's view on whether this rate (a) actually holds in our setting after Lemma 5.3 reduces to the Gaussian sequence model on projected residuals, and (b) is worth proving rigorously for the BlackBoxNLP submission, or whether downgrading to conjecture is the right call. We currently lean toward the latter, given empirical work is the binding constraint on the timeline.

5. **Curve-case drift bound $\eta_0$.** Theorem 5.2(C) replaces the strict $V \subseteq \bigcap_p N_p M$ assumption by the explicit drift bound $\sup_p \|P_V (P^N_p - P^N_{p_0}) P_V\|_{\mathrm{op}} \leq \eta_0$. We have not derived an a-priori bound on $\eta_0$ in terms of the helix curvature $\kappa_{\max}$; we expect $\eta_0 = O(\kappa_{\max} \cdot \mathrm{diam}(M))$ for short curves and $O(1)$ for long curves where the tangent rotates many times. Confirming this would tighten the curve-case theorem.

6. **Operator-norm Wedin vs Frobenius Davis–Kahan in Theorem 6.2.** We use the operator-norm form of Wedin because we believe it gives a constant that is sharper by factor $\sqrt{K}$. We have not double-checked that the dependence on $K$ in the OLS Frobenius bound (Lemma 6.3) does not re-introduce the $\sqrt{K}$ that operator-norm Wedin saves.

7. **Cross-fitting under the alternative.** Theorem 7.2 gives asymptotic normality at the null, and §7.3 now explicitly verifies Neyman-orthogonality at the null. The non-asymptotic version under the alternative ($\mu_\xi \neq 0$) requires checking that the manifold-estimation residual bias of order $\|\widehat U - U\|_{\mathrm{op}} \cdot \|\mu_\xi\|$ is $o_\mathbb{P}(\|\mu_\xi\|^2)$, which holds when Theorem 6.2 gives $\sin\theta_{\max} = o_\mathbb{P}(\Delta)$. The threshold relating $n_c$, $\sigma$, and $\Delta$ that ensures this has not been worked out explicitly.

8. **Higher-order Taylor for Proposition 9.3.** The remainder bound is at order $L \|\widehat{\mu}_{\xi}\|^{2} / 2$. Higher-order corrections involving $\nabla^{3} \mathrm{LD}$ would tighten the bound when the LD landscape has bounded third derivatives, which holds generically but has not been verified for our specific transformer readouts. The synthetic toy in the companion document found a magnitude-vs-Taylor ratio of about $5\times$ in one calibrated case; understanding whether this is mostly argmax-flip non-smoothness or higher-order Taylor would help.

9. **The argmax-flip / smooth-max question.** Real logit-difference $\mathrm{LD}(h) = \mathrm{logit}_{s}(h) - \max_{t \neq s} \mathrm{logit}_{t}(h)$ is non-smooth at points where the argmax over $t \neq s$ flips. We propose using log-sum-exp at temperature $\tau$ as a smooth surrogate, but the dependence of the Hessian bound $L$ on $\tau$ near argmax flips ($L = O(1/\tau)$) means the bound deteriorates as $\tau \to 0$. We do not have a clean way to handle this and would value advice on whether a different smoothing (e.g., random tie-breaking) gives a uniformly bounded $L$.

10. **What is the right noise model for activations?** Real activations are a deterministic function of the input plus within-prompt structural variation. Modeling this as i.i.d. sub-Gaussian noise is standard but is a substantial idealization. After Theorem 4.10's anisotropic generalization is in place, the remaining question is whether the noise is sub-Weibull (heavier-tailed) and how the rates would change. We will diagnose this empirically by examining tails of the residual distribution.

### 11.1 Issues addressed in this revision (responding to AI-reviewer feedback)

The previous draft of this document was reviewed by an AI reviewer (Stanford ML Group's `paperreview.ai`). Most of the issues raised were valid and have been addressed:

- **Notation consistency in Theorem 4.2.** The reviewer claimed the theorem statement uses $\mu_\varepsilon, \Sigma_\varepsilon$ but the proofs use $\mu_\xi, \Sigma_\xi$. On re-reading, the theorem already used $\xi$-notation correctly; the reviewer hallucinated this bug. We have nevertheless added explicit text to Theorem 4.2(a)(b) explaining why the $\varepsilon$-contributions cancel under GM(iii), to forestall the same misreading by future readers.

- **The $V \subseteq \bigcap_p N_p M$ assumption is too strong for a curved manifold.** Addressed: Theorem 5.2 now splits into case (L) (linear span, where the assumption is automatic) and case (C) (curve, where it is replaced by an explicit drift bound $\eta_0$).

- **The $r\sigma^4/\Delta^4$ minimax lower bound proof is incoherent.** Addressed: Section 5.6 now proves only the two-point Le Cam bound (5.2a) at rate $\sigma^2/\Delta^2$, and explicitly downgrades the matching rate (5.2b) to a conjecture with proper Ingster citations.

- **Lemma 4.3 constants and $P^N\delta$ vs $P^T\delta$ separation.** Addressed: Lemma 4.3 now has an explicit $(1 - c_0)$ denominator, distinguishes tangent vs normal contributions, and cites Aamari–Levrard for the modern statement.

- **Anisotropic noise needs a stand-alone theorem.** Addressed: Remark 4.10 promoted to Theorem 4.10 with full statement, estimator $\widehat k_{\mathrm{eff}}$, and concentration via Hanson–Wright.

- **Cross-fitting Neyman-orthogonality not shown.** Addressed: §7.3 now contains an explicit Gateaux-derivative calculation verifying Neyman-orthogonality at the null and bounding the residual bias under the alternative.

- **Permutation section lacks Freedman–Lane reference and exchangeability conditions.** Addressed: §8 now cites Freedman–Lane, Anderson–Robinson, Hemerik–Goeman; states the exchangeability condition explicitly as $h \perp y \mid \phi(a, b)$; adds Remarks 8.4 and 8.5 on sparse bins and exchangeability diagnostics.

- **$D_{\max}^2$ in composition bound is loose.** Addressed: Remark 6.7 distinguishes uncentered $D_{\max}$ from centered $D_{\max}^{\mathrm{ctr}}$ and notes the typical $0.1$–$0.3$ ratio, plus the unstable-subspace sharpening.

- **Reach and common-normal verification deferred.** Addressed: Remark 3.6 lists explicit empirical estimators $\widehat\tau, \widehat\kappa_{\max}, \widehat\sigma_{\mathrm{eff}}, \widehat\eta$ and pre-registers them as filters before the main test.

- **Missing related-work connections (DMET, diffusion-geometry estimators, local-quadratic models).** Addressed: references added to bibliography. Discussion in §10 still primarily empirical.

---

## 12. Framing and venue fit

This section steps back from the math to record the framing we have settled on for the paper, the contribution we believe is genuinely new, and the reviewer attacks we expect. It is intended as a working note, not part of the eventual paper.

### 12.1 Core claim, restated

The paper's central question is:

> *When language models make arithmetic errors, do their internal representations stay on the learned number manifold, or do they leave it in geometrically localized ways?*

This builds on Kantamneni and Tegmark (2025), who establish that GPT-J, Pythia, and Llama 3.1 represent integers on a generalized helix and manipulate that representation for arithmetic. KT's contribution is the representation and its causal relevance for correct computations. Our paper asks a different, complementary question: *what geometry distinguishes correct from wrong computations?*

We are not claiming to have discovered the helix. We are using KT's representation as a fixed scaffold and asking a failure-analysis question on top of it.

### 12.2 What the paper claims, after the revisions

The paper's empirical claims, in order of importance:

1. Wrong activations have significantly larger off-span residuals than correct activations.
2. This difference persists after stratifying by sum size, carry structure, operand decile, and tokenization class.
3. The effect is localized to specific layers and to a low-dimensional subspace within the residual complement.
4. Removing the localized residual component from wrong activations improves logits; injecting it into correct activations worsens them.
5. Random same-dimensional subspaces do not reproduce the effect.
6. The pattern replicates qualitatively across GPT-J, Pythia, and Llama 3.1.

The mathematical claims (Theorems 1–3, Propositions, the matched permutation result) are the statistical infrastructure that makes the empirical claims credible: they convert "wrong activations have larger residual norms" into a calibrated test with bias-corrected estimation and exact conditional null calibration.

### 12.3 What the paper does not claim

We do not claim:

- A complete smooth-manifold minimax theory for LLM arithmetic failures. The matching minimax lower bound is downgraded to a conjecture; we use the achievability rate only.
- That residual norm differences alone are mechanistic evidence. The matched permutation and causal patching steps are what carry the mechanistic claim, and we frame the paper that way.
- That the framework generalizes beyond two-digit addition. We restrict scope to the same setting as KT.
- That the helix is the only valid representation. We use it because KT establish it; alternative representations would be follow-up work.

### 12.4 Main reviewer attack and the response

The strongest expected reviewer objection is:

> "This is just Kantamneni–Tegmark plus residual norms."

The response is that residual norms by themselves are not the contribution. The contribution is the four-part chain:

$$
\text{representation (KT)} \;\to\; \text{failure geometry} \;\to\; \text{matched-difficulty statistical test} \;\to\; \text{causal patching with specificity baseline}.
$$

KT does not have the failure-geometry, matched-test, or specificity-baseline steps. The localized subspace $V$ and the per-failure-mode decomposition (on-curve, off-curve in-span, off-span) are specifically about wrong computations and do not appear in KT.

A second expected objection is that wrong examples may simply be harder, larger, noisier, or distributionally different inputs, and that the residual difference reflects input difficulty rather than representational geometry. The matched permutation construction (Theorem 8.3) is the response. We frame it not as a side detail but as the statistical core of the credibility argument: the headline empirical claim is conditional on matched arithmetic difficulty, not on raw counts.

### 12.5 Theorem level required for the workshop

For BlackboxNLP, the necessary mathematical content is:

1. Exact linear-span expectation calculation (Theorem 4.2(a)) with anisotropic generalization (Theorem 4.10).
2. Finite-sample null concentration (Theorem 4.2(c)).
3. Matched permutation conditional validity (Theorem 8.3) with Freedman-Lane-style fallback.
4. Span-recovery perturbation bound (Theorem 6.2).
5. Composition bound (Theorem 6.6) with centered tightening (Remark 6.7).
6. Causal Taylor proposition with corrected displacement and Hessian-bounded remainder (Proposition 9.3).
7. Cross-fitting Neyman-orthogonality (§7.3 Gateaux calculation).

The matching minimax lower bound (5.2b) is conjectural in this revision; we present only the two-point Le Cam bound (5.2a) as proven. The curved-manifold version of Theorem 5.2 (case (C)) replaces the strict common-normal-bundle assumption by an empirical drift bound $\eta_0$.

### 12.6 Venue

**BlackboxNLP 2026:** primary target. The workshop describes itself as analyzing and interpreting neural networks for NLP, including evaluation of explanation methods. The combination of a clear mechanistic object (the helix), a clear failure hypothesis (off-manifold residuals), a clear statistical test (matched permutation with calibrated null), and causal patching is exactly the analysis-with-interventions pattern the workshop tends to value.

**ACL or EMNLP main:** possible only if the empirical results across all three models are unusually clean and the paper is rewritten to lead with the interpretability finding rather than the mathematical infrastructure. We do not target this for the current submission.

**ICLR or NeurIPS:** unlikely without (a) the full minimax lower bound proven (currently a conjecture), and (b) a demonstration that the framework generalizes beyond two-digit addition. Both are out of scope for the current submission.

### 12.7 Working title

The previous mathematical title *Off-Manifold Failure: A Geometric Test for Localized Computational Errors in Language Models* is theorem-heavy and leads with infrastructure rather than finding. We expect the eventual paper title to be closer to one of:

- "Off the Number Manifold: Geometric Signatures of Arithmetic Failures in Language Models"
- "When Arithmetic Fails: Off-Manifold Residuals in Language Model Number Representations"

The current document keeps the technical title for review purposes.

### 12.8 What success looks like

A solid workshop paper requires that 5 or more of the 6 empirical claims listed above land cleanly on at least 2 of the 3 models. If fewer than 5 land or if results are split across models in confusing ways, the paper becomes a methodology contribution with mixed empirical results, which is a weaker but still publishable shape. If 2 or fewer empirical claims land, the test is not picking up real geometric structure, and the right response is to re-evaluate the representation hypothesis rather than push the paper through.

The synthetic toy validation (passed 27 of 27 pre-registered checks) gives us confidence that the test is correctly calibrated in controlled settings. Whether the real activations satisfy the generative-model assumptions tightly enough for the test to detect geometric structure is what the empirical work will establish.

---

## References

- [federer1959] H. Federer. Curvature measures. *Transactions of the American Mathematical Society*, 93:418–491, 1959.

- [niyogi2008] P. Niyogi, S. Smale, and S. Weinberger. Finding the homology of submanifolds with high confidence from random samples. *Discrete & Computational Geometry*, 39(1-3):419–441, 2008.

- [laurent2000] B. Laurent and P. Massart. Adaptive estimation of a quadratic functional by model selection. *The Annals of Statistics*, 28(5):1302–1338, 2000.

- [vershynin2018] R. Vershynin. *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge University Press, 2018.

- [wainwright2019] M. J. Wainwright. *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press, 2019.

- [tropp2015] J. A. Tropp. An introduction to matrix concentration inequalities. *Foundations and Trends in Machine Learning*, 8(1-2):1–230, 2015.

- [tyurin2010] I. S. Tyurin. New estimates of the convergence rate in the Lyapunov theorem. *Doklady Mathematics*, 82(2):760–762, 2010.

- [wedin1972] P. Å. Wedin. Perturbation bounds in connection with singular value decomposition. *BIT Numerical Mathematics*, 12(1):99–111, 1972.

- [stewart1990] G. W. Stewart and J. Sun. *Matrix Perturbation Theory*. Academic Press, 1990.

- [magnus1999] J. R. Magnus and H. Neudecker. *Matrix Differential Calculus with Applications in Statistics and Econometrics*. Wiley, revised edition, 1999.

- [davies1980] R. B. Davies. The distribution of a linear combination of $\chi^{2}$ random variables. *Applied Statistics*, 29(3):323–333, 1980.

- [romano2005] J. P. Romano and M. Wolf. Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4):1237–1282, 2005.

- [tsybakov2009] A. B. Tsybakov. *Introduction to Nonparametric Estimation*. Springer, 2009.

- [lehmann2005] E. L. Lehmann and J. P. Romano. *Testing Statistical Hypotheses*. Springer, third edition, 2005.

- [chernozhukov2018] V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins. Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1):C1–C68, 2018.

- [singer2012] A. Singer and H.-T. Wu. Vector diffusion maps and the connection Laplacian. *Communications on Pure and Applied Mathematics*, 65(8):1067–1144, 2012.

- [coifman2006] R. R. Coifman and S. Lafon. Diffusion maps. *Applied and Computational Harmonic Analysis*, 21(1):5–30, 2006.

- [scholkopf1998] B. Schölkopf, A. Smola, and K.-R. Müller. Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*, 10(5):1299–1319, 1998.

- [wang2022] K. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. *ICLR*, 2023.

- [kantamneni2025] S. Kantamneni and M. Tegmark. Language models use trigonometry to do addition. *arXiv:2502.00873*, 2025.

- [ingster1993] Y. I. Ingster. Asymptotically minimax hypothesis testing for nonparametric alternatives, I–III. *Mathematical Methods of Statistics*, 2(2):85–114, 2(3):171–189, 2(4):249–268, 1993.

- [ingster2003] Y. I. Ingster and I. A. Suslina. *Nonparametric Goodness-of-Fit Testing under Gaussian Models*. Springer Lecture Notes in Statistics 169, 2003.

- [baraud2002] Y. Baraud. Non-asymptotic minimax rates of testing in signal detection. *Bernoulli*, 8(5):577–606, 2002.

- [collier2017] O. Collier, L. Comminges, and A. B. Tsybakov. Minimax estimation of linear and quadratic functionals on sparsity classes. *The Annals of Statistics*, 45(3):923–958, 2017.

- [aamari2019] E. Aamari and C. Levrard. Nonasymptotic rates for manifold, tangent space and curvature estimation. *The Annals of Statistics*, 47(1):177–204, 2019.

- [freedmanlane1983] D. Freedman and D. Lane. A nonstochastic interpretation of reported significance levels. *Journal of Business & Economic Statistics*, 1(4):292–298, 1983.

- [andersonrobinson2001] M. J. Anderson and J. Robinson. Permutation tests for linear models. *Australian & New Zealand Journal of Statistics*, 43(1):75–88, 2001.

- [hemerik2018] J. Hemerik and J. Goeman. Exact testing with random permutations. *TEST*, 27:811–825, 2018.

- [dmet2025] *DMET (Discrete Manifold Evolution Theory of large language models).* *arXiv:2505.20340*, 2025. *Author list to be filled in from arXiv metadata before submission. Cited as related framework: LLM trajectories on low-dimensional manifolds.*

- [diffgeom2024] *Diffusion-geometry estimators for tangent spaces, dimension and curvature.* *arXiv:2411.04100*, 2024. *Author list to be filled in from arXiv metadata before submission. Cited as nonparametric robustness check for $M$-estimation.*

- [regdim2025] *Regression-based intrinsic dimension with curvature modeling.* *arXiv:2510.15141*, 2025. *Author list to be filled in from arXiv metadata before submission. Cited as alternative for tangent/curvature validation.*

- [conmy2023] A. Conmy, A. Mavor-Parker, A. Lynch, S. Heimersheim, and A. Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. *NeurIPS*, 2023. *Cited for the activation-patching / downstream-forward-pass formalism used in Definition 9.2.*
