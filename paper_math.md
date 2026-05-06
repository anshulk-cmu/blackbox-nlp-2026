# Off-Manifold Failure: A Geometric Test for Localized Computational Errors in Language Models

*Mathematical Proposal — Draft for Review*

**Authors.** Anshul Kumar and Barnábás Póczos

## Abstract

This document is a mathematical proposal. It lays out the problem we want to study, the test statistic we propose, and the theorems we can prove. Each theorem is stated as a target with a proof outline; the full proofs are work in progress.

We are sharing this in advance of running the empirical pipeline on GPT-J 6B, Pythia 6.9B, and Llama 3.1 8B (target venue: BlackboxNLP 2026, direct submission deadline July 17, 2026), so that the math can be reviewed and tightened in parallel with the experimental work rather than after. We flag every place where we are uncertain about the statement, the proof structure, or the constants. Section 11 collects these into a list of open questions we would like your input on.

The technical content we propose: (1) a non-asymptotic concentration bound for an off-manifold residual statistic, (2) a localized version with a power upper bound and a matching minimax lower bound via Le Cam chaining, (3) a finite-sample manifold-recovery bound for the parametric estimator with operator-norm Wedin and an explicit misspecification term, (4) a cross-fitting protocol with Neyman-orthogonal influence function, (5) exact conditional validity of a stratified permutation null, (6) a causal proposition with Hessian-controlled second-order Taylor remainder.

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

Define the correctness label

$$
y(a, b) =
\begin{cases}
1 & \text{if } \arg\max_{t} \mathrm{logit}_{t}(h(a, b)) = s, \\
0 & \text{otherwise.}
\end{cases}
$$

We refer to the set $\{(a, b) : y(a, b) = 1\}$ as the *correct population* and to $\{(a, b) : y(a, b) = 0\}$ as the *wrong population*.

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

(a) *On-curve, in-span*: both small. $h$ lies on $M_{C}$ at a nearby integer.
(b) *Off-curve, in-span*: $\|r_{\mathrm{within}}(h)\|$ large, $\|r_{S}(h)\|$ small. $h$ lies in $M_{S}$ but at a non-integer parameter value.
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

**Lemma 3.2 (Without-loss-of-generality re-centering of $\xi$).**
Let $\xi(a, b) \in \mathbb{R}^{d}$ be any perturbation. Decompose $\xi = P^T_{m_{c}(a,b)} \xi + P^N_{m_{c}(a,b)} \xi$, and define

$$
\widetilde{m}_{c}(a, b) := m_{c}(a, b) + P^T_{m_{c}(a,b)} \xi(a, b),
\quad
\widetilde{\xi}(a, b) := P^N_{\widetilde{m}_{c}(a, b)} \xi(a, b).
$$

Then to first order in the curvature of $M$ at $m_{c}(a, b)$ we have $\widetilde{m}_{c}(a, b) \in M$ and $\widetilde{\xi}(a, b) \in N_{\widetilde{m}_{c}(a, b)} M$.

*Proof outline.* The tangent component $P^T_{m_{c}(a,b)} \xi$ moves $m_{c}$ along $M$ at linear rate $1 + O(\kappa \cdot \|P^T \xi\|)$, where $\kappa$ is the local sectional curvature. Hence to first order $\widetilde{m}_{c} \in M$. The remaining component $\widetilde{\xi}$ is in the normal space at the new base point. The second-order correction is absorbed into the linearization error of Section 4.3 below. We have not written out the exponential-map calculation rigorously; this is a place where Barnábás's input would help confirm the rate. ∎

**Remark 3.3 (Same noise across populations).**
Assumption GM(iii) requires $\Sigma_{\varepsilon}$ to be the same for correct and wrong populations. This is realistic for activation noise driven by within-prompt token variation, but may fail if the wrong population is drawn from a categorically different prompt distribution. We address relaxation of this assumption in Section 4 Remark 4.9.

### 3.1 Boundedness assumption

**Assumption 3.4 (BD, boundedness).**
There exists $D_{\max} < \infty$ (depending on the model and layer) such that, with probability at least $1 - \delta$ over the noise, $\|h_{c}^{(i)}\|_{2} \leq D_{\max}$ for every correct activation in the analysis sample, and similarly for wrong activations.

For sub-Gaussian noise with parameter $\sigma$ and a clean signal $m_{p}(a, b)$ uniformly bounded on a compact $M$, standard sub-Gaussian maximal inequalities give $D_{\max} = O(\|m_{p}\|_{\infty} + \sigma \sqrt{d + \log(1/\delta)})$ with probability $1 - \delta$. For activations after layer normalization, $D_{\max} = O(\sqrt{d})$ typically.

### 3.2 Regularity assumption on the manifold

**Assumption 3.5 (REG, regularity).**
There exist $\tau_{\min} > 0$ and $\kappa_{\max} < \infty$ such that $\tau(M) \geq \tau_{\min}$ uniformly and the maximum sectional curvature of $M$ is bounded by $\kappa_{\max}$ at every point of the data support. Moreover $\sigma \cdot \kappa_{\max} \leq c_{0}$ for a small absolute constant $c_{0} \in (0, 1]$.

The condition $\sigma \kappa_{\max} \leq c_{0}$ ensures that the typical noise displacement is small relative to the manifold's curvature scale, so the linearization of Section 4.3 is valid. We verify this empirically per model and layer before running the main experiments.

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

where $R_{1}(\sigma, \kappa_{\max}) = O(\sigma^{2} \kappa_{\max}^{2})$ is a linearization-error term that vanishes when $M$ is linear.

(b) **Null behavior.** Under $H_{0}: \mu_{\xi} = 0$ and $\Sigma_{\xi} = 0$, $\mathbb{E}[T_{n}] = R_{1}(\sigma, \kappa_{\max})$ which is identically zero when $M$ is linear and $O(\sigma^{2} \kappa_{\max}^{2})$ in general.

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

**Lemma 4.3 (Tubular-neighborhood projection — proposed).**
Let $p \in M$ and $\delta \in \mathbb{R}^{d}$ with $\|\delta\|_{2} < \tau(M) / 2$. We propose

$$
\Pi_{M}(p + \delta) = p + P^T_{p} \delta + R_{2}(\delta, p),
\quad
\|R_{2}(\delta, p)\|_{2} \leq \frac{\kappa_{\max}}{2} \|P^N_{p} \delta\|_{2}^{2}. \tag{4.3}
$$

Consequently the residual at $h = p + \delta$ should satisfy

$$
r(h) = P^N_{p} \delta - R_{2}(\delta, p),
$$

with the same bound on $R_{2}$.

*Proof outline.* Federer [federer1959] proved that for $x$ within reach, the closest-point projection $\Pi_{M}$ is $C^{1}$ on $\mathbb{R}^{d} \setminus M$ with derivative

$$
D \Pi_{M}(p + \delta) = P^T_{p + O(\delta)}.
$$

Differentiating once more gives the second fundamental form of $M$ at $p$, whose operator norm is bounded by $\kappa_{\max}$. By Taylor's theorem with integral remainder,

$$
\Pi_{M}(p + \delta) - p - P^T_{p} \delta
= \int_{0}^{1} (1 - s) D^{2} \Pi_{M}(p + s \delta)[\delta, \delta] \, ds.
$$

The second derivative is bounded in operator norm by $\kappa_{\max}$, giving $\|R_{2}\|_{2} \leq \tfrac{1}{2} \kappa_{\max} \|\delta\|_{2}^{2}$. The factor of $\|P^N_{p} \delta\|_{2}^{2}$ in place of $\|\delta\|_{2}^{2}$ comes from the fact that the tangent component is a free first-order motion along $M$, contributing zero to the second-order remainder; this is the standard tubular-neighborhood estimate (Niyogi–Smale–Weinberger [niyogi2008]).

*We are not fully sure the second-fundamental-form bound is sharp at the constant $1/2$; Federer's original bound is in this form but we have not verified Niyogi–Smale–Weinberger's reformulation matches.* ∎

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

**Remark 4.10 (Anisotropic effective dimension).**
For anisotropic $\Sigma_{\varepsilon}$, the chi-squared variance $2 k \sigma^{4}$ in Theorem 4.2 is replaced by $2 \mathrm{tr}((P^N \Sigma_{\varepsilon})^{2})$ (Magnus–Neudecker [magnus1999] Theorem 11.21). Define the *effective rank* of the residual covariance:

$$
k_{\mathrm{eff}} := \frac{(\mathrm{tr}(P^N \Sigma_{\varepsilon}))^{2}}{\mathrm{tr}((P^N \Sigma_{\varepsilon})^{2})}.
$$

For isotropic $\Sigma_{\varepsilon} = \sigma^{2} I$, $k_{\mathrm{eff}} = k$. For strongly anisotropic noise (e.g., low-rank covariance), $k_{\mathrm{eff}} \ll k$. Theorem 4.2 continues to hold with $k$ replaced by $k_{\mathrm{eff}}$ throughout, giving sharper bounds when noise is concentrated.

**Remark 4.11 (Berry–Esseen rate to normality).**
By the Berry–Esseen theorem with the modern Esseen constant [tyurin2010],

$$
\sup_{t \in \mathbb{R}} \left|\mathbb{P}\!\left[\tfrac{T_{n} - \mathbb{E}[T_{n}]}{\sqrt{\mathrm{Var}(T_{n})}} \leq t\right] - \Phi(t)\right|
\leq \frac{0.4748 \cdot \mathbb{E}|X_{1} - \mathbb{E}[X_{1}]|^{3}}{\sqrt{n} \cdot (\mathrm{Var}(X_{1}))^{3/2}}
= O\!\left(\sqrt{\tfrac{k}{n}}\right).
$$

This gives the explicit rate at which the standardized $T_{n}$ is approximately standard normal, supporting Wald-type confidence intervals when $n \gg k$.

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

**Theorem 5.2 (Proposed: localization power and minimax lower bound).**
Suppose Assumptions GM, BD, REG hold with isotropic Gaussian noise. Suppose $\xi(a, b) \in V$ almost surely (so $\mu_{\xi} \in V$ and $\Sigma_{\xi}$ has support in $V$), where $V$ is a fixed $r$-dimensional subspace satisfying $V \subseteq \bigcap_{p \in M} N_{p} M$ (the common normal bundle, the intersection of normal spaces over the data support of $M$). We propose:

(a) **Achievability (upper bound).** For the test that rejects $H_{0}$ when $T_{n}^{V} > c_{\alpha}$ with $c_{\alpha}$ calibrated to level $\alpha$, the test should achieve power at least $1 - \beta$ provided

$$
n \;\geq\; C_{1} \cdot \frac{r \sigma^{4}}{\Delta^{4}} \cdot \log\!\bigl(\tfrac{1}{\beta}\bigr), \tag{5.1}
$$

where $\Delta := \|\mu_{\xi}\|_{2}$ and $C_{1}$ is an absolute constant.

(b) **Lower bound (minimax).** Let $\mathcal{P}_{\Delta}$ denote the class of distributions in Assumption GM with $\|\mu_{\xi}\|_{2} \geq \Delta$ and $\xi \in V$. Any test $\psi$ with size at most $\alpha$ that achieves uniform power $\inf_{P \in \mathcal{P}_{\Delta}} \mathbb{P}_{P}[\psi = 1] \geq 1 - \beta$ requires

$$
n \;\geq\; C_{2} \cdot \frac{r \sigma^{4}}{\Delta^{4}} \cdot \log\!\bigl(\tfrac{1}{\beta(1 - \alpha)}\bigr), \tag{5.2}
$$

for an absolute constant $C_{2}$. Hence $T_{n}^{V}$ is minimax-optimal in $r, \sigma^{2}, \Delta^{2}$ up to absolute constants.

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
P_V r(h_{c}) &= P_V \varepsilon_{c} + O_{\mathbb{P}}(\sigma^{2} \kappa_{\max}), \\
P_V r(h_{w}) &= \xi + P_V \varepsilon_{w} + O_{\mathbb{P}}(\sigma^{2} \kappa_{\max}).
\end{aligned}
$$

*Proof outline.* By Corollary 4.4, $r(h_{c}) = P^N \varepsilon_{c} + O_{\mathbb{P}}(\sigma^{2} \kappa_{\max})$. Apply $P_V$:

$$
P_V r(h_{c}) = P_V P^N \varepsilon_{c} + O_{\mathbb{P}}(\sigma^{2} \kappa_{\max}).
$$

Since $V \subseteq N_{p} M$ for every $p \in M$ (the common-normal-bundle hypothesis), $P_V$ acts as the identity on the normal space intersected with $V$, so $P_V P^N_{p} = P_V$ for every $p$ in the data support. Hence $P_V r(h_{c}) = P_V \varepsilon_{c}$ to leading order.

For $r(h_{w}) = \xi + P^N \varepsilon_{w} + O_{\mathbb{P}}(\sigma^{2} \kappa_{\max})$: $P_V \xi = \xi$ (since $\xi \in V$), and $P_V P^N \varepsilon_{w} = P_V \varepsilon_{w}$ as above. Combining gives the claim. ∎

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

### 5.6 Step 4: minimax lower bound via Le Cam

We now prove the matching lower bound, which is one of the key upgrades over the previous draft.

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

Tensorizing across $n$ i.i.d. wrong samples gives the stated bound: $\chi^{2}$ is sub-additive under products and the result is a product of $n$ identical Gaussians, giving the exponential $n \Delta^{2}/\sigma^{2}$. The correct samples contribute zero to the divergence since their distribution is unchanged. ∎

*Proposed proof of Theorem 5.2(b), the lower bound.* By Lemma 5.5, any test with size $\alpha$ and uniform power $1 - \beta$ over $\mathcal{P}_{\Delta}$ requires $\chi^{2}(\mathcal{P}_{1}^{\otimes n} \| \mathcal{P}_{0}^{\otimes n}) \geq (1 - \alpha - \beta)^{2}$. Combine with (5.3):

$$
\exp(n \Delta^{2} / \sigma^{2}) - 1 \geq (1 - \alpha - \beta)^{2},
$$

so $n \Delta^{2} / \sigma^{2} \geq \log\bigl(1 + (1 - \alpha - \beta)^{2}\bigr)$. This gives $n \geq C \sigma^{2} / \Delta^{2} \cdot \log(\cdots)$ for a suitable constant.

To get the $\Delta^{4}$ rate (which matches the upper bound), apply Le Cam to the harder problem of distinguishing $\mathcal{P}_{0}$ from a *mixture* of $r$ point alternatives $\{v_{1}, \dots, v_{r}\}$ each of norm $\Delta/\sqrt{2}$ in mutually orthogonal directions in $V$. This is the standard Fano-style approach (Wainwright [wainwright2019] Chapter 15): the chi-squared between the mixture and $\mathcal{P}_{0}$ is $\frac{1}{r}(\exp(n \Delta^{2} / (2 r \sigma^{2})) - 1)$, and the Fano lower bound becomes $n \Delta^{2} / (2 r \sigma^{2}) \geq \log(r / e)$, yielding

$$
n \geq C_{2} \cdot \frac{r \sigma^{2}}{\Delta^{2}} \cdot \log r.
$$

The full $\Delta^{4}$ rate is obtained by chaining the standard optimization argument over the alternative class; the explicit constant comes from Tsybakov [tsybakov2009] Theorem 2.7. The matching $r \sigma^{4} / \Delta^{4}$ rate follows. ∎

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
Let Assumption GM hold with sub-Gaussian noise $\|\varepsilon\|_{\psi_{2}} \leq \sigma$. Let $\lambda_{\min}^{B} := \lambda_{\min}(\mathbb{E}[b b^{\top}])$ where $b = (b_{1}(a), \dots, b_{K}(a))^{\top}$ and the expectation is over the empirical distribution of integer labels in the data. Let $\sigma_{K}(C^*)$ denote the smallest singular value of the population coefficient matrix $C^* \in \mathbb{R}^{K \times d}$. Suppose the parametric class is well-specified, i.e. $\mathbb{E}[H_c \mid B] = B (C^*)^{\top}$ exactly. We propose to prove that with probability at least $1 - \delta$,

$$
\sin \theta_{\max}\bigl(\widehat{M}_{\mathrm{param}}, M\bigr)
\;\leq\;
C_{3} \cdot \frac{\sigma}{\lambda_{\min}^{B} \cdot \sigma_{K}(C^*)} \cdot \sqrt{\frac{d \log(d / \delta)}{n_c}}, \tag{6.1}
$$

for an absolute constant $C_{3}$.

Our proposed proof has two steps: (1) Frobenius bound on $\widehat{C}$ via matrix concentration, (2) Davis–Kahan / Wedin to convert Frobenius to sin-theta. The second step uses the operator-norm form of Wedin (rather than the Frobenius Davis–Kahan), which we believe gives a sharper constant by factor $\sqrt{K}$.

### 6.3 Step 1: Frobenius bound on $\widehat{C}$

**Lemma 6.3 (OLS estimator concentration).**
Under the hypotheses of Theorem 6.2, with probability at least $1 - \delta$,

$$
\|\widehat{C} - C^*\|_{\mathrm{op}}
\;\leq\;
C \cdot \sigma \cdot \sqrt{\frac{d + K \log(d / \delta)}{n_c \cdot \lambda_{\min}^{B}}}. \tag{6.2}
$$

Converting to Frobenius via $\|\cdot\|_{F} \leq \sqrt{K} \|\cdot\|_{\mathrm{op}}$:

$$
\|\widehat{C} - C^*\|_{F}
\;\leq\;
C \cdot \sigma \cdot \sqrt{\frac{K(d + K \log(d / \delta))}{n_c \cdot \lambda_{\min}^{B}}}. \tag{6.3}
$$

*Proof outline.* Write $H_c = B (C^*)^{\top} + E$ with $E \in \mathbb{R}^{n_c \times d}$ a noise matrix with i.i.d. sub-Gaussian rows of parameter $\sigma$. The OLS residual is

$$
\widehat{C} - C^* = (B^{\top} B)^{-1} B^{\top} E,
$$

a transpose convention. Apply the matrix concentration bound for sub-Gaussian matrices (Vershynin [vershynin2018] Section 4.7): $\|B^{\top} E / n_c\|_{\mathrm{op}} \leq C \sigma \sqrt{(d + K)/n_c} \cdot \sqrt{\|B^{\top} B / n_c\|_{\mathrm{op}}}$ with the stated probability. Invert $(B^{\top} B / n_c)$ which has minimum eigenvalue $\lambda_{\min}^{B} (1 - O(\sqrt{K \log K / n_c}))$ by matrix Bernstein (Tropp [tropp2015]); the leading term is $\lambda_{\min}^{B}$ for $n_c \gg K \log K$. Combine. ∎

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
\underbrace{C_{3} \cdot \frac{\sigma}{\lambda_{\min}^{B} \sigma_{K}(C^*)} \sqrt{\frac{d \log(d / \delta)}{n_c}}}_{\text{variance}}. \tag{6.5}
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

(The orthogonality of the influence function to manifold-estimation errors at first order is what makes cross-fitting bias-free; this is the Neyman-orthogonality property of Chernozhukov 2018.) Hence

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

Wrong samples are systematically harder than correct samples (more carries, larger sums). A naive permutation null would reject for *any* between-population difference, including pure input-difficulty differences that have nothing to do with manifold geometry. We propose stratified within-bin permutation as a fix, and we believe this can be shown to give exact conditional Type I control via the Lehmann–Romano permutation-group argument.

### 8.1 The bin partition and exchangeability

**Definition 8.1 (Coarse bin function).**
Define $\phi: \{0, \dots, 99\}^{2} \to \mathcal{B}$, where $\mathcal{B}$ is a finite set of bins, by

$$
\phi(a, b) := \bigl(\mathrm{sumbin}(a + b),\, \mathrm{carry}(a, b),\, \mathrm{decile}(a),\, \mathrm{decile}(b),\, \mathrm{tokclass}(a, b)\bigr).
$$

Each component takes finitely many values; in our pre-registration the total number of bins is $|\mathcal{B}| = 4 \times 4 \times 10 \times 10 \times 2 = 3200$, of which approximately $200$–$300$ are non-empty on the 10{,}000-pair grid.

### 8.2 Conditional null and exchangeability

**Definition 8.2 (Conditional null).**
The conditional null hypothesis is

$$
H_{0}^{\mathrm{cond}}: \quad
\bigl\{(H_c^{(i)}, y_{i})\bigr\}_{i \in \phi^{-1}(B)}
\overset{d}{=}
\bigl\{(H_c^{(\pi(i))}, y_{\pi(i)})\bigr\}_{i \in \phi^{-1}(B)}
\quad \forall B \in \mathcal{B}, \forall \pi \in \mathfrak{S}(\phi^{-1}(B)).
$$

That is, within each bin, the joint distribution of $(h, y)$ is invariant under permutation of indices.

**Theorem 8.3 (Proposed: exact conditional validity).**
Let $T_{n}^{\mathrm{matched}}$ be the bin-weighted average statistic defined in the main paper Section 3.3, and let $c_{\alpha}$ be its $(1 - \alpha)$-quantile under the bin-restricted permutation distribution. Then under $H_{0}^{\mathrm{cond}}$,

$$
\mathbb{P}[T_{n}^{\mathrm{matched}} > c_{\alpha} \mid \{\phi(a_{i}, b_{i})\}_{i}] \leq \alpha.
$$

*Proof outline.* Conditional on the bin assignments and the within-bin index sets, the joint exchangeability condition of Definition 8.2 implies that the within-bin labels $\{y_{i}\}_{i \in \phi^{-1}(B)}$ are exchangeable for each $B$. Hence any statistic computed as a sum (or weighted sum) of within-bin functions of $(h, y)$ is itself exchangeable under the within-bin permutation group $\prod_{B} \mathfrak{S}(\phi^{-1}(B))$. By Lehmann–Romano [lehmann2005] Theorem 15.2.1 applied to this permutation group, the permutation-distribution quantile $c_{\alpha}$ is exact: $\mathbb{P}[T_{n}^{\mathrm{matched}} > c_{\alpha}] \leq \alpha$ under $H_{0}^{\mathrm{cond}}$. ∎

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
Let $\widehat{M} \subset \mathbb{R}^{d}$ denote the estimated manifold, $V \subset \widehat{M}^{\perp}$ the localized failure subspace, $V^{\mathrm{rand}}$ a random subspace of the same dimension as $V$, and $\widehat{\mu}_{\xi}$ the cross-fitted estimate of $\mu_{\xi}$ from wrong-population residuals projected onto $V$. Let $\mathrm{LD}(h)$ denote the logit-difference at the answer position (Wang–Variengien [wang2022]). Then:

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
Assume the model's logit-difference $\mathrm{LD}: \mathbb{R}^{d} \to \mathbb{R}$ is twice continuously differentiable, with operator-norm-bounded Hessian: $\sup_{h \in B(h_{c}, \rho)} \|\nabla^{2} \mathrm{LD}(h)\|_{\mathrm{op}} \leq L$ for $\rho > 2 \|\widehat{\mu}_{\xi}\|$. Then

$$
\bigl|\mathrm{ACE}_{S} - \mathrm{ACE}_{S}^{\mathrm{linear}}\bigr|
\;\leq\;
\frac{L}{2} \|\widehat{\mu}_{\xi}\|^{2}, \tag{9.1}
$$

where the first-order linear prediction is

$$
\mathrm{ACE}_{S}^{\mathrm{linear}} := \mathbb{E}_{h_{c}}\!\bigl[ \langle \nabla \mathrm{LD}(h_{c}), U_{V} \widehat{\mu}_{\xi} \rangle \bigr]
= \widehat{\mu}_{\xi}^{\top} U_{V}^{\top} \mathbb{E}_{h_{c}}[\nabla \mathrm{LD}(h_{c})].
$$

Moreover, the magnitude lower bound

$$
\bigl|\mathrm{ACE}_{S}^{\mathrm{linear}}\bigr|
\;\geq\;
\|\widehat{\mu}_{\xi}\| \cdot \|U_{V}^{\top} \mathbb{E}[\nabla \mathrm{LD}(h_{c})]\| \cdot \cos \theta_{\xi, \nabla}, \tag{9.2}
$$

holds, where $\theta_{\xi, \nabla}$ is the angle between $\widehat{\mu}_{\xi}$ and $U_{V}^{\top} \mathbb{E}[\nabla \mathrm{LD}(h_{c})]$.

*Proof outline.* Apply Taylor's theorem with integral remainder:

$$
\mathrm{LD}(h + \Delta h) - \mathrm{LD}(h)
= \langle \nabla \mathrm{LD}(h), \Delta h \rangle + \int_{0}^{1} (1 - t) \langle \nabla^{2} \mathrm{LD}(h + t \Delta h) \Delta h, \Delta h \rangle \, dt.
$$

With $\Delta h = U_{V} \widehat{\mu}_{\xi}$ (the injection patch), $\|\Delta h\| = \|\widehat{\mu}_{\xi}\|$ since $U_{V}$ is orthonormal. The remainder term is bounded by $\frac{1}{2} L \|\Delta h\|^{2} = \frac{L}{2} \|\widehat{\mu}_{\xi}\|^{2}$. Take expectations over $h_{c}$ and $\nabla^{2} \mathrm{LD}$ to get (9.1).

The magnitude bound is Cauchy–Schwarz: $|\langle u, v\rangle| \geq \|u\| \cdot \|v\| \cdot \cos \theta(u, v)$ applied to $u = \widehat{\mu}_{\xi}$ and $v = U_{V}^{\top} \mathbb{E}[\nabla \mathrm{LD}]$. ∎

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

This document is a proposal, not a finished result. The following are the specific places where we are uncertain, where the proof outlines are sketches rather than full proofs, or where the constants and assumptions could likely be tightened. We would value your feedback on each.

1. **The WLOG re-centering of $\xi$.** Lemma 3.2 claims that any tangent component of $\xi$ can be absorbed into a different choice of $m_{c}(a, b) \in M$, leaving the remaining $\widetilde{\xi}$ orthogonal to the tangent space. We have written this as a first-order claim; the second-order error comes from the curvature of $M$ at $m_{c}$. We have not written out the exponential-map calculation rigorously and would value confirmation that the rate $O(\kappa \cdot \|P^T \xi\|)$ is correct.

2. **Sharpness of the linearization-error term $R_{1}$.** Theorem 4.2's remainder $R_{1}(\sigma, \kappa_{\max})$ is bounded by $O(\sigma^{2} \kappa_{\max}^{2})$. We believe the $\sigma^{2}$ scaling (rather than $\sigma$) is correct because the tangent-orthogonal noise component is what enters the second-order Taylor remainder, not the tangential component. The exact constant has not been tracked through.

3. **Exact constants $C_{1}, C_{2}, C_{3}, C_{4}$.** The absolute constants in Theorems 5.2, 6.2, and 6.6 are stated abstractly. Tracking them through the proofs is mechanical but would yield explicit numerical bounds and tighten the practical interpretability of the rates. We have not done this and would welcome guidance on which constants matter most for the application.

4. **The minimax lower bound's matching rate.** The lower bound in Theorem 5.2(b) gives $n \geq C_{2} r \sigma^{2} / \Delta^{2} \log r$ via two-point Le Cam directly. The full $r \sigma^{4} / \Delta^{4}$ rate is obtained by the standard Fano chaining argument over a packing of the alternative class (Tsybakov [tsybakov2009] Theorem 2.7). *The chaining step is the place where we are least confident in the writeup.* We have sketched the argument but not rederived it carefully; in particular the construction of the packing of normal-bundle perturbations of norm $\Delta / \sqrt{2}$ in mutually orthogonal directions in $V$ has not been verified to give the exact $r \sigma^{4} / \Delta^{4}$ constant claimed.

5. **Operator-norm Wedin vs Frobenius Davis–Kahan in Theorem 6.2.** We use the operator-norm form of Wedin because we believe it gives a constant that is sharper by factor $\sqrt{K}$. We have not double-checked that the dependence on $K$ in the OLS Frobenius bound (Lemma 6.3) does not re-introduce the $\sqrt{K}$ that operator-norm Wedin saves.

6. **Cross-fitting under the alternative.** Theorem 7.2 gives asymptotic normality at the null. The non-asymptotic version, with explicit finite-sample corrections at the alternative, is what would actually be used to set the sample size in the experiments. This is a standard but tedious calculation in the Chernozhukov et al. 2018 machinery; we have not done it.

7. **Higher-order Taylor for Proposition 9.3.** The remainder bound is at order $L \|\widehat{\mu}_{\xi}\|^{2} / 2$. Higher-order corrections involving $\nabla^{3} \mathrm{LD}$ would tighten the bound when the LD landscape has bounded third derivatives, which holds generically but has not been verified for our specific transformer readouts. The synthetic toy in the companion document found a magnitude-vs-Taylor ratio of about $5\times$ in one calibrated case; understanding whether this is mostly argmax-flip non-smoothness or higher-order Taylor would help.

8. **The argmax-flip / smooth-max question.** Real logit-difference $\mathrm{LD}(h) = \mathrm{logit}_{s}(h) - \max_{t \neq s} \mathrm{logit}_{t}(h)$ is non-smooth at points where the argmax over $t \neq s$ flips. We propose using log-sum-exp at temperature $\tau$ as a smooth surrogate, but the dependence of the Hessian bound $L$ on $\tau$ near argmax flips ($L = O(1/\tau)$) means the bound deteriorates as $\tau \to 0$. We do not have a clean way to handle this and would value advice on whether a different smoothing (e.g., random tie-breaking) gives a uniformly bounded $L$.

9. **Anisotropic noise generalization.** The remarks in Section 4 sketch the generalization of Theorem 1 to anisotropic noise via effective rank $k_{\mathrm{eff}} = \mathrm{tr}(P^N \Sigma_{\varepsilon})^{2} / \mathrm{tr}((P^N \Sigma_{\varepsilon})^{2})$. We have not propagated this through Theorems 2 and 3. Real activations after layer normalization are unlikely to be isotropic; the empirical work will need the anisotropic version.

10. **What is the right noise model for activations?** Real activations are a deterministic function of the input plus within-prompt structural variation. Modeling this as i.i.d. sub-Gaussian noise is standard but is a substantial idealization. We would welcome perspective on whether sub-Weibull (heavier-tailed) noise is more realistic and how the rates would change.

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

1. Exact linear-span expectation calculation (Theorem 4.2(a)).
2. Finite-sample null concentration (Theorem 4.2(c)).
3. Matched permutation conditional validity (Theorem 8.3).
4. Span-recovery perturbation bound (Theorem 6.2).
5. Causal Taylor proposition with corrected displacement and Hessian-bounded remainder (Proposition 9.3).

The minimax lower bound is bonus content; we plan to include it as conjectural with sketched intuition rather than as a stated theorem. Curved-manifold extensions are deferred to follow-up work.

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
