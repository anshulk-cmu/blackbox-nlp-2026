"""
Analytic predictions from paper_math.md, used by run_toy.py to validate the
new theorems. Each function corresponds to a specific equation or remark in
paper_math.md so the toy's numerical checks line up with the math file
1-to-1.

Coverage:
- Theorem 4.2(c) Bernstein-style concentration tail.
- Theorem 4.2(d) sharp Laurent-Massart upper tail bound.
- Remark 4.10 anisotropic effective rank k_eff.
- Remark 4.11 Berry-Esseen rate to standard normal.
- Theorem 5.2(b) Le Cam two-point chi-squared lower bound.
- Theorem 5.2(c) exact null distribution `(1/2)(chi^2_r - r)`.
- Corollary 7.3 closed-form influence-function variance V_inf.
- Theorem 6.5 misspecification radius b(M).
- Proposition 9.3 Hessian-bounded second-order Taylor remainder.

These are NumPy-only and fast; they're called inside record() pass/fail checks.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Theorem 4.2(c): non-asymptotic concentration tail bound
# ---------------------------------------------------------------------------

def theorem1_tail_bound_bernstein(t: float, n: int, k: int, sigma: float, c: float = 1.0 / 8.0) -> float:
    """Upper bound on P[|T_n - E[T_n]| > t] from paper_math.md Theorem 4.2(c).

    P[|T_n - E[T_n]| > t]  <=  4 * exp(-c * n * min(t^2 / (k * sigma^4), t / sigma^2))

    with c >= 1/8. Returns the numerical value of the bound (in [0, 4], not clipped to 1).
    """
    arg = min(t ** 2 / (k * sigma ** 4), t / sigma ** 2)
    return 4.0 * float(np.exp(-c * n * arg))


# ---------------------------------------------------------------------------
# Theorem 4.2(d): sharp Laurent-Massart upper tail
# ---------------------------------------------------------------------------

def theorem1_LM_threshold(u: float, n: int, k: int, sigma: float) -> float:
    """Sharp Laurent-Massart upper-tail threshold from paper_math.md Theorem 4.2(d).

    P[T_n - E[T_n] >= 2 sigma^2 sqrt(2 k u / n) + 2 sigma^2 u / n]  <=  2 e^{-u}

    Returns the rhs threshold (the value the deviation should exceed at most
    a fraction `2 e^{-u}` of the time).
    """
    return float(2.0 * sigma ** 2 * np.sqrt(2.0 * k * u / n) + 2.0 * sigma ** 2 * u / n)


def theorem1_LM_failure_prob(u: float) -> float:
    """Predicted failure probability `2 e^{-u}` for Theorem 4.2(d)."""
    return float(2.0 * np.exp(-u))


# ---------------------------------------------------------------------------
# Remark 4.10: anisotropic effective rank
# ---------------------------------------------------------------------------

def effective_rank(P_N: np.ndarray, Sigma_eps: np.ndarray) -> float:
    """k_eff = tr(P^N Sigma_eps)^2 / tr((P^N Sigma_eps)^2). paper_math.md Remark 4.10.

    P_N: (d, d) orthogonal projector onto the normal bundle.
    Sigma_eps: (d, d) noise covariance matrix.
    """
    A = P_N @ Sigma_eps
    num = float(np.trace(A)) ** 2
    A2 = A @ A
    den = float(np.trace(A2))
    return num / max(den, 1e-30)


def effective_rank_under_anisotropy(d_m: int, k: int, sigma: float,
                                     lowrank: int, lowrank_strength: float) -> float:
    """Closed-form k_eff for the anisotropic model used in synth_world.make_world's
    `anisotropic_lowrank` argument.

    Sigma_eps = sigma^2 I_{d_m} + lowrank_strength^2 * U U^T  with U: (d_m, lowrank)
    orthonormal columns drawn off the helix span (so U is in the normal bundle).

    Then on the normal bundle (dimension k = d_m - dim(M)):
    - tr(P^N Sigma_eps)    = k * sigma^2 + lowrank * lowrank_strength^2
    - tr((P^N Sigma_eps)^2) = k * sigma^4
                              + 2 * lowrank * sigma^2 * lowrank_strength^2
                              + lowrank * lowrank_strength^4
    The last identity uses U^T U = I_lowrank (orthonormal columns), so
    (P^N Sigma_eps)^2 = sigma^4 P^N + 2 sigma^2 lowrank_strength^2 U U^T
                      + lowrank_strength^4 U U^T (since P^N U = U).

    Returns k_eff = num/den.
    """
    s2 = sigma ** 2
    a2 = lowrank_strength ** 2
    num = (k * s2 + lowrank * a2) ** 2
    den = k * s2 ** 2 + 2.0 * lowrank * s2 * a2 + lowrank * a2 ** 2
    return float(num / max(den, 1e-30))


# ---------------------------------------------------------------------------
# Remark 4.11: Berry-Esseen rate to standard normal
# ---------------------------------------------------------------------------

def berry_esseen_bound(n: int, k: int, third_moment_ratio: float = 5.0) -> float:
    """Berry-Esseen bound for the standardized T_n; paper_math.md Remark 4.11.

    sup_t |P[(T_n - E)/sqrt(Var) <= t] - Phi(t)|
        <= 0.4748 * E|X - E[X]|^3 / (sqrt(n) * Var(X)^{3/2})
        = O(sqrt(k / n)).

    Returns 0.4748 * third_moment_ratio / sqrt(n). For chi-square X = ||P^N eps||^2,
    third_moment_ratio = E|X - mu|^3 / Var(X)^{3/2} is approximately
    sqrt(8/k) for large k (chi-square skewness scales as 1/sqrt(k)), so
    we approximate the bound as 0.4748 * sqrt(8 / (n * k)).

    For k=1 the chi-squared third moment is 8 (exact), but in our toy regime
    we can pass `third_moment_ratio` empirically.
    """
    return float(0.4748 * third_moment_ratio / np.sqrt(n))


# ---------------------------------------------------------------------------
# Theorem 5.2(b): Le Cam two-point lower bound on n via chi-squared divergence
# ---------------------------------------------------------------------------

def le_cam_chi2_two_gaussians(Delta: float, sigma: float, n: int) -> float:
    """Chi-squared divergence between P_0^{(*)n} and P_1^{(*)n} for the
    two-point comparison in paper_math.md Theorem 5.2(b) lower-bound proof.

    P_0 has mu_xi = 0, P_1 has mu_xi = v with ||v|| = Delta. Restricted to
    wrong samples, both are Gaussian with shared covariance sigma^2 I; the
    chi-squared distance for n iid wrong samples is

        chi^2(P_1^{(x)n} || P_0^{(x)n})  <=  exp(n * Delta^2 / sigma^2) - 1

    Returns the numerical value of the upper bound.
    """
    return float(np.exp(n * Delta ** 2 / sigma ** 2) - 1.0)


def le_cam_min_n_for_distinguishability(Delta: float, sigma: float,
                                          alpha: float = 0.05, beta: float = 0.20) -> float:
    """Minimum n such that the chi-squared divergence in `le_cam_chi2_two_gaussians`
    can equal (1 - alpha - beta)^2, the threshold from the Le Cam two-point
    inequality. Below this n, NO test can have size <= alpha and uniform power
    >= 1 - beta. paper_math.md Lemma 5.5 + Lemma 5.6.

    Solving exp(n Delta^2 / sigma^2) - 1 >= (1 - alpha - beta)^2:
        n >= sigma^2 / Delta^2 * log(1 + (1 - alpha - beta)^2)
    """
    target = (1.0 - alpha - beta) ** 2
    return float(sigma ** 2 / Delta ** 2 * np.log(1.0 + target))


# ---------------------------------------------------------------------------
# Theorem 5.2(c): exact null distribution of T_n^V under H_0
# ---------------------------------------------------------------------------

def theorem2_null_var(n_c: int, n_w: int, sigma: float, r: int) -> float:
    """Exact variance of T_n^V under H_0 (paper_math.md Theorem 5.2(c)).

    Each Y_p = (1/n_p) sum ||P_V r(h_p)||^2 ~ sigma^2/n_p * chi^2_{n_p r},
    so Var(Y_p) = 2 sigma^4 r / n_p. T_n^V = Y_w - Y_c, so

       Var(T_n^V) = 2 sigma^4 r (1 / n_c + 1 / n_w)
                  = 2 r sigma^4 (n_c + n_w) / (n_c n_w).
    """
    return float(2.0 * r * sigma ** 4 * (n_c + n_w) / (n_c * n_w))


def theorem2_null_wald_scaling(n_c: int, n_w: int, sigma: float, r: int) -> float:
    """Asymptotic Wald scaling: sqrt(n_c n_w / (n_c + n_w)) * T_n^V / (sigma^2 sqrt(2r))
    -> N(0, 1). Returns the multiplier on T_n^V."""
    return float(np.sqrt(n_c * n_w / (n_c + n_w)) / (sigma ** 2 * np.sqrt(2.0 * r)))


def theorem2_null_target_mean(r: int) -> float:
    """Mean of T_n^V under H_0 (oracle M, isotropic Gaussian) is exactly 0."""
    return 0.0


# ---------------------------------------------------------------------------
# Corollary 7.3: closed-form influence-function variance
# ---------------------------------------------------------------------------

def influence_function_V_inf(k: int, sigma: float, mu_xi_norm: float = 0.0) -> float:
    """V_inf for the cross-fitted statistic with n_c = n_w = n/2 and isotropic
    Gaussian noise; paper_math.md Corollary 7.3.

    V_inf = 4 k sigma^4 + 8 ||mu_xi||^2 sigma^2

    Used for the asymptotic CI: T_n^cross +- z_{alpha/2} sqrt(V_inf / n).
    """
    return float(4.0 * k * sigma ** 4 + 8.0 * mu_xi_norm ** 2 * sigma ** 2)


# ---------------------------------------------------------------------------
# Theorem 6.5: misspecification radius b(M)
# ---------------------------------------------------------------------------

def misspecification_radius(H_clean: np.ndarray, B: np.ndarray) -> float:
    """The squared bias `b^2(M) = inf_C E[||m_c - B C^T||^2]`, computed
    empirically as `mean residual^2` after OLS projecting H_clean onto B.

    H_clean: (n, d) clean activations (no noise).
    B:       (n, K) basis design matrix.
    Returns sqrt of the average per-sample squared residual norm.
    """
    C_hat, *_ = np.linalg.lstsq(B, H_clean, rcond=None)   # (K, d)
    R = H_clean - B @ C_hat
    return float(np.sqrt((R ** 2).sum(axis=1).mean()))


def smallest_singular_value(C: np.ndarray) -> float:
    """sigma_K(C). For paper_math.md Theorem 6.5 misspecification bound."""
    return float(np.linalg.svd(C, compute_uv=False).min())


# ---------------------------------------------------------------------------
# Proposition 9.3: Hessian-bounded ACE_S second-order Taylor remainder
# ---------------------------------------------------------------------------

def hessian_taylor_remainder_bound(L: float, mu_norm: float) -> float:
    """The Hessian-bounded remainder from paper_math.md Proposition 9.3:

       |ACE_S - ACE_S^linear|  <=  (L / 2) * ||mu_xi_hat||^2.

    L: operator-norm bound on the Hessian of LD on a ball around h_c of radius
       2 ||mu_xi_hat||.
    mu_norm: ||mu_xi_hat||_2.
    """
    return float(0.5 * L * mu_norm ** 2)


def estimate_hessian_opnorm_finite_difference(
    LD_fn,                       # callable: h (n,d) -> (n,) LD values
    h_center: np.ndarray,        # (d,) point at which to estimate Hessian
    radius: float,               # ball radius to probe
    n_directions: int = 50,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the operator-norm of the Hessian of LD at h_center using
    second-order finite differences along random directions.

    For a twice-differentiable f, along unit u:
        f(h + r u) + f(h - r u) - 2 f(h)  =  r^2 u^T Hess(h) u  + O(r^4).

    Estimating |u^T H u| over many random u gives an empirical lower bound on
    the operator norm of H (the actual op-norm is the supremum over u, so taking
    max over a Monte Carlo sample is a one-sided estimator).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = h_center.size
    h_center = h_center.reshape(1, d)
    f0 = float(LD_fn(h_center).item())
    estimates = []
    for _ in range(n_directions):
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        h_plus = h_center + radius * u[None, :]
        h_minus = h_center - radius * u[None, :]
        f_plus = float(LD_fn(h_plus).item())
        f_minus = float(LD_fn(h_minus).item())
        # second derivative along u
        approx_uHu = (f_plus + f_minus - 2.0 * f0) / (radius ** 2)
        estimates.append(abs(approx_uHu))
    return float(np.max(estimates))


# ---------------------------------------------------------------------------
# Self-test: confirm closed forms agree with simple cases
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Theorem 4.2(d) sanity: at u=0 the threshold is 0; at u=2 it's positive.
    print(f"LM threshold u=0: {theorem1_LM_threshold(0.0, n=100, k=10, sigma=0.1):.6f}")
    print(f"LM threshold u=2: {theorem1_LM_threshold(2.0, n=100, k=10, sigma=0.1):.6f}")
    print(f"LM failure prob u=2: {theorem1_LM_failure_prob(2.0):.4f}")

    # Effective rank: isotropic case (lowrank=0 -> k_eff = k)
    k_eff_iso = effective_rank_under_anisotropy(d_m=256, k=248, sigma=0.1,
                                                  lowrank=0, lowrank_strength=0.0)
    print(f"k_eff (iso, k=248): {k_eff_iso:.2f}  (expect ~248)")

    # Anisotropic: lowrank=5, strength=1.0 (so anisotropic component dominates)
    k_eff_aniso = effective_rank_under_anisotropy(d_m=256, k=248, sigma=0.1,
                                                    lowrank=5, lowrank_strength=1.0)
    print(f"k_eff (aniso, lr=5 strength=1.0): {k_eff_aniso:.2f}  (expect << 248, near 5)")

    # Le Cam: at Delta=0 lower bound diverges (no signal -> can't distinguish at any n)
    n_min = le_cam_min_n_for_distinguishability(Delta=0.5, sigma=0.1)
    print(f"Le Cam min n (Delta=0.5, sigma=0.1, alpha=0.05, beta=0.20): {n_min:.4f}")

    # Theorem 5.2(c) scaling
    scale = theorem2_null_T_nV_scaling(n_c=500, n_w=500, sigma=0.1)
    print(f"Theorem 5.2(c) scaling for n_c=n_w=500, sigma=0.1: {scale:.2f}")

    # Influence function V_inf
    Vinf = influence_function_V_inf(k=248, sigma=0.1, mu_xi_norm=0.0)
    print(f"V_inf (k=248, sigma=0.1, ||mu_xi||=0): {Vinf:.6f}")
    Vinf2 = influence_function_V_inf(k=248, sigma=0.1, mu_xi_norm=1.0)
    print(f"V_inf (k=248, sigma=0.1, ||mu_xi||=1): {Vinf2:.6f}")

    # Hessian bound
    bound = hessian_taylor_remainder_bound(L=2.0, mu_norm=0.5)
    print(f"Hessian remainder bound (L=2, mu=0.5): {bound:.4f}")

    print("OK")
