"""
Test statistics and inference machinery (paper sections 1.5, 2.1, 2.3, 3.3, 3.4, 3.7, 3.8).

T_n        : off-manifold residual difference (Theorem 1).
T_n_V      : localized residual on subspace V (Theorem 2).
T_curve    : residual against the discrete integer-helix curve M_C.
T_span     : residual against the helix span M_S.
Cross-fit  : K-fold cross-fitting protocol for unbiased correct-residual baseline (paper section 3.4).
Matched permutation: within-bin label shuffling under a confound (paper section 3.3).
BH-FDR     : Benjamini-Hochberg correction across multiple V's.
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Projections and residuals
# ---------------------------------------------------------------------------

def project_subspace(h: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Orthogonal projection of h onto span(U).

    h: (n, d_m) or (d_m,)
    U: (d_m, k) with orthonormal columns (we don't re-orthonormalise here).
    Returns: same shape as h.
    """
    return h @ U @ U.T


def residual_subspace(h: np.ndarray, U: np.ndarray) -> np.ndarray:
    """h - P_U h."""
    return h - project_subspace(h, U)


def project_curve(h: np.ndarray, curve_points: np.ndarray) -> np.ndarray:
    """Project each row of h onto the closest point in curve_points.

    h: (n, d_m), curve_points: (n_curve, d_m). Returns (n, d_m).
    """
    # ||h_i - c_j||^2 for all (i, j): use (a-b)^2 = a^2 - 2ab + b^2 expansion.
    h2 = (h * h).sum(axis=1, keepdims=True)             # (n, 1)
    c2 = (curve_points * curve_points).sum(axis=1)[None, :]   # (1, n_curve)
    cross = h @ curve_points.T                          # (n, n_curve)
    dists2 = h2 - 2.0 * cross + c2
    nearest = dists2.argmin(axis=1)
    return curve_points[nearest]


def residual_curve(h: np.ndarray, curve_points: np.ndarray) -> np.ndarray:
    return h - project_curve(h, curve_points)


# ---------------------------------------------------------------------------
# T_n statistics
# ---------------------------------------------------------------------------

def T_n_subspace(H_c: np.ndarray, H_w: np.ndarray, U: np.ndarray) -> float:
    """T_n with span-projection residuals (Theorem 1, T_span).

    T_n = mean_w ||r(h_w)||^2 - mean_c ||r(h_c)||^2.
    """
    r_c = residual_subspace(H_c, U)
    r_w = residual_subspace(H_w, U)
    return float((r_w * r_w).sum(axis=1).mean() - (r_c * r_c).sum(axis=1).mean())


def T_n_curve(H_c: np.ndarray, H_w: np.ndarray, curve_points: np.ndarray) -> float:
    r_c = residual_curve(H_c, curve_points)
    r_w = residual_curve(H_w, curve_points)
    return float((r_w * r_w).sum(axis=1).mean() - (r_c * r_c).sum(axis=1).mean())


def T_n_V(H_c: np.ndarray, H_w: np.ndarray, U_M: np.ndarray, V: np.ndarray) -> float:
    """Localized statistic T_n^V (Theorem 2).

    Compute residuals against M_hat (span of U_M), then project onto V before
    taking squared norm. V should already be approximately orthogonal to span(U_M).
    """
    r_c = residual_subspace(H_c, U_M)
    r_w = residual_subspace(H_w, U_M)
    rV_c = r_c @ V
    rV_w = r_w @ V
    return float((rV_w * rV_w).sum(axis=1).mean() - (rV_c * rV_c).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Cross-fitting (paper section 3.4)
# ---------------------------------------------------------------------------

def cross_fit_T_n(
    H_c: np.ndarray,
    H_w: np.ndarray,
    s_c: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    K: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """K-fold cross-fitted T_n.

    fit_fn : (H_c_train, s_c_train) -> U_hat with shape (d_m, k).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_c = H_c.shape[0]
    perm = rng.permutation(n_c)
    folds = np.array_split(perm, K)

    correct_terms = []
    wrong_residuals = []  # average ||r||^2 across K M_hats for each wrong sample
    n_w = H_w.shape[0]
    wrong_acc = np.zeros(n_w)

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        U_k = fit_fn(H_c[train_idx], s_c[train_idx])
        # Correct fold: residual on held-out.
        r_test = residual_subspace(H_c[test_idx], U_k)
        correct_terms.append((r_test * r_test).sum(axis=1).mean())
        # Wrong samples: residual under U_k, accumulated.
        r_w = residual_subspace(H_w, U_k)
        wrong_acc += (r_w * r_w).sum(axis=1) / K

    correct_mean = float(np.mean(correct_terms))
    wrong_mean = float(wrong_acc.mean())
    return wrong_mean - correct_mean


def naive_T_n(
    H_c: np.ndarray,
    H_w: np.ndarray,
    s_c: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    """T_n with single-fit overfit baseline (no cross-fitting). Biased."""
    U = fit_fn(H_c, s_c)
    return T_n_subspace(H_c, H_w, U)


# ---------------------------------------------------------------------------
# Permutation calibration (Theorem 1 / 2 null distribution)
# ---------------------------------------------------------------------------

def permutation_pvalue(
    H_all: np.ndarray,
    is_wrong: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Two-sample permutation test for any statistic.

    statistic_fn(H_c, H_w) returns the observed value.

    Returns (observed, p_value).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    H_c = H_all[~is_wrong]
    H_w = H_all[is_wrong]
    observed = statistic_fn(H_c, H_w)

    n = H_all.shape[0]
    n_w = int(is_wrong.sum())
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        wrong_perm_idx = perm[:n_w]
        correct_perm_idx = perm[n_w:]
        null_stats[i] = statistic_fn(H_all[correct_perm_idx], H_all[wrong_perm_idx])

    pvalue = float((null_stats >= observed).mean())
    return observed, pvalue


# ---------------------------------------------------------------------------
# Matched permutation (paper section 3.3)
# ---------------------------------------------------------------------------

def make_phi_bins(
    a: np.ndarray, b: np.ndarray,
    sum_edges: tuple = (0, 50, 100, 150, 199),
    decile_edges: tuple = tuple(range(0, 110, 10)),
) -> np.ndarray:
    """Compute coarse phi-bin index for each sample.

    Bins: (sum_bin, carry_pattern, a_decile, b_decile).
    sum_bin: 4 bins via sum_edges.
    carry_pattern: 4 bins (none, ones, tens, both).
    a_decile, b_decile: 10 bins each via decile_edges.

    Returns (n,) integer bin index.
    """
    s = a + b
    sum_bin = np.searchsorted(sum_edges, s, side="right") - 1
    sum_bin = np.clip(sum_bin, 0, len(sum_edges) - 2)

    ones_carry = ((a % 10) + (b % 10) >= 10).astype(int)
    tens_carry = (s >= 100).astype(int)
    carry_pattern = ones_carry * 2 + tens_carry  # 0..3

    a_dec = np.searchsorted(decile_edges, a, side="right") - 1
    b_dec = np.searchsorted(decile_edges, b, side="right") - 1
    a_dec = np.clip(a_dec, 0, 9)
    b_dec = np.clip(b_dec, 0, 9)

    n_sum = len(sum_edges) - 1
    bin_idx = ((sum_bin * 4 + carry_pattern) * 10 + a_dec) * 10 + b_dec
    return bin_idx


def matched_permutation_test(
    H_all: np.ndarray,
    is_wrong: np.ndarray,
    bin_idx: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 1000,
    min_per_bin: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, dict]:
    """Within-bin label permutation (paper section 3.3).

    Returns (observed, pvalue, info). info contains diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Identify bins with enough correct AND wrong samples
    unique_bins, inv = np.unique(bin_idx, return_inverse=True)
    keep_mask = np.zeros(H_all.shape[0], dtype=bool)
    info_bins = []
    for bi in range(len(unique_bins)):
        bm = inv == bi
        n_c_bin = int((bm & ~is_wrong).sum())
        n_w_bin = int((bm & is_wrong).sum())
        if n_c_bin >= min_per_bin and n_w_bin >= min_per_bin:
            keep_mask |= bm
            info_bins.append((unique_bins[bi], n_c_bin, n_w_bin))

    if not keep_mask.any():
        return float("nan"), float("nan"), {"retained_bins": 0, "n_kept": 0}

    H_kept = H_all[keep_mask]
    is_wrong_kept = is_wrong[keep_mask]
    bin_kept = bin_idx[keep_mask]

    # Observed stat on kept samples
    H_c = H_kept[~is_wrong_kept]
    H_w = H_kept[is_wrong_kept]
    observed = statistic_fn(H_c, H_w)

    # Within-bin permutation
    null_stats = np.empty(n_permutations)
    bin_groups = {b: np.flatnonzero(bin_kept == b) for b in np.unique(bin_kept)}
    for i in range(n_permutations):
        labels = is_wrong_kept.copy()
        for b, grp in bin_groups.items():
            shuffled = rng.permutation(grp)
            labels[grp] = is_wrong_kept[shuffled]
        H_c_perm = H_kept[~labels]
        H_w_perm = H_kept[labels]
        null_stats[i] = statistic_fn(H_c_perm, H_w_perm)

    pvalue = float((null_stats >= observed).mean())
    info = {
        "retained_bins": len(info_bins),
        "n_kept": int(keep_mask.sum()),
        "n_kept_correct": int((~is_wrong_kept).sum()),
        "n_kept_wrong": int(is_wrong_kept.sum()),
    }
    return observed, pvalue, info


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------

def bh_fdr(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR control.

    Returns boolean array, True at indices where the corresponding p-value
    is significant after BH correction at level q.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(m, dtype=bool)
    # Largest k with ranked[k] <= thresholds[k]
    k_max = int(np.flatnonzero(passed).max())
    significant = np.zeros(m, dtype=bool)
    significant[order[: k_max + 1]] = True
    return significant


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    statistic_fn: Callable[[], float],
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Helper that returns (lo, hi) percentile CI from a bootstrap resampler.

    Caller is expected to pass a thunk that *resamples the data* and computes
    the statistic. We don't manage the resampling here; this is just the
    percentile assembly.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    samples = np.array([statistic_fn() for _ in range(n_bootstrap)])
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


if __name__ == "__main__":
    # Smoke test: identical populations should give T_n approx 0 with p approx 0.5
    from synth_world import make_world
    from manifold_methods import fit_parametric, HELIX_DIM

    w = make_world(d_m=64, n_pairs=2000, sigma=0.05, wrong_frac=0.1,
                   perturbation_kind="none", build_readout=False)
    H_all = w.h
    U = fit_parametric(w.h[~w.is_wrong], w.s[~w.is_wrong])

    def stat(Hc, Hw):
        return T_n_subspace(Hc, Hw, U)

    obs, p = permutation_pvalue(H_all, w.is_wrong, stat, n_permutations=200)
    print(f"Null world: T_n={obs:.4f}, perm p-value={p:.3f} (should be ~0.5)")

    # Now off-span world
    w2 = make_world(d_m=64, n_pairs=2000, sigma=0.05, wrong_frac=0.1,
                    perturbation_kind="off_span", perturbation_norm=0.5, build_readout=False)
    U2 = fit_parametric(w2.h[~w2.is_wrong], w2.s[~w2.is_wrong])
    obs2, p2 = permutation_pvalue(w2.h, w2.is_wrong, lambda c, w: T_n_subspace(c, w, U2),
                                   n_permutations=200)
    print(f"Off-span world: T_n={obs2:.4f}, expected ~={0.5**2 * w2.is_wrong.mean()*0 + 0.5**2:.4f}, perm p-value={p2:.3f}")

    # Cross-fit vs naive
    cf = cross_fit_T_n(w.h[~w.is_wrong], w.h[w.is_wrong], w.s[~w.is_wrong], fit_parametric)
    naive = naive_T_n(w.h[~w.is_wrong], w.h[w.is_wrong], w.s[~w.is_wrong], fit_parametric)
    print(f"Null world: naive T_n={naive:.4f} (biased upward), cross-fit T_n={cf:.4f}")

    # Phi bins
    bins = make_phi_bins(w.a, w.b)
    print(f"Phi-bins: {len(np.unique(bins))} unique bins (out of {4*4*10*10}=1600 possible)")

    # BH-FDR
    pv = np.array([0.001, 0.01, 0.04, 0.5, 0.9])
    sig = bh_fdr(pv, q=0.05)
    print(f"BH-FDR test: pvalues={pv} -> sig={sig}")

    print("OK")
