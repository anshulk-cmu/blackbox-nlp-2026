"""
Synthetic world for the toy experiment.

Constructs:
- A 9-parameter helix in R^{d_m} matching KT (1 linear + 4 cos + 4 sin, periods 2,5,10,100).
- All 10,000 ordered pairs (a,b) in {0,...,99}^2 mapped to activations h(a,b) on the
  answer-helix M_C = {helix(a+b) : a+b in {0,...,198}} plus isotropic noise.
- A correct/wrong split: most samples are correct (h on the helix); a chosen subset is
  "wrong" by adding a perturbation xi in a planted subspace V_true.
- A synthetic linear readout W: R^{d_m} -> R^{199} that argmax-recovers the correct
  answer on un-perturbed activations.

Everything is parameterised by a single SEED so figures are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


SEED = 20260504
PERIODS = (2, 5, 10, 100)
# At integer s, sin(2*pi*s/2) = sin(pi*s) = 0 identically, so the sin term at T=2
# carries no information and the effective basis is 8-dimensional, not 9. KT calls
# this the "T=2 fragility" (their Figure 12). We drop sin(pi*s) from the basis
# and use an 8-dim helix span throughout the toy.
HELIX_DIM = 8   # 1 linear + 1 cos(pi*s) + 3 cos/sin pairs for T in {5,10,100}
ANSWER_MAX = 198   # max value of a+b for a,b in {0,...,99}
OPERAND_MAX = 99


def basis_vector(s: int | np.ndarray) -> np.ndarray:
    """B(s) = [(s-99)/99, cos(pi*s), cos(2*pi*s/5), sin(2*pi*s/5), ..., cos(2*pi*s/100), sin(2*pi*s/100)].

    Linear axis rescaled to [-1, 1]. Note: at T=2 only the cos column is included
    because sin(pi*s) is identically zero on integers (KT T=2 fragility).

    Accepts a scalar or a 1D array. Returns shape (HELIX_DIM,) or (n, HELIX_DIM).
    """
    s = np.asarray(s, dtype=float)
    s_norm = (s - 99.0) / 99.0
    out = [s_norm]
    for T in PERIODS:
        theta = 2.0 * np.pi * s / T
        out.append(np.cos(theta))
        if T != 2:
            out.append(np.sin(theta))
    return np.stack(out, axis=-1)


def make_helix_directions(d_m: int, rng: np.random.Generator) -> np.ndarray:
    """Sample C in R^{d_m x 9} with orthonormal columns: the 9 helix directions.

    Returns: C of shape (d_m, 9), C.T @ C = I_9.
    """
    A = rng.standard_normal((d_m, HELIX_DIM))
    C, _ = np.linalg.qr(A)
    return C  # (d_m, 9)


def helix(s: int | np.ndarray, C: np.ndarray) -> np.ndarray:
    """helix(s) = C @ B(s).

    Returns shape (d_m,) for scalar s, (n, d_m) for vector s.
    Uses centred-and-rescaled linear axis so that |helix(s)| stays O(1)
    across s in [0, 198]: linear axis is s -> (s - 99) / 99, in [-1, 1].
    """
    B_s = basis_vector(s)          # (..., 9), with the rescaled linear axis
    return B_s @ C.T               # (..., d_m)


@dataclass
class World:
    """Container for the synthetic world's tensors."""

    d_m: int
    n_pairs: int             # number of (a,b) pairs used (10000 or 1000)
    sigma: float             # noise std
    seed: int
    C: np.ndarray            # (d_m, 9): helix directions, orthonormal columns
    a: np.ndarray            # (n_pairs,): operand a
    b: np.ndarray            # (n_pairs,): operand b
    s: np.ndarray            # (n_pairs,): a + b
    h_clean: np.ndarray      # (n_pairs, d_m): noise-free helix(a+b)
    h: np.ndarray            # (n_pairs, d_m): h_clean + noise
    is_wrong: np.ndarray     # (n_pairs,) bool: which samples are "wrong"
    xi: np.ndarray           # (n_pairs, d_m): perturbation applied (zero on correct samples)
    V_true: Optional[np.ndarray] = None   # (d_m, r): the planted perturbation subspace, or None
    W_readout: Optional[np.ndarray] = None  # (199, d_m): linear readout

    @property
    def correct_idx(self) -> np.ndarray:
        return np.flatnonzero(~self.is_wrong)

    @property
    def wrong_idx(self) -> np.ndarray:
        return np.flatnonzero(self.is_wrong)

    @property
    def n_correct(self) -> int:
        return int((~self.is_wrong).sum())

    @property
    def n_wrong(self) -> int:
        return int(self.is_wrong.sum())


def carry_indicator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 if (a%10) + (b%10) >= 10 (ones-place carry)."""
    return ((a % 10) + (b % 10)) >= 10


def random_unit_normal(d_m: int, M_basis: np.ndarray, r: int, rng: np.random.Generator) -> np.ndarray:
    """Return r orthonormal vectors in R^{d_m} that are perpendicular to span(M_basis).

    M_basis: (d_m, k) with orthonormal columns.
    Returns (d_m, r).
    """
    k = M_basis.shape[1]
    # Sample r random vectors, project off M_basis, orthonormalise.
    raw = rng.standard_normal((d_m, r))
    raw = raw - M_basis @ (M_basis.T @ raw)
    Q, _ = np.linalg.qr(raw)
    # QR may return d_m x r; columns are orthonormal.
    # Sanity check that they are still in M_basis^perp (small numerical drift OK).
    assert Q.shape == (d_m, r), f"unexpected QR shape {Q.shape}"
    return Q


def make_world(
    d_m: int = 256,
    n_pairs: int = 10_000,
    sigma: float = 0.1,
    wrong_frac: float = 0.10,
    perturbation_dim: int = 1,
    perturbation_norm: float = 1.0,
    perturbation_kind: str = "off_span",
    confound_carry: bool = False,
    carry_extra_noise: float = 0.0,
    anisotropic_lowrank: int = 0,
    anisotropic_strength: float = 1.0,
    seed: int = SEED,
    build_readout: bool = True,
) -> World:
    """Construct a synthetic world.

    Parameters
    ----------
    d_m : ambient dimension.
    n_pairs : number of (a,b) pairs sampled. If n_pairs >= 10000, use all 10000 ordered pairs;
              else sample uniformly.
    sigma : noise std (isotropic Gaussian).
    wrong_frac : fraction of samples that are "wrong" (perturbed).
    perturbation_dim : dim of V_true.
    perturbation_norm : ||mu_xi||_2; perturbation is mu_xi (no variance term, deterministic shift).
    perturbation_kind : "off_span" (perturb perpendicular to M_S),
                        "off_curve_in_span" (perturb in M_S but off the integer curve),
                        "on_curve" (perturb the integer index by +/-1, no off-span component),
                        "none" (mu_xi = 0, Sigma_xi = 0; null world),
                        "variance_only" (mu_xi = 0, Sigma_xi = (perturbation_norm**2 / r) * P_V).
    confound_carry : if True, "wrong" labels concentrate on samples with a ones-place carry,
                     so the wrong subset has a different marginal distribution over (a,b)
                     than the correct subset. Used for E6 (matched-permutation safeguard).
    anisotropic_lowrank : if > 0, replace isotropic noise with `Σ_ε = σ² I + (anisotropic_strength)² · UU^T`
                          where U has `anisotropic_lowrank` random orthonormal columns drawn off the helix
                          span. This is the regime where the effective rank
                          `k_eff = tr(P^N Σ_ε)² / tr((P^N Σ_ε)²)` is much smaller than
                          `k = d_m - dim(M)`. Used for E15 (anisotropic effective-rank test).
    anisotropic_strength : per-direction extra std along the anisotropy axes. Effective when
                          `anisotropic_lowrank > 0`.
    seed : RNG seed.
    build_readout : if True, construct W_readout.
    """
    rng = np.random.default_rng(seed)

    # 1. Helix directions (orthonormal columns of C)
    C = make_helix_directions(d_m, rng)          # (d_m, 9)

    # 2. Sample (a, b) pairs
    if n_pairs >= 10_000:
        a_grid, b_grid = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
        a = a_grid.ravel().astype(int)
        b = b_grid.ravel().astype(int)
        n_pairs = a.size
    else:
        a = rng.integers(0, 100, size=n_pairs)
        b = rng.integers(0, 100, size=n_pairs)
    s = a + b

    # 3. Clean activations on the answer helix
    h_clean = helix(s, C)                        # (n_pairs, d_m)

    # 4. Choose "wrong" samples
    if confound_carry:
        # Wrong samples concentrate on carry positions: prob(wrong) = base * 4 if carry, base if not.
        # base is set so overall wrong_frac matches.
        carry = carry_indicator(a, b)
        # wrong_frac = base * (4*p_carry + 1*(1-p_carry))
        p_carry = float(carry.mean())
        base = wrong_frac / (4.0 * p_carry + (1.0 - p_carry))
        prob = np.where(carry, 4.0 * base, base)
        prob = np.clip(prob, 0.0, 1.0)
        is_wrong = rng.uniform(size=n_pairs) < prob
    else:
        is_wrong = rng.uniform(size=n_pairs) < wrong_frac

    # 5. Build the planted perturbation subspace V_true
    V_true: Optional[np.ndarray] = None
    if perturbation_kind == "off_span":
        V_true = random_unit_normal(d_m, C, perturbation_dim, rng)
    elif perturbation_kind == "off_curve_in_span":
        # V_true is inside M_S (so projection onto M_S leaves it; T_span = 0).
        # Pick perturbation_dim columns from C as V_true.
        idx = rng.choice(HELIX_DIM, size=perturbation_dim, replace=False)
        V_true = C[:, idx]
    elif perturbation_kind in ("on_curve", "none", "variance_only"):
        V_true = random_unit_normal(d_m, C, perturbation_dim, rng)  # placeholder direction
    else:
        raise ValueError(f"unknown perturbation_kind {perturbation_kind!r}")

    # 6. Build the perturbation vector xi for each sample (zero on correct samples)
    xi = np.zeros((n_pairs, d_m))
    if perturbation_kind == "none":
        pass  # xi stays zero
    elif perturbation_kind == "variance_only":
        # mu_xi = 0; xi(a) ~ N(0, (perturbation_norm**2 / r) * P_V) on wrong samples.
        per_dim_std = perturbation_norm / np.sqrt(perturbation_dim)
        coeffs = rng.standard_normal((n_pairs, perturbation_dim)) * per_dim_std
        xi_full = coeffs @ V_true.T   # (n_pairs, d_m)
        xi[is_wrong] = xi_full[is_wrong]
    elif perturbation_kind == "on_curve":
        # Wrong sample = helix(s + delta) - helix(s). No off-span component.
        delta = rng.choice([-2, -1, 1, 2], size=n_pairs)
        s_shifted = np.clip(s + delta, 0, ANSWER_MAX)
        h_shifted = helix(s_shifted, C)
        xi_full = h_shifted - h_clean
        xi[is_wrong] = xi_full[is_wrong]
    elif perturbation_kind == "off_curve_in_span":
        # Wrong sample = helix(s + alpha) for alpha in (0,1), random per sample.
        # This stays in span(M_S basis) only approximately because the basis is helix-evaluated;
        # but B(s + alpha) is in the column space of C as long as we evaluate helix() on a
        # non-integer s. So xi = helix(s + alpha) - helix(s).
        alpha = rng.uniform(0.3, 0.7, size=n_pairs)
        h_shifted = helix(s + alpha, C)
        xi_full = h_shifted - h_clean
        xi[is_wrong] = xi_full[is_wrong]
    elif perturbation_kind == "off_span":
        # Deterministic shift mu_xi = perturbation_norm * (V_true @ unit_coefs)
        # where unit_coefs is fixed (so all wrong samples get the same shift).
        coefs = rng.standard_normal(perturbation_dim)
        coefs /= np.linalg.norm(coefs)
        mu = V_true @ coefs * perturbation_norm    # (d_m,)
        xi[is_wrong] = mu

    # 7. Final activations.
    # Optionally add carry-correlated extra noise: ALL carry samples (regardless of
    # correct/wrong label) get a wider noise distribution. This creates a difficulty
    # confound (carry samples have larger residual norm) without making
    # within-bin distributions differ between correct and wrong populations -- so
    # matched permutation can still control Type I rate while naive permutation
    # rejects spuriously.
    noise = rng.standard_normal((n_pairs, d_m)) * sigma
    if carry_extra_noise > 0.0:
        carry = carry_indicator(a, b)
        boost = np.where(carry, carry_extra_noise, 0.0)[:, None]   # (n_pairs, 1)
        extra = rng.standard_normal((n_pairs, d_m)) * boost
        noise = noise + extra
    if anisotropic_lowrank > 0:
        # Add a low-rank covariance term off the helix span. Σ_ε = σ²I + s² U U^T,
        # so noise = σε_iso + s · U · ε_lr, where ε_lr ~ N(0, I_lowrank).
        U_aniso = random_unit_normal(d_m, C, anisotropic_lowrank, rng)  # (d_m, lowrank)
        eps_lr = rng.standard_normal((n_pairs, anisotropic_lowrank))    # (n, lowrank)
        anisotropic_extra = (anisotropic_strength * eps_lr) @ U_aniso.T  # (n, d_m)
        noise = noise + anisotropic_extra
    h = h_clean + xi + noise

    # 8. Build readout W if requested.
    # Real LM readouts (lm_head) read both helix-aligned and off-helix components.
    # The toy readout has three terms:
    #   W_helix[s]   = helix(s) / ||helix(s)||^2   (cosine head, decodes correctly on M_S)
    #   W_offspan[s] = ramp_amplitude * (s - 99) / 99 * v_flip   (a "shift" direction
    #                  that mis-classifies in V_true: injecting +v_flip into a correct
    #                  helix(s) lifts the s+Delta_s logit above s. Used for E8 ACE_S.)
    #   W_noise[s]   = small_amplitude * random_offspan       (rich gradients in all
    #                  off-span directions so ACE_R isn't pathologically zero)
    # ramp_amplitude is calibrated so injecting a unit perturbation along v_flip into
    # h_clean = helix(s) shifts argmax by ~1 step. Concretely, between adjacent s, the
    # helix-similarity logit gap is O(1), so we set ramp_amplitude = 5.0 -- with a
    # perturbation of magnitude ~1 along v_flip, the s+1 logit beats the s logit.
    # v_flip = V_true's first column (1D V_true means v_flip = V_true).
    W_readout = None
    if build_readout and V_true is not None:
        all_s = np.arange(ANSWER_MAX + 1)
        H_all = helix(all_s, C)                # (199, d_m)
        norms = np.einsum("ij,ij->i", H_all, H_all)   # (199,)
        W_helix = H_all / norms[:, None]       # (199, d_m), aligned with helix
        v_flip = V_true[:, 0]                  # (d_m,) -- first column of V_true
        # Linear ramp in s, along v_flip. Amplitude must be small enough that
        # sigma-scale noise along v_flip doesn't flip argmax (so the un-perturbed
        # accuracy stays ~100%) but large enough that perturbation-scale shifts do
        # flip it. With ramp_amplitude = 1.0: max noise contribution is sigma*1.0
        # per logit (~0.1 << helix gap ~1.0); perturbation contribution is
        # perturbation_norm * 1.0 (~1-2 >> helix gap). Sweet spot.
        ramp_amplitude = 1.0
        ramp = ramp_amplitude * (all_s - 99) / 99.0   # (199,)
        W_ramp = ramp[:, None] * v_flip[None, :]      # (199, d_m)
        # Small random off-span term (non-zero gradient in every off-span direction)
        W_noise = rng.standard_normal((ANSWER_MAX + 1, d_m))
        W_noise = W_noise - W_noise @ C @ C.T          # project out in-span
        W_noise = 0.05 * W_noise / np.linalg.norm(W_noise, axis=1, keepdims=True)
        W_readout = W_helix + W_ramp + W_noise
    elif build_readout:
        # No V_true (e.g., perturbation_kind='none'); just build helix + small noise.
        all_s = np.arange(ANSWER_MAX + 1)
        H_all = helix(all_s, C)
        norms = np.einsum("ij,ij->i", H_all, H_all)
        W_helix = H_all / norms[:, None]
        W_noise = rng.standard_normal((ANSWER_MAX + 1, d_m))
        W_noise = W_noise - W_noise @ C @ C.T
        W_noise = 0.05 * W_noise / np.linalg.norm(W_noise, axis=1, keepdims=True)
        W_readout = W_helix + W_noise

    return World(
        d_m=d_m,
        n_pairs=n_pairs,
        sigma=sigma,
        seed=seed,
        C=C,
        a=a, b=b, s=s,
        h_clean=h_clean,
        h=h,
        is_wrong=is_wrong,
        xi=xi,
        V_true=V_true,
        W_readout=W_readout,
    )


def logit_difference(h: np.ndarray, target: np.ndarray | int, W: np.ndarray) -> np.ndarray:
    """LD(h) = logit_target(h) - max_{t != target} logit_t(h).

    h: (n, d_m) or (d_m,)
    target: int or (n,)
    W: (n_classes, d_m)

    Returns (n,) or scalar.
    """
    h = np.atleast_2d(h)
    logits = h @ W.T               # (n, n_classes)
    if np.isscalar(target):
        target_arr = np.full(h.shape[0], int(target))
    else:
        target_arr = np.asarray(target).astype(int)
    n = h.shape[0]
    target_logit = logits[np.arange(n), target_arr]
    masked = logits.copy()
    masked[np.arange(n), target_arr] = -np.inf
    other_max = masked.max(axis=1)
    return target_logit - other_max


def fraction_correct(world: World) -> float:
    """argmax of W @ h compared to s, on samples where xi is zero."""
    if world.W_readout is None:
        raise ValueError("world has no readout")
    mask = ~world.is_wrong
    H = world.h[mask]
    pred = (H @ world.W_readout.T).argmax(axis=1)
    return float((pred == world.s[mask]).mean())


if __name__ == "__main__":
    # Smoke test.
    w = make_world(d_m=64, n_pairs=200, sigma=0.05, wrong_frac=0.1,
                   perturbation_kind="off_span", perturbation_norm=0.5)
    print(f"World: d_m={w.d_m}, n={w.n_pairs}, n_wrong={w.n_wrong}, sigma={w.sigma}")
    print(f"  C shape={w.C.shape}, orthonormality residual={np.linalg.norm(w.C.T @ w.C - np.eye(HELIX_DIM)):.2e}")
    print(f"  V_true shape={w.V_true.shape}")
    print(f"  W_readout shape={w.W_readout.shape}")
    print(f"  fraction correct on un-perturbed samples: {fraction_correct(w):.3f}")
    print("OK")
