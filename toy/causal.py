"""
Four-intervention causal pipeline (paper section 3.9, paper_math.md section 9)
plus Proposition 9.3 (Hessian-bounded ACE_S, second-order Taylor).

Patch(h, V, alpha, delta) = h + alpha * U_V * delta - alpha * P_V * h
                          = h + alpha * (U_V * delta - P_V h)

Removal patch (alpha=1, delta=0):  h' = (I - P_V) h
Injection patch (alpha=1, delta=v):  h' = h + U_V v - P_V h = (I - P_V) h + U_V v

Interventions:
  N    (necessity):           Patch with V = M_hat^perp, alpha=1, delta=0
                              -- zero out the off-manifold residual on wrong samples.
  N-V  (localized necessity): Patch with V = subspace, alpha=1, delta=0
                              -- zero out V-component on wrong samples.
  S    (sufficiency):         Patch with V = subspace, alpha=1, delta=mu_xi_hat
                              -- inject mu_xi into correct samples.
  R    (specificity):         Same as N-V but with V = random subspace,
                              -- 100 replicates -> baseline distribution.

Proposition 9.3 (paper_math.md):
  ACE_S^linear  =  mu_xi_hat^T * U_V^T * E[grad_LD(h_c)]
  |ACE_S - ACE_S^linear|  <=  (L / 2) * ||mu_xi_hat||^2     [Hessian-bounded remainder]
  |ACE_S^linear|           >=  ||mu_xi_hat|| * ||U_V^T E[grad LD]|| * cos(theta_xi_grad)

For a linear readout W: R^{d_m} -> R^{n_classes}, LD(h) = W[target] @ h - max_other.
grad_h LD = W[target] - W[argmax_other].   (For correctly-classified samples,
argmax_other is well-defined.)

Smooth-max: argmax over t != target is non-smooth. We provide LD_tau, the
log-sum-exp temperature-tau approximation, with closed-form gradient and Hessian.
For tau -> 0, LD_tau -> LD; the Hessian operator-norm bound L scales as 1/tau
near argmax flips.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from synth_world import logit_difference


# ---------------------------------------------------------------------------
# Patch operation
# ---------------------------------------------------------------------------

def patch(h: np.ndarray, U_V: np.ndarray, alpha: float = 1.0, delta: Optional[np.ndarray] = None) -> np.ndarray:
    """Patch(h, V, alpha, delta) = h + alpha * (U_V @ delta - P_V h).

    h: (n, d_m) or (d_m,)
    U_V: (d_m, r) orthonormal columns
    alpha: scalar
    delta: (r,), or None (defaults to zeros).
    """
    h = np.atleast_2d(h)
    P_V_h = h @ U_V @ U_V.T   # (n, d_m)
    if delta is None:
        return h - alpha * P_V_h
    inj = (U_V @ np.asarray(delta).reshape(-1))[None, :]   # (1, d_m)
    return h + alpha * (inj - P_V_h)


# ---------------------------------------------------------------------------
# Logit difference and gradient
# ---------------------------------------------------------------------------

def grad_logit_diff_linear(h: np.ndarray, target: np.ndarray | int, W: np.ndarray) -> np.ndarray:
    """Analytic gradient of LD wrt h for a linear readout W.

    For each sample, grad_h LD = W[target_i] - W[argmax_{t != target_i} W[t] @ h_i].
    Returns (n, d_m).
    """
    h = np.atleast_2d(h)
    logits = h @ W.T   # (n, n_classes)
    if np.isscalar(target):
        target_arr = np.full(h.shape[0], int(target))
    else:
        target_arr = np.asarray(target).astype(int)
    n = h.shape[0]
    masked = logits.copy()
    masked[np.arange(n), target_arr] = -np.inf
    other_argmax = masked.argmax(axis=1)
    grad = W[target_arr] - W[other_argmax]   # (n, d_m)
    return grad


# ---------------------------------------------------------------------------
# Average causal effects
# ---------------------------------------------------------------------------

def ACE_N(world, U_M_hat: np.ndarray) -> tuple[float, float]:
    """Necessity: project wrong activations onto M_hat (zero out perp component).

    Returns (mean_LD_shift, std_LD_shift).
    """
    H_w = world.h[world.is_wrong]
    s_w = world.s[world.is_wrong]
    W = world.W_readout
    LD_before = logit_difference(H_w, s_w, W)
    # Zero out M_hat-perpendicular component: P_M h = h @ U @ U.T
    H_w_patched = H_w @ U_M_hat @ U_M_hat.T
    LD_after = logit_difference(H_w_patched, s_w, W)
    diff = LD_after - LD_before
    return float(diff.mean()), float(diff.std())


def ACE_NV(world, U_V: np.ndarray) -> tuple[float, float]:
    """Localized necessity: zero out V component of wrong activations."""
    H_w = world.h[world.is_wrong]
    s_w = world.s[world.is_wrong]
    W = world.W_readout
    LD_before = logit_difference(H_w, s_w, W)
    H_w_patched = patch(H_w, U_V, alpha=1.0, delta=None)
    LD_after = logit_difference(H_w_patched, s_w, W)
    diff = LD_after - LD_before
    return float(diff.mean()), float(diff.std())


def ACE_S(world, U_V: np.ndarray, mu_xi_hat: np.ndarray) -> tuple[float, float]:
    """Sufficiency: inject mu_xi_hat (in V-coordinates) into correct activations.

    mu_xi_hat: (r,) coordinate vector in V's basis (same r as U_V's columns).
    """
    H_c = world.h[~world.is_wrong]
    s_c = world.s[~world.is_wrong]
    W = world.W_readout
    LD_before = logit_difference(H_c, s_c, W)
    H_c_patched = patch(H_c, U_V, alpha=1.0, delta=mu_xi_hat)
    LD_after = logit_difference(H_c_patched, s_c, W)
    diff = LD_after - LD_before
    return float(diff.mean()), float(diff.std())


def ACE_R(world, M_hat: np.ndarray, r: int, n_replicates: int = 100,
         rng: Optional[np.random.Generator] = None) -> tuple[float, float]:
    """Specificity: random V_random of dim r perpendicular to M_hat,
    apply the N-V intervention, return distribution.

    Returns (mean across replicates, std across replicates).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_w = world.n_wrong
    H_w = world.h[world.is_wrong]
    s_w = world.s[world.is_wrong]
    W = world.W_readout
    LD_before = logit_difference(H_w, s_w, W)
    d_m = M_hat.shape[0]
    diffs = np.empty(n_replicates)
    for i in range(n_replicates):
        # Random V_rand of dim r, orthogonal to M_hat
        raw = rng.standard_normal((d_m, r))
        raw = raw - M_hat @ (M_hat.T @ raw)
        Q, _ = np.linalg.qr(raw)
        V_rand = Q
        H_w_patched = patch(H_w, V_rand, alpha=1.0, delta=None)
        LD_after = logit_difference(H_w_patched, s_w, W)
        diffs[i] = float((LD_after - LD_before).mean())
    return float(diffs.mean()), float(diffs.std())


# ---------------------------------------------------------------------------
# Proposition 4 prediction (analytic)
# ---------------------------------------------------------------------------

def predicted_ACE_S_linear(world, U_V: np.ndarray, mu_xi_hat: np.ndarray) -> float:
    """Signed first-order Taylor prediction: mean over correct samples of grad_LD @ (U_V @ mu_xi).

    For a SMALL perturbation (no argmax flipping) this matches actual ACE_S exactly.
    For a LARGE perturbation, max-based LD is non-smooth and this prediction can be
    near zero while |actual ACE_S| is large -- the first-order Taylor breaks down.
    Paper's Proposition 4 states a magnitude lower bound; see predicted_ACE_S_magnitude.
    """
    H_c = world.h[~world.is_wrong]
    s_c = world.s[~world.is_wrong]
    W = world.W_readout
    grad = grad_logit_diff_linear(H_c, s_c, W)   # (n_c, d_m)
    proj = grad @ U_V @ np.asarray(mu_xi_hat).reshape(-1)   # (n_c,)
    return float(proj.mean())


# ---------------------------------------------------------------------------
# Smooth-max logit difference LD_tau (paper_math.md Remark 9.4)
# ---------------------------------------------------------------------------

def logit_difference_smooth(h: np.ndarray, target: np.ndarray | int,
                              W: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """LD_tau(h) = logit_target(h) - tau * log(sum_{t != target} exp(logit_t(h) / tau)).

    For tau -> 0, LD_tau -> LD; for tau -> infty, LD_tau -> logit_target - mean_other - tau*log(n_classes-1).

    Returns (n,) for h: (n, d) and target: int or (n,).
    """
    h = np.atleast_2d(h)
    logits = h @ W.T   # (n, n_classes)
    if np.isscalar(target):
        target_arr = np.full(h.shape[0], int(target))
    else:
        target_arr = np.asarray(target).astype(int)
    n = h.shape[0]
    target_logit = logits[np.arange(n), target_arr]
    masked = logits.copy()
    masked[np.arange(n), target_arr] = -np.inf
    # log-sum-exp over t != target, with temperature tau:
    #   tau * logsumexp(logits / tau)
    scaled = masked / tau
    m = scaled.max(axis=1, keepdims=True)            # (n, 1) numerical-stability shift
    # m may contain -inf if every entry is -inf, but masked has only one -inf per row.
    log_sum = m.squeeze(1) + np.log(np.exp(scaled - m).sum(axis=1))
    return target_logit - tau * log_sum


def grad_LD_smooth_linear(h: np.ndarray, target: np.ndarray | int,
                            W: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Gradient of LD_tau wrt h, closed form for a linear readout W.

    LD_tau(h) = W[target] @ h - tau * log(sum_{t != target} exp(W[t] @ h / tau)).
    grad_h LD_tau = W[target] - sum_{t != target} softmax(W[t] @ h / tau)_t * W[t].

    Returns (n, d_m).
    """
    h = np.atleast_2d(h)
    logits = h @ W.T
    if np.isscalar(target):
        target_arr = np.full(h.shape[0], int(target))
    else:
        target_arr = np.asarray(target).astype(int)
    n = h.shape[0]
    masked = logits.copy()
    masked[np.arange(n), target_arr] = -np.inf
    scaled = masked / tau
    m = scaled.max(axis=1, keepdims=True)
    exps = np.exp(scaled - m)
    weights = exps / exps.sum(axis=1, keepdims=True)   # (n, n_classes), softmax
    # grad_h LD_tau = W[target_i] - sum_t weights[i, t] * W[t]
    weighted_W = weights @ W                            # (n, d_m)
    return W[target_arr] - weighted_W


def hessian_opnorm_smooth_LD_linear(h: np.ndarray, target: np.ndarray | int,
                                       W: np.ndarray, tau: float) -> np.ndarray:
    """Operator-norm of the Hessian of LD_tau wrt h, for a linear readout.

    H = -(1/tau) * (W^T diag(p) W - W^T p p^T W),
    where p is the softmax over t != target. ||H|| <= (1/tau) * ||W diag(sqrt(p))||_op^2
    by a standard convexity bound; the simpler bound used here is ||W||_op^2 / tau,
    which is data-independent. Returns a scalar (per-sample).

    For the toy this is tighter: we compute the actual ||H||_op per sample.
    """
    h = np.atleast_2d(h)
    if np.isscalar(target):
        target_arr = np.full(h.shape[0], int(target))
    else:
        target_arr = np.asarray(target).astype(int)
    logits = h @ W.T
    n = h.shape[0]
    masked = logits.copy()
    masked[np.arange(n), target_arr] = -np.inf
    scaled = masked / tau
    m = scaled.max(axis=1, keepdims=True)
    exps = np.exp(scaled - m)
    p = exps / exps.sum(axis=1, keepdims=True)   # (n, n_classes)
    # Per-sample Hessian: H_i = -(1/tau) (W^T diag(p_i) W - W^T p_i p_i^T W)
    # Compute its operator norm by an upper bound: ||H||_op <= (1/tau) * largest sing value of W
    # (the ||W||_op bound). This is a uniform overestimate that works for the toy.
    op_W = float(np.linalg.svd(W, compute_uv=False).max())
    return float(op_W ** 2 / tau)


# ---------------------------------------------------------------------------
# ACE_S using smooth-max LD_tau (Proposition 9.3 testable form)
# ---------------------------------------------------------------------------

def ACE_S_smooth(world, U_V: np.ndarray, mu_xi_hat: np.ndarray,
                   tau: float = 1.0) -> tuple[float, float]:
    """ACE_S using LD_tau (smooth-max) instead of LD.

    Returns (mean LD_tau shift, std). For tau -> 0 this approaches the
    standard ACE_S; for moderate tau it's smooth and Proposition 9.3's
    Hessian remainder bound is well-defined.
    """
    H_c = world.h[~world.is_wrong]
    s_c = world.s[~world.is_wrong]
    W = world.W_readout
    LD_before = logit_difference_smooth(H_c, s_c, W, tau=tau)
    H_c_patched = patch(H_c, U_V, alpha=1.0, delta=mu_xi_hat)
    LD_after = logit_difference_smooth(H_c_patched, s_c, W, tau=tau)
    diff = LD_after - LD_before
    return float(diff.mean()), float(diff.std())


def predicted_ACE_S_linear_smooth(world, U_V: np.ndarray, mu_xi_hat: np.ndarray,
                                     tau: float = 1.0) -> float:
    """ACE_S^linear using grad_LD_tau."""
    H_c = world.h[~world.is_wrong]
    s_c = world.s[~world.is_wrong]
    W = world.W_readout
    grad = grad_LD_smooth_linear(H_c, s_c, W, tau=tau)   # (n_c, d_m)
    proj = grad @ U_V @ np.asarray(mu_xi_hat).reshape(-1)
    return float(proj.mean())


def predicted_ACE_S_magnitude(world, U_V: np.ndarray, mu_xi_hat: np.ndarray) -> float:
    """Magnitude lower bound from Proposition 4 (paper section 3.9.5):

       |ACE_S| >= ||mu_xi_hat|| * mean_c ||U_V^T grad_LD(h_c)||  -  O(sigma)

    This is the Cauchy-Schwarz bound on |<grad, perturbation>| averaged over
    correct samples. It is a lower bound on |actual ACE_S| in the linear regime
    and (empirically) an even more conservative lower bound when the
    perturbation is large enough to flip argmax.
    """
    H_c = world.h[~world.is_wrong]
    s_c = world.s[~world.is_wrong]
    W = world.W_readout
    grad = grad_logit_diff_linear(H_c, s_c, W)            # (n_c, d_m)
    grad_proj_V = grad @ U_V                              # (n_c, r)
    norms = np.linalg.norm(grad_proj_V, axis=1)           # (n_c,)
    mu_norm = float(np.linalg.norm(mu_xi_hat))
    return float(mu_norm * norms.mean())


# ---------------------------------------------------------------------------
# Torch-autograd cross-check (validates the same code path real models will use)
# ---------------------------------------------------------------------------

def predicted_ACE_S_autograd(world, U_V: np.ndarray, mu_xi_hat: np.ndarray) -> float:
    """Same prediction but computed via torch.autograd.

    Even though the readout is linear and we have the closed-form, we go through
    autograd here so the toy exercises the same code path that the real-model
    causal pipeline will use (where W is the lm_head and gradients can't be
    computed analytically).
    """
    import torch
    H_c = world.h[~world.is_wrong]
    s_c = world.s[~world.is_wrong]
    W = world.W_readout
    h_t = torch.tensor(H_c, dtype=torch.float64, requires_grad=True)
    W_t = torch.tensor(W, dtype=torch.float64)
    s_t = torch.tensor(s_c, dtype=torch.long)

    logits = h_t @ W_t.T   # (n_c, n_classes)
    n_classes = logits.shape[1]
    target_logit = logits.gather(1, s_t.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, s_t.unsqueeze(1), float("-inf"))
    other_max, _ = masked.max(dim=1)
    LD = (target_logit - other_max).sum()    # scalar; sum so grad gives per-sample
    LD.backward()

    grad_np = h_t.grad.detach().numpy()   # (n_c, d_m)
    proj = grad_np @ U_V @ np.asarray(mu_xi_hat).reshape(-1)
    return float(proj.mean())


if __name__ == "__main__":
    from synth_world import make_world
    from manifold_methods import fit_parametric

    w = make_world(d_m=64, n_pairs=2000, sigma=0.05, wrong_frac=0.1,
                   perturbation_kind="off_span", perturbation_norm=0.5,
                   perturbation_dim=1, build_readout=True)
    H_c = w.h[~w.is_wrong]
    s_c = w.s[~w.is_wrong]
    U_M = fit_parametric(H_c, s_c)

    # Estimate mu_xi from wrong-population residuals
    H_w = w.h[w.is_wrong]
    r_w = H_w - H_w @ U_M @ U_M.T
    r_c = H_c - H_c @ U_M @ U_M.T
    mu_xi_hat_full = r_w.mean(axis=0) - r_c.mean(axis=0)   # (d_m,)
    # Project into V_true to get the localized perturbation in V-coords
    mu_xi_hat_V = w.V_true.T @ mu_xi_hat_full   # (r,)

    print(f"True ||mu_xi|| = {np.linalg.norm(w.xi[w.wrong_idx[0]]):.3f}, estimated ||mu_xi||_V = {np.linalg.norm(mu_xi_hat_V):.3f}")

    n_acc, n_std = ACE_N(w, U_M)
    nv_acc, nv_std = ACE_NV(w, w.V_true)
    s_acc, s_std = ACE_S(w, w.V_true, mu_xi_hat_V)
    r_acc, r_std = ACE_R(w, U_M, r=1, n_replicates=50)

    pred_signed = predicted_ACE_S_linear(w, w.V_true, mu_xi_hat_V)
    pred_autograd = predicted_ACE_S_autograd(w, w.V_true, mu_xi_hat_V)
    pred_mag = predicted_ACE_S_magnitude(w, w.V_true, mu_xi_hat_V)

    print(f"  ACE_N    = {n_acc:+.4f} +- {n_std:.4f}")
    print(f"  ACE_NV   = {nv_acc:+.4f} +- {nv_std:.4f}")
    print(f"  ACE_S    = {s_acc:+.4f} +- {s_std:.4f}    (sufficient = negative)")
    print(f"  ACE_R    = {r_acc:+.4f} +- {r_std:.4f}    (random baseline)")
    print(f"  predicted_ACE_S (signed Taylor)   = {pred_signed:+.4f}  (matches actual only in linear regime)")
    print(f"  predicted_ACE_S (autograd Taylor) = {pred_autograd:+.4f}")
    print(f"  predicted |ACE_S| (Prop 4 LB)     = {pred_mag:.4f}   (need |ACE_S| >= this)")
    print(f"  bound holds: {abs(s_acc) >= pred_mag - 0.01}")
    print("OK")
