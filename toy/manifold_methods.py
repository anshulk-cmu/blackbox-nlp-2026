"""
Five manifold-recovery methods (paper section 2.6).

Each method takes correct-population activations H_c (n_c, d_m) plus auxiliary
labels (e.g. the integer answer s for each sample) and returns an (d_m, k)
matrix U with orthonormal columns whose column span is the estimated manifold M_hat.

We also expose principal_angle(U1, U2) to compute sin(theta_max) between two
estimates and a `recover` convenience routine that calls all five.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from synth_world import basis_vector, HELIX_DIM


# ---------------------------------------------------------------------------
# Method 1: Parametric helix-basis OLS
# ---------------------------------------------------------------------------

def make_design_matrix(s: np.ndarray) -> np.ndarray:
    """Return B in R^{n x 9} with rows B(s_i)."""
    return basis_vector(s)   # (n, 9)


def fit_parametric(H_c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Method 1: parametric OLS on helix basis.

    Returns U_hat: (d_m, 9) orthonormal columns spanning M_hat.
    """
    B = make_design_matrix(s)              # (n, 9)
    # Solve B C^T = H_c -> C_hat = (B^T B)^{-1} B^T H_c   (shape: 9 x d_m)
    C_hat, *_ = np.linalg.lstsq(B, H_c, rcond=None)   # (9, d_m)
    # Column span of C_hat^T (in R^{d_m}) is M_hat. Orthonormalise.
    Q, _ = np.linalg.qr(C_hat.T)            # (d_m, 9)
    return Q


# ---------------------------------------------------------------------------
# Method 2: PCA on class means (with binning)
# ---------------------------------------------------------------------------

def fit_pca_bins(H_c: np.ndarray, s: np.ndarray, k: int = HELIX_DIM, K_bins: int = 20) -> np.ndarray:
    """Method 2: bin s into K_bins equal-width bins, average within bin, SVD of bin means.

    Returns U_hat: (d_m, k) orthonormal columns.
    """
    s_min, s_max = float(s.min()), float(s.max())
    edges = np.linspace(s_min, s_max + 1e-9, K_bins + 1)
    bin_idx = np.searchsorted(edges, s, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, K_bins - 1)
    means = []
    for k_bin in range(K_bins):
        mask = bin_idx == k_bin
        if mask.sum() > 0:
            means.append(H_c[mask].mean(axis=0))
    M_means = np.stack(means, axis=0)        # (K_bins_used, d_m)
    M_centered = M_means - M_means.mean(axis=0, keepdims=True)
    U, _, _ = np.linalg.svd(M_centered, full_matrices=False)
    # U: (K_bins_used, K_bins_used). We want top-k *right* singular vectors of M_centered:
    # M_centered = U S Vt -> column space of M_centered (in R^{d_m}) is span(Vt[:rank].T)
    _, _, Vt = np.linalg.svd(M_centered, full_matrices=False)
    U_hat = Vt[:k].T                        # (d_m, k)
    return U_hat


# ---------------------------------------------------------------------------
# Method 3: Local PCA (Singer-Wu 2012)
# ---------------------------------------------------------------------------

def fit_local_pca(
    H_c: np.ndarray,
    intrinsic_dim: int = 3,
    k_neighbors: int = 30,
) -> np.ndarray:
    """Method 3: local PCA, average projection matrices across all points.

    Returns U_hat: (d_m, intrinsic_dim) orthonormal columns.
    """
    from sklearn.neighbors import NearestNeighbors

    n_c, d_m = H_c.shape
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(H_c)
    _, idxs = nbrs.kneighbors(H_c)          # (n_c, k_neighbors)

    P_avg = np.zeros((d_m, d_m))
    for i in range(n_c):
        nbrs_i = H_c[idxs[i]] - H_c[i]      # (k_neighbors, d_m), centered at h_i
        # Local SVD: top intrinsic_dim right singular vectors are local tangent
        _, _, Vt = np.linalg.svd(nbrs_i, full_matrices=False)
        U_local = Vt[:intrinsic_dim].T      # (d_m, intrinsic_dim)
        P_avg += U_local @ U_local.T
    P_avg /= n_c

    # Eigendecompose averaged projection; top intrinsic_dim eigenvectors -> U_hat
    eigvals, eigvecs = np.linalg.eigh(P_avg)
    # Largest eigenvalues are at the end of eigh's output
    U_hat = eigvecs[:, -intrinsic_dim:]
    return U_hat


# ---------------------------------------------------------------------------
# Method 4: Diffusion Maps + regression lift
# ---------------------------------------------------------------------------

def _pairwise_sq_dists(H: np.ndarray) -> np.ndarray:
    """Return (n, n) matrix of squared distances. Avoids the (n, n, d_m) broadcast."""
    sq_norms = (H * H).sum(axis=1)
    G = H @ H.T
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * G
    np.maximum(d2, 0, out=d2)
    return d2


def fit_diffusion_maps(
    H_c: np.ndarray,
    k: int = HELIX_DIM,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Method 4: Diffusion maps embedding then regression lift.

    Returns U_hat: (d_m, k) orthonormal columns.
    """
    dists2 = _pairwise_sq_dists(H_c)        # (n, n)

    if bandwidth is None:
        # Median heuristic on off-diagonal entries
        off = dists2[~np.eye(dists2.shape[0], dtype=bool)]
        bandwidth = float(np.sqrt(np.median(off)))
    sigma_DM2 = 2.0 * bandwidth ** 2

    K = np.exp(-dists2 / sigma_DM2)
    d = K.sum(axis=1, keepdims=True)
    P = K / d                              # row-normalised

    eigvals, eigvecs = np.linalg.eig(P)    # not symmetric, use eig
    # Sort by |eigval| descending
    order = np.argsort(-np.abs(eigvals))
    eigvecs = np.real(eigvecs[:, order])
    eigvals = np.real(eigvals[order])

    # Skip the trivial constant eigenvector (eigval = 1) - take indices 1..k
    Phi = eigvecs[:, 1:k + 1]              # (n, k)

    # Lift back: solve H_c ~ Phi A   ->  A = (Phi^T Phi)^-1 Phi^T H_c
    A, *_ = np.linalg.lstsq(Phi, H_c, rcond=None)   # (k, d_m)
    Q, _ = np.linalg.qr(A.T)               # (d_m, k)
    return Q


# ---------------------------------------------------------------------------
# Method 5: Kernel PCA (RBF) + regression lift
# ---------------------------------------------------------------------------

def fit_kernel_pca(
    H_c: np.ndarray,
    k: int = HELIX_DIM,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """Method 5: kernel PCA with RBF + regression lift.

    Returns U_hat: (d_m, k) orthonormal columns.
    """
    dists2 = _pairwise_sq_dists(H_c)

    if gamma is None:
        off = dists2[~np.eye(dists2.shape[0], dtype=bool)]
        median_d2 = float(np.median(off))
        gamma = 1.0 / median_d2

    K = np.exp(-gamma * dists2)
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_c = K - one_n @ K - K @ one_n + one_n @ K @ one_n   # double-centred

    eigvals, eigvecs = np.linalg.eigh(K_c)
    # Sort descending
    order = np.argsort(-eigvals)
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]
    Phi = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0.0))   # (n, k)

    # Lift
    A, *_ = np.linalg.lstsq(Phi, H_c, rcond=None)   # (k, d_m)
    Q, _ = np.linalg.qr(A.T)               # (d_m, k)
    return Q


# ---------------------------------------------------------------------------
# Principal angles
# ---------------------------------------------------------------------------

def principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Return all principal angles between span(U1) and span(U2), in radians, ascending.

    U1: (d_m, k1), U2: (d_m, k2), columns orthonormal.
    """
    M = U1.T @ U2
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.arccos(sv)


def sin_theta_max(U1: np.ndarray, U2: np.ndarray) -> float:
    """Return sin of the maximum principal angle (i.e. sin theta_max)."""
    angles = principal_angles(U1, U2)
    return float(np.sin(angles.max()))


# ---------------------------------------------------------------------------
# Convenience: run all 5 methods
# ---------------------------------------------------------------------------

@dataclass
class ManifoldFits:
    parametric: np.ndarray
    pca_bins: np.ndarray
    local_pca: np.ndarray
    diffusion_maps: np.ndarray
    kernel_pca: np.ndarray

    @property
    def all(self) -> dict:
        return {
            "parametric": self.parametric,
            "pca_bins": self.pca_bins,
            "local_pca": self.local_pca,
            "diffusion_maps": self.diffusion_maps,
            "kernel_pca": self.kernel_pca,
        }


def fit_all(
    H_c: np.ndarray,
    s: np.ndarray,
    k: int = HELIX_DIM,
    intrinsic_dim_local: int = 3,
    K_bins: int = 20,
    k_neighbors: int = 30,
) -> ManifoldFits:
    return ManifoldFits(
        parametric=fit_parametric(H_c, s),
        pca_bins=fit_pca_bins(H_c, s, k=k, K_bins=K_bins),
        local_pca=fit_local_pca(H_c, intrinsic_dim=intrinsic_dim_local, k_neighbors=k_neighbors),
        diffusion_maps=fit_diffusion_maps(H_c, k=k),
        kernel_pca=fit_kernel_pca(H_c, k=k),
    )


if __name__ == "__main__":
    from synth_world import make_world

    w = make_world(d_m=64, n_pairs=2000, sigma=0.05, wrong_frac=0.0,
                   perturbation_kind="none", build_readout=False)
    H_c = w.h[~w.is_wrong]
    s_c = w.s[~w.is_wrong]
    M_true = w.C   # (d_m, 9)

    fits = fit_all(H_c, s_c, k=HELIX_DIM)
    for name, U in fits.all.items():
        sintheta = sin_theta_max(U, M_true)
        print(f"  {name:20s}: sin(theta_max) = {sintheta:.4f}  ({np.degrees(np.arcsin(sintheta)):.2f} deg)")
    print("OK")
