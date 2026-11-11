"""Chapter 1 polynomial curve fitting: shared helpers (NumPy only).

Used by the Ch1 wiki pages (concepts/algorithms) and by make_figures.py.
Data: t = sin(2*pi*x) + Gaussian noise (sigma = 0.3), x on an evenly spaced grid in [0, 1].
"""
import numpy as np


def make_data(n, rng, sigma=0.3):
    """Synthetic data of Ch1 section 1.2.1: evenly spaced x, noisy sin(2*pi*x)."""
    x = np.linspace(0.0, 1.0, n)
    t = np.sin(2 * np.pi * x) + rng.normal(0.0, sigma, size=n)
    return x, t


def design_poly(x, M):
    """Design matrix Phi[n, j] = x_n ** j, shape (N, M+1)."""
    return np.vander(x, M + 1, increasing=True)


def fit_ls(Phi, t):
    """Least squares w* = argmin 0.5*||Phi w - t||^2 (same solution as the normal equations).

    lstsq is used instead of solve(Phi.T @ Phi, ...) because Phi.T @ Phi is extremely
    ill-conditioned for M = 9.  For rank-deficient Phi it returns the minimum-norm solution.
    """
    return np.linalg.lstsq(Phi, t, rcond=None)[0]


def fit_normal(Phi, t):
    """Textbook normal equations: solve (Phi^T Phi) w = Phi^T t."""
    return np.linalg.solve(Phi.T @ Phi, Phi.T @ t)


def fit_ridge(Phi, t, lam):
    """Regularized least squares w = (Phi^T Phi + lam I)^{-1} Phi^T t  (eq. (4) of the notes)."""
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ t)


def erms(Phi, t, w):
    """Root-mean-square error, eq. (3): sqrt(mean((Phi w - t)^2))."""
    return float(np.sqrt(np.mean((Phi @ w - t) ** 2)))
