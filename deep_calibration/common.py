"""Shared definitions for the deep-learning volatility component.

Depends only on numpy, so it is importable from *both* Python environments:
the MSYS2 interpreter that holds the compiled `heston` module (data generation)
and the torch venv that trains the network. It fixes the (maturity, strike)
grid and the Heston parameter box that the whole pipeline is built around.

Reference: Horvath, Muguruza & Tomas (2019), "Deep Learning Volatility".
The network learns the map  theta -> implied-vol surface  on a fixed grid.
"""
from __future__ import annotations

import numpy as np

# ---- Fixed market (we learn the *volatility* surface, so rates are held at 0)
S0 = 100.0
R = 0.0
Q = 0.0

# ---- Fixed grid: 8 maturities x 11 strikes = 88 "pixels" (paper uses 8 x 11).
MATURITIES = np.array([0.10, 0.30, 0.60, 0.90, 1.20, 1.50, 1.80, 2.00])
STRIKES = np.linspace(80.0, 120.0, 11)
GRID_SHAPE = (MATURITIES.size, STRIKES.size)   # (8, 11)
N_GRID = int(MATURITIES.size * STRIKES.size)   # 88

# ---- Heston parameter box  (v0, kappa, theta, sigma, rho).
PARAM_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
PARAM_BOX = np.array([
    [0.01, 0.16],   # v0     initial variance   (vol 10%..40%)
    [0.50, 5.00],   # kappa  mean reversion
    [0.01, 0.16],   # theta  long-run variance
    [0.10, 1.00],   # sigma  vol-of-vol
    [-0.95, 0.00],  # rho    spot/vol correlation (equity skew: negative)
])
N_PARAMS = PARAM_BOX.shape[0]  # 5


def normalize_params(x: np.ndarray) -> np.ndarray:
    """Map parameters from the box to [-1, 1] (per the paper's normalisation)."""
    lo, hi = PARAM_BOX[:, 0], PARAM_BOX[:, 1]
    return (2.0 * x - (hi + lo)) / (hi - lo)


def denormalize_params(z: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize_params`: map [-1, 1] back to the box."""
    lo, hi = PARAM_BOX[:, 0], PARAM_BOX[:, 1]
    return 0.5 * (z * (hi - lo) + (hi + lo))


def latin_hypercube(n: int, seed: int = 0) -> np.ndarray:
    """Latin-hypercube sample of `n` parameter vectors inside the box.

    Better space-filling than plain uniform sampling for surrogate training.
    """
    rng = np.random.default_rng(seed)
    d = N_PARAMS
    out = np.empty((n, d))
    for j in range(d):
        # stratified positions in [0,1), one per row, then shuffled per column
        cuts = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(cuts)
        lo, hi = PARAM_BOX[j]
        out[:, j] = lo + cuts * (hi - lo)
    return out
