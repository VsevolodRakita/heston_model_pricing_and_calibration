"""Deep calibration: invert the trained surface network.

Step (ii) of the two-step approach. Given a market implied-vol surface on the
fixed grid, recover the Heston parameters by

    argmin_theta  || NN(theta) - sigma_market ||^2

solved with LBFGS through the differentiable network (parameters optimised in
the normalised [-1,1] box, so the constraints hold by construction). Because a
forward pass is a handful of small matrix products, this runs in milliseconds.
"""
from __future__ import annotations

import time

import torch

from model import HestonSurfaceMLP

# The network is tiny (a few thousand weights). PyTorch's default intra-op
# threading adds far more scheduling overhead than it saves on matrices this
# small, so a single thread makes calibration ~100x faster (ms, not seconds).
torch.set_num_threads(1)


def load_model(path: str):
    ckpt = torch.load(path, weights_only=False)
    model = HestonSurfaceMLP(ckpt["n_params"], ckpt["n_grid"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _denormalize(model: HestonSurfaceMLP, z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (z * (model.param_hi - model.param_lo) + (model.param_hi + model.param_lo))


def calibrate_surface(model: HestonSurfaceMLP, target_iv,
                      n_restarts: int = 4, max_iter: int = 100, seed: int = 0):
    """Recover Heston parameters from one implied-vol surface (shape (n_grid,)).

    Returns a dict with the recovered params, the RMSE fit (in IV units), the
    fitted surface, and the wall-clock time.
    """
    target = torch.as_tensor(target_iv, dtype=torch.float32).flatten()
    gen = torch.Generator().manual_seed(seed)

    best = None
    t0 = time.perf_counter()
    for _ in range(n_restarts):
        z = (2.0 * torch.rand(model.param_lo.numel(), generator=gen) - 1.0).requires_grad_(True)
        opt = torch.optim.LBFGS([z], lr=1.0, max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = model.iv_from_normalized(z.clamp(-1.0, 1.0))
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            zc = z.clamp(-1.0, 1.0)
            pred = model.iv_from_normalized(zc)
            rmse = torch.sqrt(((pred - target) ** 2).mean()).item()
            if best is None or rmse < best["rmse"]:
                best = {
                    "params": _denormalize(model, zc).numpy(),
                    "rmse": rmse,
                    "iv": pred.numpy(),
                }
    best["seconds"] = time.perf_counter() - t0
    return best
