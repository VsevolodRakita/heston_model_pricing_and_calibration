"""The implied-volatility surface approximation network (PyTorch).

A fully-connected feed-forward network following Horvath, Muguruza & Tomas
(2019): 4 hidden layers of 30 ELU units, mapping the 5 Heston parameters to the
88-point implied-vol grid.

Normalisation is baked into the module as buffers, so a loaded checkpoint maps
*raw* Heston parameters straight to implied vols (and is differentiable end to
end, which the calibration step relies on):

    parameters --(to [-1,1] via box)--> MLP --(un-standardise)--> implied vols
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HestonSurfaceMLP(nn.Module):
    def __init__(self, n_params: int, n_grid: int,
                 hidden: int = 30, n_hidden: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_params
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ELU()]
            d = hidden
        layers += [nn.Linear(d, n_grid)]
        self.net = nn.Sequential(*layers)

        # Normalisation buffers (persisted with the checkpoint). Defaults are
        # identity; set them via `set_normalisation` before training.
        self.register_buffer("param_lo", torch.zeros(n_params))
        self.register_buffer("param_hi", torch.ones(n_params))
        self.register_buffer("y_mean", torch.zeros(n_grid))
        self.register_buffer("y_std", torch.ones(n_grid))

    def set_normalisation(self, param_box, y_mean, y_std) -> None:
        box = torch.as_tensor(param_box, dtype=torch.float32)
        self.param_lo.copy_(box[:, 0])
        self.param_hi.copy_(box[:, 1])
        self.y_mean.copy_(torch.as_tensor(y_mean, dtype=torch.float32))
        self.y_std.copy_(torch.as_tensor(y_std, dtype=torch.float32))

    def normalize_params(self, x: torch.Tensor) -> torch.Tensor:
        return (2.0 * x - (self.param_hi + self.param_lo)) / (self.param_hi - self.param_lo)

    def iv_from_normalized(self, z: torch.Tensor) -> torch.Tensor:
        """Map parameters already in [-1,1] to implied vols (un-standardised)."""
        return self.net(z) * self.y_std + self.y_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map *raw* Heston parameters (..., 5) to implied vols (..., 88)."""
        return self.iv_from_normalized(self.normalize_params(x))
