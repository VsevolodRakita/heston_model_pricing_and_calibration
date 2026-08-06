"""Train the implied-volatility surface network.

Runs in the torch environment. Reads the .npz produced by make_dataset.py and
writes a self-contained checkpoint (weights + normalisation + grid metadata).

Usage (from the torch venv):
    python train.py --data data/heston_iv.npz --out models/surface_mlp.pt
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import HestonSurfaceMLP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/heston_iv.npz")
    ap.add_argument("--out", type=str, default="models/surface_mlp.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(args.data, allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    Y = torch.tensor(d["Y"], dtype=torch.float32)
    box = d["param_box"]
    n_params, n_grid = X.shape[1], Y.shape[1]

    # Train / validation split.
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(round(args.val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xval, Yval = X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]

    # Output standardisation from the training split (per the paper).
    y_mean = Ytr.mean(0)
    y_std = Ytr.std(0).clamp_min(1e-8)

    model = HestonSurfaceMLP(n_params, n_grid)
    model.set_normalisation(box, y_mean, y_std)

    # Pre-compute standardised targets and normalised inputs for training.
    Ztr = model.normalize_params(Xtr)
    Ttr = (Ytr - y_mean) / y_std
    loader = DataLoader(TensorDataset(Ztr, Ttr), batch_size=args.batch, shuffle=True)

    Zval = model.normalize_params(Xval)
    Tval = (Yval - y_mean) / y_std

    opt = torch.optim.Adam(model.net.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    since_improve = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for zb, tb in loader:
            opt.zero_grad()
            loss = loss_fn(model.net(zb), tb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model.net(Zval), Tval).item()
            # RMSE in raw implied-vol units, for interpretability.
            pred_iv = model.iv_from_normalized(Zval)
            val_rmse = torch.sqrt(torch.mean((pred_iv - Yval) ** 2)).item()

        if val_loss < best_val - 1e-6:
            best_val, best_state, since_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since_improve += 1

        if epoch % 10 == 0 or since_improve == 0:
            print(f"epoch {epoch:3d}  val_mse(std)={val_loss:.5f}  val_rmse(IV)={val_rmse:.5f}  "
                  f"best={best_val:.5f}  patience={since_improve}/{args.patience}")

        if since_improve >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_rmse = torch.sqrt(torch.mean((model(Xval) - Yval) ** 2)).item()
        final_max = (model(Xval) - Yval).abs().max().item()
    dt = time.perf_counter() - t0

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_params": n_params, "n_grid": n_grid,
        "maturities": d["maturities"], "strikes": d["strikes"],
        "param_box": box, "param_names": d["param_names"],
        "s0": float(d["s0"]), "r": float(d["r"]), "q": float(d["q"]),
    }, out_path)
    print(f"\ntrained in {dt:.1f}s  |  validation RMSE {final_rmse:.5f}  max abs err {final_max:.5f} (IV)")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
