"""Generate the training dataset for the deep-learning volatility surrogate.

Runs in the interpreter that has the compiled `heston` module (the MSYS2 UCRT64
Python). For each sampled Heston parameter vector it prices the fixed 8x11 grid
with the Fourier engine and inverts to Black-Scholes implied vols, producing the
map  theta -> 88 implied vols  that the network will learn.

Output: an .npz with X (N, 5) parameters, Y (N, 88) implied vols, and the grid
metadata. This file is the hand-off to the torch training environment.

Usage (from the MSYS2 UCRT64 shell):
    python make_dataset.py --n-samples 40000 --nodes 4000 --out data/heston_iv.npz
"""
from __future__ import annotations

import argparse
import os

# Each worker does only light numpy; stop OpenBLAS from spawning a thread pool
# per process (avoids oversubscription across the process Pool). Must precede
# the numpy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pathlib
import sys
import time

import numpy as np

import common

# Make the compiled `heston` module importable (built into build/bindings).
_HERE = pathlib.Path(__file__).resolve().parent
for _c in (_HERE / ".." / "build" / "bindings", pathlib.Path.cwd() / "build" / "bindings"):
    if _c.exists() and (list(_c.glob("heston*.pyd")) or list(_c.glob("heston*.so"))):
        sys.path.insert(0, str(_c.resolve()))
        break

import heston as h  # noqa: E402

# Per-process globals (each worker builds its own pricer and option grid once).
_PRICER = None
_MKT = None
_OPTIONS = None  # list of 88 VanillaOption, row-major over (maturity, strike)


def _init_worker(nodes: int) -> None:
    global _PRICER, _MKT, _OPTIONS
    _MKT = h.Market(common.S0, common.R, common.Q)
    _PRICER = h.HestonFourierPricer(1.5, 200.0, int(nodes))
    _OPTIONS = [
        h.VanillaOption(h.OptionType.Call, float(K), float(T))
        for T in common.MATURITIES
        for K in common.STRIKES
    ]


def _price_surface(x: np.ndarray):
    """Return the 88 implied vols for one parameter vector, or None on failure."""
    p = h.HestonParams(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
    out = np.empty(common.N_GRID)
    for i, opt in enumerate(_OPTIONS):
        try:
            px = _PRICER.price(opt, _MKT, p)
            out[i] = h.implied_vol(opt, _MKT, px)
        except Exception:
            return None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=40000)
    ap.add_argument("--nodes", type=int, default=4000, help="Fourier Simpson intervals")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=str, default=str(_HERE / "data" / "heston_iv.npz"))
    args = ap.parse_args()

    X = common.latin_hypercube(args.n_samples, seed=args.seed)

    t0 = time.perf_counter()
    from multiprocessing import Pool
    with Pool(processes=args.processes, initializer=_init_worker, initargs=(args.nodes,)) as pool:
        results = pool.map(_price_surface, list(X), chunksize=64)
    dt = time.perf_counter() - t0

    keep = [i for i, r in enumerate(results) if r is not None]
    X_ok = X[keep]
    Y_ok = np.array([results[i] for i in keep])
    n_drop = len(results) - len(keep)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X_ok, Y=Y_ok,
        maturities=common.MATURITIES, strikes=common.STRIKES,
        param_box=common.PARAM_BOX, param_names=np.array(common.PARAM_NAMES),
        s0=common.S0, r=common.R, q=common.Q, nodes=args.nodes,
    )
    print(f"generated {len(X_ok)} surfaces ({n_drop} dropped) in {dt:.1f}s "
          f"using {args.processes} processes -> {out_path}")
    print(f"IV range: [{Y_ok.min():.4f}, {Y_ok.max():.4f}]  mean {Y_ok.mean():.4f}")


if __name__ == "__main__":
    main()
