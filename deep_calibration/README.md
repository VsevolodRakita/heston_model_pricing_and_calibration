# Deep Learning Volatility (Heston)

A neural-network pricing map and fast calibration for the Heston model, following
Horvath, Muguruza & Tomas (2019), *Deep Learning Volatility*.

A feed-forward network (**4 hidden layers × 30 ELU units**, ~5.7k weights) learns

```
Heston parameters (v0, kappa, theta, sigma, rho)  ->  implied-vol surface on a fixed 8×11 grid
```

trained *offline* against the C++ Fourier pricer. Calibration then inverts the
fast, differentiable network in milliseconds (LBFGS through autodiff), instead of
running the numerical pricer inside the optimiser loop.

## Two environments

The pipeline deliberately spans two Python interpreters that hand off a file:

| Stage | Script | Interpreter | Needs |
|-------|--------|-------------|-------|
| Generate data | `make_dataset.py` | MSYS2/UCRT64 Python | the compiled `heston` module + numpy |
| Train | `train.py` | torch venv | torch + numpy |
| Calibrate | `calibrate.py` | torch venv | torch + numpy |

They never talk in-process (the `heston` extension is built for the MSYS2 Python;
PyTorch has no wheels for it or for Python 3.14). The `.npz` dataset is the bridge.

### 1. Generate the training data (MSYS2 UCRT64 shell)

Build the `heston` module first (`-DHESTON_BUILD_PYTHON=ON`), then:

```bash
cd deep_calibration
python make_dataset.py --n-samples 40000 --nodes 4000 --out data/heston_iv.npz
```

~10 min on 11 cores. A small held-out set (`data/heston_iv_test.npz`, seed 999) is
committed so the notebook runs without regenerating the full set.

### 2. Train (torch venv)

Create the venv once (needs [`uv`](https://docs.astral.sh/uv/)):

```bash
uv venv .venv-torch --python 3.12
uv pip install --python .venv-torch/Scripts/python.exe -r deep_calibration/requirements-torch.txt
```

Then:

```bash
.venv-torch/Scripts/python deep_calibration/train.py \
    --data deep_calibration/data/heston_iv.npz \
    --out  deep_calibration/models/surface_mlp.pt
```

~2.5 min on CPU; early-stops around 10 bps validation RMSE. The checkpoint
(`models/surface_mlp.pt`, committed) is self-contained: it stores the weights,
the parameter box, the output standardisation, and the grid.

### 3. Calibrate / explore

See [`../notebooks/deep_calibration_demo.ipynb`](../notebooks/deep_calibration_demo.ipynb)
for the accuracy, calibration round-trip, and speed-up results.

## Results (this repo, held-out set)

- Approximation: **~10 bps RMSE**, 95th-pct ~12 bps (errors concentrate at the
  short-maturity wing, as in the paper).
- Calibration: full surface fit in **~100 ms**; parameter recovery is limited by
  the usual Heston identifiability (κ/σ), not the fit.
- The network prices a surface **~10⁵× faster** than the Fourier engine.

## Grid & parameter box

Fixed grid: 8 maturities `[0.1 … 2.0]y` × 11 strikes `[80 … 120]` (spot 100,
`r = q = 0`). Parameter box in [`common.py`](common.py). Both are stored in the
dataset and the checkpoint, so nothing downstream hard-codes them.
