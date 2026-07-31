# %% [markdown]
# # Heston model — pricing, implied-vol surface, Greeks & calibration
#
# This notebook drives the C++ `heston` engine through its `pybind11` bindings.
# It shows the three things the library is built to do:
#
# 1. Build a **Heston implied-volatility surface** (Fourier pricing + BS inversion).
# 2. Read off **semi-analytic Greeks** (delta / gamma) across strikes.
# 3. **Calibrate** the five Heston parameters back from a set of option prices.
#
# Build the module first with `-DHESTON_BUILD_PYTHON=ON` (see the README).

# %%
import sys
import pathlib

import numpy as np
import matplotlib
try:
    get_ipython()  # noqa: F821  -- only defined inside a Jupyter/IPython kernel
except NameError:
    matplotlib.use("Agg")  # plain script: render headless and just save PNGs
import matplotlib.pyplot as plt

# Locate the compiled `heston` extension (built into build/bindings).
_here = pathlib.Path(globals().get("__file__", pathlib.Path.cwd() / "x")).resolve().parent
_candidates = [_here / ".." / "build" / "bindings", pathlib.Path.cwd() / "build" / "bindings"]
for _c in _candidates:
    if _c.exists() and (list(_c.glob("heston*.pyd")) or list(_c.glob("heston*.so"))):
        sys.path.insert(0, str(_c.resolve()))
        break

import heston as h  # noqa: E402

IMG = _here / "img"
IMG.mkdir(exist_ok=True)

# %% [markdown]
# ## Market & model
# A spot of 100, 3% rate, 1% dividend yield, and a Heston parameter set with a
# pronounced negative spot/vol correlation (which produces the equity skew).

# %%
mkt = h.Market(s0=100.0, r=0.03, q=0.01)
params = h.HestonParams(v0=0.04, kappa=1.5, theta=0.05, sigma=0.6, rho=-0.7)
fp = h.HestonFourierPricer(alpha=1.5, u_max=250.0, n_intervals_even=12000)

atm = h.VanillaOption(h.OptionType.Call, 100.0, 1.0)
print("ATM 1y call price :", round(fp.price(atm, mkt, params), 6))
print("Feller satisfied  :", params.satisfies_feller())

# %% [markdown]
# ## 1. Implied-volatility surface
# For each (strike, maturity) we price a call with the Fourier engine and invert
# the Black–Scholes formula to recover the implied vol. The downward slope in
# strike is the volatility skew induced by `rho < 0`.

# %%
strikes = np.linspace(70.0, 135.0, 40)
maturities = np.linspace(0.1, 2.0, 25)
IV = np.full((maturities.size, strikes.size), np.nan)

for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        opt = h.VanillaOption(h.OptionType.Call, float(K), float(T))
        try:
            px = fp.price(opt, mkt, params)
            IV[i, j] = h.implied_vol(opt, mkt, px)
        except Exception:
            pass  # leave NaN where inversion is ill-posed (deep wings)

Kg, Tg = np.meshgrid(strikes, maturities)
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(Kg, Tg, IV * 100.0, cmap="viridis", linewidth=0, antialiased=True)
ax.set_xlabel("Strike")
ax.set_ylabel("Maturity (years)")
ax.set_zlabel("Implied vol (%)")
ax.set_title("Heston implied-volatility surface")
fig.colorbar(surf, shrink=0.6, aspect=12, label="IV (%)")
fig.tight_layout()
fig.savefig(IMG / "iv_surface.png", dpi=130)
print("saved", IMG / "iv_surface.png")
plt.show()

# %% [markdown]
# ## 2. Semi-analytic Greeks across strikes (T = 1y)
# Delta and gamma come from differentiating the Carr–Madan integral under the
# integral sign — a single quadrature pass returns price, delta, gamma and rho.

# %%
Ks = np.linspace(70.0, 135.0, 70)
delta = np.empty_like(Ks)
gamma = np.empty_like(Ks)
for idx, K in enumerate(Ks):
    opt = h.VanillaOption(h.OptionType.Call, float(K), 1.0)
    g = fp.analytic_greeks(opt, mkt, params)
    delta[idx] = g.delta
    gamma[idx] = g.gamma

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(Ks, delta, color="C0", label="delta")
ax1.set_xlabel("Strike")
ax1.set_ylabel("Delta", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")
ax1.axvline(mkt.s0, ls=":", color="grey", alpha=0.7)
ax2 = ax1.twinx()
ax2.plot(Ks, gamma, color="C3", label="gamma")
ax2.set_ylabel("Gamma", color="C3")
ax2.tick_params(axis="y", labelcolor="C3")
ax1.set_title("Analytic call delta & gamma vs strike (T = 1y)")
fig.tight_layout()
fig.savefig(IMG / "greeks.png", dpi=130)
print("saved", IMG / "greeks.png")
plt.show()

# %% [markdown]
# ## 3. Calibration round-trip
# We generate synthetic call prices from a *true* parameter set, then calibrate
# (CMA-ES) starting from a deliberately wrong guess and check that we recover the
# smile. The table reports the recovered parameters and the fit loss.

# %%
true = params
quote_strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
quote_mats = [0.5, 1.0, 1.5]
quotes = []
for T in quote_mats:
    for K in quote_strikes:
        opt = h.VanillaOption(h.OptionType.Call, K, T)
        quotes.append(h.PriceQuote(opt, fp.price(opt, mkt, true), 1.0))

guess = h.HestonParams(v0=0.05, kappa=1.0, theta=0.06, sigma=0.4, rho=-0.3)
report = h.calibrate_heston_to_prices_cmaes(quotes, mkt, guess)
rec = report.params

print(f"{'param':>7} {'true':>10} {'recovered':>12}")
for name in ("v0", "kappa", "theta", "sigma", "rho"):
    print(f"{name:>7} {getattr(true, name):>10.4f} {getattr(rec, name):>12.4f}")
print(f"calibration loss = {report.loss:.3e}   iters = {report.iters}")

# Compare true vs recovered smile at T = 1y.
smile_K = np.linspace(80.0, 120.0, 30)
iv_true = np.empty_like(smile_K)
iv_rec = np.empty_like(smile_K)
for idx, K in enumerate(smile_K):
    opt = h.VanillaOption(h.OptionType.Call, float(K), 1.0)
    iv_true[idx] = h.implied_vol(opt, mkt, fp.price(opt, mkt, true))
    iv_rec[idx] = h.implied_vol(opt, mkt, fp.price(opt, mkt, rec))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(smile_K, iv_true * 100.0, "o", color="C0", label="true", markersize=5)
ax.plot(smile_K, iv_rec * 100.0, "-", color="C3", label="calibrated")
ax.set_xlabel("Strike")
ax.set_ylabel("Implied vol (%)")
ax.set_title("Calibration round-trip: recovered smile (T = 1y)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG / "calibration.png", dpi=130)
print("saved", IMG / "calibration.png")
plt.show()

# %% [markdown]
# ## 4. GPR surrogate pricer
# Following De Spiegeleer, Madan, Reyners & Schoutens (2018), we train a
# **Gaussian-process surrogate** for the Heston price over a box in the seven
# features `(K, T, v0, kappa, theta, sigma, rho)`. The surrogate is trained
# offline against the Fourier engine and then prices new points with a cheap
# kernel-vector product. It implements the same `IVanillaPricer` interface, so
# it is a drop-in engine. Here we reproduce the paper's Fig. 8: out-of-sample
# GPR prices scattered against the "true" Fourier prices.

# %%
import time

gpr = h.GprPricer(
    reference=mkt,
    K=(80.0, 120.0), T=(0.30, 1.50), v0=(0.02, 0.08), kappa=(1.0, 3.0),
    theta=(0.02, 0.08), sigma=(0.20, 0.60), rho=(-0.80, -0.20),
    n_train=1200, seed=20240607, optimize_hyperparameters=True, max_opt_iter=80,
)

t0 = time.perf_counter()
gpr.train(fp)                       # offline training against the Fourier engine
train_secs = time.perf_counter() - t0
print(f"trained on 1200 points in {train_secs:.1f}s")

# Out-of-sample test set (drawn strictly inside the trained box).
rng = np.random.default_rng(2024)
n_test = 400
true_px = np.empty(n_test)
gpr_px = np.empty(n_test)
t_fourier = 0.0
t_gpr = 0.0
for i in range(n_test):
    K = rng.uniform(82.0, 118.0)
    T = rng.uniform(0.35, 1.45)
    p = h.HestonParams(
        v0=rng.uniform(0.025, 0.075), kappa=rng.uniform(1.2, 2.8),
        theta=rng.uniform(0.025, 0.075), sigma=rng.uniform(0.25, 0.55),
        rho=rng.uniform(-0.75, -0.25),
    )
    opt = h.VanillaOption(h.OptionType.Call, float(K), float(T))
    a = time.perf_counter(); true_px[i] = fp.price(opt, mkt, p); t_fourier += time.perf_counter() - a
    b = time.perf_counter(); gpr_px[i] = gpr.price(opt, mkt, p); t_gpr += time.perf_counter() - b

abs_err = np.abs(gpr_px - true_px)
print(f"out-of-sample abs error   mean = {abs_err.mean():.4e}   max = {abs_err.max():.4e}")
print(f"avg price time   Fourier = {1e6 * t_fourier / n_test:.1f} us   "
      f"GPR = {1e6 * t_gpr / n_test:.1f} us   (speed-up x{t_fourier / max(t_gpr, 1e-12):.1f})")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
lo, hi = float(true_px.min()), float(true_px.max())
axL.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1, label="y = x")
axL.scatter(true_px, gpr_px, s=10, alpha=0.6, color="C0", label="test points")
axL.set_xlabel("Fourier price (truth)")
axL.set_ylabel("GPR price")
axL.set_title("GPR vs Fourier (out-of-sample)")
axL.legend()
axL.grid(alpha=0.3)

axR.hist(abs_err, bins=30, color="C3", alpha=0.8)
axR.set_xlabel("Absolute price error")
axR.set_ylabel("Count")
axR.set_title(f"Error distribution (mean {abs_err.mean():.1e})")
axR.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG / "gpr_surrogate.png", dpi=130)
print("saved", IMG / "gpr_surrogate.png")
plt.show()
