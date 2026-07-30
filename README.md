# Heston Model — Pricing & Calibration

A modern **C++20** library for pricing European options under the **Heston stochastic-volatility model** and calibrating the model to market data. It ships two independent pricing engines (a Carr–Madan Fourier pricer and a Monte Carlo simulator), Black–Scholes / implied-vol utilities, and two from-scratch optimizers (Nelder–Mead and CMA-ES) driving two calibration workflows.

No third-party runtime dependencies — only the C++ standard library. The unit tests use a vendored copy of [doctest](https://github.com/doctest/doctest) (header-only, included in-tree).

---

## The model

The Heston model lets the instantaneous variance of the underlying be random and mean-reverting, rather than the constant volatility assumed by Black–Scholes. Under the risk-neutral measure:

```
dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t^v
d<W^S, W^v>_t = rho dt
```

It is described by five parameters (`heston::HestonParams`):

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| `v0`      | v₀     | initial instantaneous variance |
| `kappa`   | κ      | mean-reversion speed of variance |
| `theta`   | θ      | long-run variance |
| `sigma`   | σ      | volatility of variance ("vol of vol") |
| `rho`     | ρ      | correlation between the price and variance shocks (drives the skew) |

`HestonParams` also validates itself (`is_valid_basic()`) and reports the **Feller condition** `2·κ·θ ≥ σ²` via `satisfies_feller()`, which guarantees the variance process stays strictly positive.

---

## Features

- **Heston characteristic function** — "Little Heston Trap" formulation for numerical stability (`HestonCharacteristicFunction`).
- **Fourier pricer** — Carr–Madan damped-transform pricing with Simpson integration (`HestonFourierPricer`). Fast and deterministic; the workhorse for calibration.
- **Monte Carlo pricer** — full-truncation Euler, Milstein, and Andersen **QE** variance schemes, with antithetic variates and standard-error reporting (`HestonMonteCarloPricer`).
- **Volatility toolkit** — Black–Scholes price & vega, and a robust implied-vol solver (bracketing + bisection with a Newton polish).
- **Calibration** — fit the five parameters to a set of market quotes:
  - to **implied-vol** quotes via Nelder–Mead (`calibrate_heston_to_iv`),
  - to **price** quotes via CMA-ES (`calibrate_heston_to_prices_cmaes`), with an optimization trace and final residuals.
- **Unconstrained reparameterization** — calibration optimizes in ℝ⁵ using `exp`/`tanh` transforms (`param_transform.hpp`), so positivity of `v0, κ, θ, σ` and `ρ ∈ (-1, 1)` hold automatically.
- **Typed errors** — all failures throw `heston::InvalidInput` or `heston::NumericFailure` (both derive from `heston::HestonError : std::runtime_error`).

---

## Requirements

- A **C++20** compiler (MSVC 2019+, GCC 10+, or Clang 12+)
- **CMake ≥ 3.20**

## Building

```bash
git clone https://github.com/VsevolodRakita/heston_model_pricing_and_calibration.git
cd heston_model_pricing_and_calibration
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces the static library **`heston_core`** plus the test executables. Building the tests is on by default; turn it off with `-DHESTON_BUILD_TESTS=OFF`.

## Running the tests

```bash
cd build
ctest --output-on-failure
```

Tests are labelled, so you can select subsets — for example, skip the slow Monte Carlo / calibration suites:

```bash
ctest -L fast                 # fast tests only
ctest -LE slow                # everything except slow tests
```

Available labels include `fast`, `slow`, `vol`, `stability`, `monte_carlo`, and `calibration`.

---

## Project layout

```
include/                     # public headers (installed interface)
  models/heston/             # HestonParams, Market, characteristic function
  products/                  # VanillaOption, OptionType
  pricers/                   # IVanillaPricer interface, FFT + Monte Carlo pricers
  vol/                       # Black–Scholes, implied vol
  optimization/              # Nelder–Mead, CMA-ES
  calibration/               # quotes, objectives, calibrators, reports
  utils/                     # errors, numerics, stats, rng
src/                         # implementations (compiled into heston_core)
tests/                       # doctest-based unit tests (+ vendored doctest.h)
```

---

## Usage

### Price a European option (Fourier)

```cpp
#include "models/heston/market.hpp"
#include "models/heston/heston_params.hpp"
#include "products/vanilla_option.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"

using namespace heston;

Market       mkt{/*s0=*/100.0, /*r=*/0.02, /*q=*/0.0};
HestonParams params{/*v0=*/0.04, /*kappa=*/1.5, /*theta=*/0.04,
                    /*sigma=*/0.5, /*rho=*/-0.7};
VanillaOption call{OptionType::Call, /*K=*/100.0, /*T=*/1.0};

HestonFourierPricer pricer;
double price = pricer.price(call, mkt, params);
```

### Cross-check with Monte Carlo

```cpp
#include "pricers/mc/heston_monte_carlo_pricer.hpp"

HestonMonteCarloPricer mc;                       // QE scheme, antithetic, by default
auto res = mc.price_with_error(call, mkt, params);
// res.price, res.std_error, res.n_used
```

Both pricers implement the common `IVanillaPricer` interface (`price`, `priceBatch`), so calibration and client code can swap engines transparently.

### Recover implied volatility

```cpp
#include "vol/implied_vol.hpp"

double iv = implied_vol_black_scholes(call, mkt, price);
```

### Calibrate to implied-vol quotes (Nelder–Mead)

```cpp
#include "calibration/iv_quote.hpp"
#include "calibration/calibrator.hpp"

std::vector<IvQuote> quotes = /* market {option, iv} pairs */;

HestonParams        guess{0.04, 1.0, 0.04, 0.5, -0.5};
CalibrationSettings settings;                    // default Nelder–Mead + Fourier pricer
CalibrationResult   fit = calibrate_heston_to_iv(quotes, mkt, guess, settings);
// fit.params, fit.loss, fit.iters
```

### Calibrate to price quotes (CMA-ES)

```cpp
#include "calibration/price_quote.hpp"
#include "calibration/calibrator_price.hpp"

std::vector<PriceQuote> quotes = /* market {option, price} pairs */;

HestonParams             guess{0.04, 1.0, 0.04, 0.5, -0.5};
PriceCalibrationSettings settings;               // default CMA-ES + Fourier pricer
CalibrationReport        report =
    calibrate_heston_to_prices_cmaes(quotes, mkt, guess, settings);
// report.params, report.loss, report.iters,
// report.trace (per-iteration path), report.final_residuals
```

> Calibration runs on the deterministic Fourier pricer and searches over the unconstrained ℝ⁵ parameterization, keeping every candidate a valid Heston parameter set.

---

## Numerical notes & conventions

- **Rates and yields** are continuously compounded; `q` is a continuous dividend yield / cost of carry.
- **Maturities** `T` are in years.
- The Fourier pricer values a call directly and obtains puts by **put–call parity**; optional guardrails floor the price and cap it at a multiple of spot.
- The QE Monte Carlo scheme follows Andersen's quadratic-exponential moment-matching, with the Euler and Milstein schemes using full truncation of the variance.
- Inputs are validated up front; invalid inputs or numerical breakdowns raise the typed exceptions above rather than returning silent `NaN`s.

---

## License

Released under the **MIT License** — © 2026 Vsevolod Rakita, Ph.D. See [LICENSE](LICENSE).
