#include "third_party/doctest/doctest.h"

#include <cmath>
#include <vector>
#include <algorithm>

#include "calibration/calibrator.hpp"
#include "calibration/iv_quote.hpp"
#include "calibration/objective.hpp"
#include "models/heston/market.hpp"
#include "models/heston/heston_params.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "utils/errors.hpp"
#include "vol/implied_vol.hpp"


using namespace heston;

// -------------------- helpers --------------------

static Market standard_market() {
  return Market{100.0, 0.02, 0.01};
}

static HestonParams base_params() {
  HestonParams p(0.04, 1.5, 0.04, 0.6, -0.7);
  return p;
}

static HestonParams rough_guess() {
  HestonParams p(0.02, 0.8, 0.06, 0.4, -0.2);
  return p;
}

static std::vector<IvQuote> make_surface(
    const HestonParams& p,
    const Market& mkt,
    const std::vector<double>& maturities,
    const std::vector<double>& strikes) {

  HestonFourierPricer pricer;
  std::vector<IvQuote> quotes;

  for (double T : maturities) {
    for (double K : strikes) {
      VanillaOption opt{OptionType::Call, K, T};
      const double px = pricer.price(opt, mkt, p);
      const double iv = implied_vol_black_scholes(opt, mkt, px);
      quotes.push_back({opt, iv, 1.0});
    }
  }
  return quotes;
}

// -------------------- TESTS --------------------

// 1) Loss strictly decreases vs initial guess
TEST_CASE("Calibration: loss strictly decreases from initial guess") {
  Market mkt = standard_market();
  auto quotes = make_surface(base_params(), mkt, {0.5, 1.0}, {90, 100, 110});

  HestonFourierPricer pricer;
  const auto init_obj = evaluate_iv_objective(quotes, mkt, rough_guess(), pricer);

  CalibrationSettings cs;
  cs.fourier_pricer = pricer;
  cs.nm.max_iter = 200;

  const auto res = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);

  CHECK(res.loss < init_obj.loss);
}

// 2) Calibration is invariant to quote ordering
TEST_CASE("Calibration: invariant to quote ordering") {
  Market mkt = standard_market();
  auto quotes = make_surface(base_params(), mkt, {0.5, 1.0}, {90, 100, 110});

  auto quotes_reversed = quotes;
  std::reverse(quotes_reversed.begin(), quotes_reversed.end());

  CalibrationSettings cs;
  cs.nm.max_iter = 200;

  const auto r1 = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);
  const auto r2 = calibrate_heston_to_iv(quotes_reversed, mkt, rough_guess(), cs);

  CHECK(r1.loss == doctest::Approx(r2.loss).epsilon(1e-6));
}

// 3) rho remains strictly in (-1,1)
TEST_CASE("Calibration: rho remains in (-1,1)") {
  Market mkt = standard_market();
  auto quotes = make_surface(base_params(), mkt, {1.0}, {80, 100, 120});

  CalibrationSettings cs;
  cs.nm.max_iter = 200;

  const auto res = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);

  CHECK(res.params.rho > -1.0);
  CHECK(res.params.rho < 1.0);
}

// 4) Calibration works with deep ITM/OTM strikes
TEST_CASE("Calibration: handles deep ITM/OTM strikes") {
  Market mkt = standard_market();
  auto quotes = make_surface(base_params(), mkt, {1.0}, {50, 70, 130, 150});

  CalibrationSettings cs;
  cs.nm.max_iter = 250;

  const auto res = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);

  CHECK(std::isfinite(res.loss));
}

// 5) Calibration with single maturity (ill-conditioned but solvable)
TEST_CASE("Calibration: single maturity surface is solvable") {
  Market mkt = standard_market();
  auto quotes = make_surface(base_params(), mkt, {1.0}, {85, 95, 105, 115});

  CalibrationSettings cs;
  cs.nm.max_iter = 250;

  const auto res = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);

  CHECK(res.loss < 1e-4);
}

// 6) Calibration robust to small noise in IVs
TEST_CASE("Calibration: small IV noise does not break convergence") {
  Market mkt = standard_market();
  auto quotes = make_surface(base_params(), mkt, {0.5, 1.0}, {90, 100, 110});

  for (auto& q : quotes) {
    q.iv *= 1.0 + 0.01 * std::sin(q.opt.K); // deterministic noise
  }

  CalibrationSettings cs;
  cs.nm.max_iter = 300;

  const auto res = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);

  CHECK(std::isfinite(res.loss));
  CHECK(res.loss < 1e-2);
}

// 7) Increasing optimizer budget does not worsen loss
TEST_CASE("Calibration: more iterations do not worsen loss") {
  Market mkt = standard_market();
  auto quotes = make_surface(base_params(), mkt, {0.5, 1.0}, {90, 100, 110});

  CalibrationSettings cs1;
  cs1.nm.max_iter = 100;

  CalibrationSettings cs2;
  cs2.nm.max_iter = 300;

  const auto r1 = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs1);
  const auto r2 = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs2);

  CHECK(r2.loss <= r1.loss);
}

// 8) Calibration stable for low vol-of-vol
TEST_CASE("Calibration: low vol-of-vol regime") {
  HestonParams p = base_params();
  p.sigma = 0.05;

  Market mkt = standard_market();
  auto quotes = make_surface(p, mkt, {0.5, 1.0}, {90, 100, 110});

  CalibrationSettings cs;
  cs.nm.max_iter = 250;

  const auto res = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);

  CHECK(std::isfinite(res.loss));
}

// 9) Calibration stable for near-zero correlation
TEST_CASE("Calibration: near-zero correlation regime") {
  HestonParams p = base_params();
  p.rho = -0.05;

  Market mkt = standard_market();
  auto quotes = make_surface(p, mkt, {0.5, 1.0}, {90, 100, 110});

  CalibrationSettings cs;
  cs.nm.max_iter = 250;

  const auto res = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);

  CHECK(std::isfinite(res.loss));
}
