#include "third_party/doctest/doctest.h"

#include <cmath>
#include <vector>

#include "calibration/calibrator.hpp"
#include "calibration/iv_quote.hpp"
#include "calibration/objective.hpp"
#include "models/heston/market.hpp"
#include "models/heston/heston_params.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "utils/errors.hpp"
#include "vol/implied_vol.hpp"

using namespace heston;

static Market standard_market() {
  return Market{100.0, 0.02, 0.01};
}

static HestonParams true_params() {
  HestonParams p(0.04,1.5,0.04,0.6,-0.7);
  return p;
}

static HestonParams rough_guess() {
  HestonParams p(0.02,1.0,0.06,0.4,-0.3);
  return p;
}

TEST_CASE("Calibration: synthetic IV recovery reduces loss a lot") {
  const Market mkt = standard_market();
  const HestonParams p_true = true_params();

  // Build synthetic "market" IVs from the Fourier pricer (no noise for baseline test).
  HestonFourierPricer::Settings fps;
  fps.alpha = 1.5;
  fps.u_max = 200.0;
  fps.n_intervals_even = 10'000;

  HestonFourierPricer fourier(fps);

  std::vector<IvQuote> quotes;
  const std::vector<double> maturities = {0.25, 0.5, 1.0};
  const std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};

  ImpliedVolSettings ivs;
  ivs.tol_abs = 1e-10;

  for (double T : maturities) {
    for (double K : strikes) {
      VanillaOption opt;
      opt.type = OptionType::Call;
      opt.K = K;
      opt.T = T;

      const double px = fourier.price(opt, mkt, p_true);
      const double iv = implied_vol_black_scholes(opt, mkt, px, ivs);

      IvQuote q;
      q.opt = opt;
      q.iv = iv;
      q.weight = 1.0;
      quotes.push_back(q);
    }
  }

  // Initial guess
  HestonParams p0 = rough_guess();

  // Evaluate initial loss
  const auto init_obj = evaluate_iv_objective(quotes, mkt, p0, fourier);

  // Calibrate
  CalibrationSettings cs;
  cs.fourier_pricer = fourier;
  cs.nm.max_iter = 300;     // baseline
  cs.nm.tol_f = 1e-10;
  cs.nm.tol_x = 1e-8;

  const auto res = calibrate_heston_to_iv(quotes, mkt, p0, cs);

  CHECK(std::isfinite(init_obj.loss));
  CHECK(std::isfinite(res.loss));

  // We don't require perfect parameter recovery (objective surface can be flat),
  // but we DO require the loss to drop dramatically on clean synthetic data.
  CHECK(res.loss < 1e-6);
  CHECK(res.loss < init_obj.loss * 1e-2);
}

TEST_CASE("Calibration: returns finite params and rho in (-1,1)") {
  const Market mkt = standard_market();
  const HestonParams p_true = true_params();

  HestonFourierPricer fourier;

  // Single maturity surface (still enough)
  std::vector<IvQuote> quotes;
  const double T = 1.0;
  for (double K : {80.0, 90.0, 100.0, 110.0, 120.0}) {
    VanillaOption opt;
    opt.type = OptionType::Call;
    opt.K = K;
    opt.T = T;

    const double px = fourier.price(opt, mkt, p_true);
    const double iv = implied_vol_black_scholes(opt, mkt, px);

    IvQuote q;
    q.opt = opt;
    q.iv = iv;
    quotes.push_back(q);
  }

  CalibrationSettings cs;
  cs.fourier_pricer = fourier;
  cs.nm.max_iter = 250;

  const auto res = calibrate_heston_to_iv(quotes, mkt, rough_guess(), cs);

  CHECK(res.params.is_finite());
  CHECK(res.params.rho > -1.0);
  CHECK(res.params.rho < 1.0);
}