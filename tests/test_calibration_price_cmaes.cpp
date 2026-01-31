#include "third_party/doctest/doctest.h"

#include <vector>
#include <algorithm>

#include "calibration/calibrator_price.hpp"
#include "calibration/prepare_quotes.hpp"
#include "models/heston/market.hpp"
#include "models/heston/heston_params.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "vol/implied_vol.hpp"

using namespace heston;

static Market standard_market() { return Market{100.0, 0.02, 0.01}; }

static HestonParams true_params() {
  return HestonParams{0.04, 1.5, 0.04, 0.6, -0.7};
}

static HestonParams rough_guess() {
  return HestonParams{0.02, 0.8, 0.06, 0.4, -0.2};
}

TEST_CASE("CMA-ES price calibration: synthetic IV surface fits well") {
  const Market mkt = standard_market();
  const HestonParams p_true = true_params();

  // Build synthetic IV quotes from Fourier, then convert to market prices once
  HestonFourierPricer pricer;

  std::vector<IvQuote> ivq;
  for (double T : {0.25, 0.5, 1.0}) {
    for (double K : {80.0, 90.0, 100.0, 110.0, 120.0}) {
      VanillaOption opt{OptionType::Call, K, T};
      const double px = pricer.price(opt, mkt, p_true);
      const double iv = implied_vol_black_scholes(opt, mkt, px);
      ivq.push_back({opt, iv, 1.0});
    }
  }

  const auto pq = prepare_price_quotes_from_iv(ivq, mkt);

  PriceCalibrationSettings cs;
  cs.pricer = pricer;
  cs.cma.max_iter = 200;
  cs.cma.sigma0 = 0.4;
  cs.cma.seed = 123;
  cs.enable_trace = true;
  cs.trace_every = 10;

  const auto rep = calibrate_heston_to_prices_cmaes(pq, mkt, rough_guess(), cs);
  //const auto init_obj = evaluate_price_objective(pq, mkt, rough_guess(), pricer);

  CHECK(rep.params.is_finite());
  //CHECK(rep.loss < init_obj.loss);
  const double rmse = std::sqrt(rep.loss / static_cast<double>(pq.size()));
  CHECK(rmse < 0.25);
  CHECK(!rep.trace.empty());
  CHECK(rep.final_residuals.size() == pq.size());
}

TEST_CASE("CMA-ES price calibration: ordering of quotes does not matter (loss)") {
  const Market mkt = standard_market();
  const HestonParams p_true = true_params();

  HestonFourierPricer pricer;

  std::vector<IvQuote> ivq;
  for (double T : {0.5, 1.0}) {
    for (double K : {90.0, 100.0, 110.0}) {
      VanillaOption opt{OptionType::Call, K, T};
      const double px = pricer.price(opt, mkt, p_true);
      const double iv = implied_vol_black_scholes(opt, mkt, px);
      ivq.push_back({opt, iv, 1.0});
    }
  }

  auto pq1 = prepare_price_quotes_from_iv(ivq, mkt);
  auto pq2 = pq1;
  std::reverse(pq2.begin(), pq2.end());

  PriceCalibrationSettings cs;
  cs.pricer = pricer;
  cs.cma.max_iter = 150;
  cs.cma.sigma0 = 0.4;
  cs.cma.seed = 123;
  cs.enable_trace = false;

  const auto r1 = calibrate_heston_to_prices_cmaes(pq1, mkt, rough_guess(), cs);
  const auto r2 = calibrate_heston_to_prices_cmaes(pq2, mkt, rough_guess(), cs);

  CHECK(r1.loss == doctest::Approx(r2.loss).epsilon(1e-6));
}