#include "third_party/doctest/doctest.h"

#include <cmath>
#include <vector>

#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "pricers/ml/gpr_pricer.hpp"
#include "products/vanilla_option.hpp"
#include "utils/rng.hpp"

using namespace heston;

namespace {
GprPricer::Config make_config() {
  GprPricer::Config cfg;
  cfg.reference = Market{100.0, 0.02, 0.01};
  cfg.box = GprPricer::Box{/*K*/ 80.0,  120.0,
                           /*T*/ 0.30,  1.50,
                           /*v0*/ 0.02, 0.08,
                           /*kappa*/ 1.0, 3.0,
                           /*theta*/ 0.02, 0.08,
                           /*sigma*/ 0.20, 0.60,
                           /*rho*/ -0.80, -0.20};
  cfg.n_train = 500;
  cfg.sample_seed = 12345;
  cfg.gp.optimize_hyperparameters = true;
  cfg.gp.max_opt_iter = 45;
  return cfg;
}
}  // namespace

// Training the surrogate is the expensive step, so this single test case trains
// once and then exercises every behaviour against that one trained pricer.
TEST_CASE("GprPricer: trained surrogate behaviour") {
  HestonFourierPricer truth;
  GprPricer gpr(make_config());
  gpr.train(truth);
  REQUIRE(gpr.is_trained());

  const Market mkt{100.0, 0.02, 0.01};

  SUBCASE("out-of-sample prices track the Fourier engine") {
    Rng rng(999);
    double sum_abs = 0.0, max_abs = 0.0;
    const int n_test = 40;
    for (int i = 0; i < n_test; ++i) {
      const double K = rng.uniform(82.0, 118.0);
      const double T = rng.uniform(0.35, 1.45);
      const HestonParams p(rng.uniform(0.025, 0.075), rng.uniform(1.2, 2.8),
                           rng.uniform(0.025, 0.075), rng.uniform(0.25, 0.55),
                           rng.uniform(-0.75, -0.25));
      const VanillaOption call{OptionType::Call, K, T};
      const double err = std::abs(gpr.price(call, mkt, p) - truth.price(call, mkt, p));
      sum_abs += err;
      if (err > max_abs) max_abs = err;
    }
    const double mean_abs = sum_abs / n_test;
    MESSAGE("GprPricer mean abs error = " << mean_abs << ", max abs error = " << max_abs);
    CHECK(mean_abs < 0.1);  // spot = 100; well under 0.1% average price error in practice
    CHECK(max_abs < 0.5);
  }

  SUBCASE("puts satisfy put-call parity by construction") {
    const HestonParams p(0.04, 2.0, 0.04, 0.4, -0.5);
    for (const double K : {85.0, 100.0, 115.0}) {
      const double c = gpr.price(VanillaOption{OptionType::Call, K, 1.0}, mkt, p);
      const double pp = gpr.price(VanillaOption{OptionType::Put, K, 1.0}, mkt, p);
      const double disc_r = std::exp(-mkt.r * 1.0);
      const double disc_q = std::exp(-mkt.q * 1.0);
      const double residual = (c - pp) - (mkt.s0 * disc_q - K * disc_r);
      CHECK(residual == doctest::Approx(0.0).epsilon(1e-9));
    }
  }

  SUBCASE("diagnostics report predictive std and flag extrapolation") {
    const HestonParams p_in(0.04, 2.0, 0.04, 0.4, -0.5);
    (void)gpr.price(VanillaOption{OptionType::Call, 100.0, 1.0}, mkt, p_in);
    CHECK(gpr.diagnostics().stdError >= 0.0);
    CHECK(gpr.diagnostics().note.empty());

    const HestonParams p_out(0.04, 8.0, 0.04, 0.4, -0.5);  // kappa above the trained range
    (void)gpr.price(VanillaOption{OptionType::Call, 100.0, 1.0}, mkt, p_out);
    CHECK(!gpr.diagnostics().note.empty());
  }

  SUBCASE("wrong market is rejected") {
    const HestonParams p(0.04, 2.0, 0.04, 0.4, -0.5);
    const Market wrong{105.0, 0.02, 0.01};  // different spot than the reference
    CHECK_THROWS_AS((void)gpr.price(VanillaOption{OptionType::Call, 100.0, 1.0}, wrong, p),
                    InvalidInput);
  }
}

TEST_CASE("GprPricer: pricing before training throws") {
  GprPricer gpr(make_config());  // not trained
  const Market mkt{100.0, 0.02, 0.01};
  const HestonParams p(0.04, 2.0, 0.04, 0.4, -0.5);
  CHECK_THROWS_AS((void)gpr.price(VanillaOption{OptionType::Call, 100.0, 1.0}, mkt, p), InvalidInput);
}
