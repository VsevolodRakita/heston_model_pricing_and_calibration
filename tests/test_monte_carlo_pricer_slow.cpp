#include "third_party/doctest/doctest.h"

#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "pricers/mc/heston_monte_carlo_pricer.hpp"

#include <cmath>

using namespace heston;

// ------------------------------------------------------------
// Helper: standard Heston test configuration
// ------------------------------------------------------------
static HestonParams standard_params() {
  return HestonParams(
      0.04,  // v0
      2.0,   // kappa
      0.04,  // theta
      0.5,   // sigma
     -0.7    // rho
  );
}

static Market standard_market() {
  return Market{100.0, 0.02, 0.01};
}

// ------------------------------------------------------------
// (1) Martingale test: E[S_T] = S0 * exp((r-q)T)
// ------------------------------------------------------------
/*TEST_CASE("MC (QE): discounted stock is a martingale") {
  Market m = standard_market();
  HestonParams p = standard_params();

  VanillaOption call{OptionType::Call, 100.0, 1.0};

  HestonMonteCarloPricer::Settings ms;
  ms.n_paths = 300'000;
  ms.n_steps = 250;
  ms.seed = 123;
  ms.antithetic = true;
  ms.variance_scheme = HestonMonteCarloPricer::VarianceScheme::QE;

  HestonMonteCarloPricer mc(ms);

  // Price a forward payoff: S_T
  auto res = mc.price_with_error(call, m, p);

  const double expected_ST = m.s0 * std::exp((m.r - m.q) * call.T);
  const double implied_ST =
      (res.price + call.K * std::exp(-m.r * call.T)) * std::exp(m.r * call.T);

  const double tol = 6.0 * res.std_error + 0.5;
  CHECK(std::abs(implied_ST - expected_ST) <= 5.0 * tol);
}*/

TEST_CASE("MC (QE): discounted stock is approximately a martingale (via parity)") {
  Market m = standard_market();
  HestonParams p = standard_params();

  const double K = 100.0;
  const double T = 1.0;

  VanillaOption call{OptionType::Call, K, T};
  VanillaOption put {OptionType::Put,  K, T};

  HestonMonteCarloPricer::Settings ms;
  ms.n_paths = 300'000;
  ms.n_steps = 250;
  ms.seed = 123;
  ms.antithetic = true;
  ms.variance_scheme = HestonMonteCarloPricer::VarianceScheme::QE;

  HestonMonteCarloPricer mc(ms);

  // Estimate C and P
  const auto c = mc.price_with_error(call, m, p);
  const auto pp = mc.price_with_error(put, m, p);

  // Infer E[S_T] from parity: C - P = e^{-rT}(E[S_T] - K)
  const double implied_ST = std::exp(m.r * T) * (c.price - pp.price) + K;

  // True E[S_T] under risk-neutral measure:
  const double expected_ST = m.s0 * std::exp((m.r - m.q) * T);

  // Conservative tolerance: combine both standard errors + small discretization buffer
  const double se_combo = c.std_error + pp.std_error;
  const double tol = 8.0 * se_combo + 0.75; // 8-sigma + buffer (slow test)

  CHECK(std::abs(implied_ST - expected_ST) <= tol);
}

// ------------------------------------------------------------
// (2) Put–call parity (strong version)
// ------------------------------------------------------------
TEST_CASE("MC (QE): put-call parity holds (strong check)") {
  Market m = standard_market();
  HestonParams p = standard_params();

  VanillaOption call{OptionType::Call, 100.0, 1.0};
  VanillaOption put {OptionType::Put,  100.0, 1.0};

  HestonMonteCarloPricer::Settings ms;
  ms.n_paths = 300'000;
  ms.n_steps = 250;
  ms.seed = 321;
  ms.antithetic = true;
  ms.variance_scheme = HestonMonteCarloPricer::VarianceScheme::QE;

  HestonMonteCarloPricer mc(ms);

  auto c = mc.price_with_error(call, m, p);
  auto p_res = mc.price_with_error(put, m, p);

  const double disc_r = std::exp(-m.r * call.T);
  const double disc_q = std::exp(-m.q * call.T);

  const double parity =
      (c.price - p_res.price) - (m.s0 * disc_q - call.K * disc_r);

  const double tol =
      6.0 * (c.std_error + p_res.std_error) + 0.5;

  CHECK(std::abs(parity) <= tol);
}

// ------------------------------------------------------------
// (3) Zero vol-of-vol sanity check (Heston -> Black-Scholes)
// ------------------------------------------------------------
TEST_CASE("MC (QE): zero vol-of-vol reduces to Black-Scholes") {
  Market m = standard_market();

  // sigma = 0 => variance deterministic
  HestonParams p(
      0.04,  // v0
      2.0,   // kappa
      0.04,  // theta
      1e-6,   // sigma
      0.0    // rho
  );

  VanillaOption call{OptionType::Call, 100.0, 1.0};

  HestonFourierPricer fourier;
  const double ref = fourier.price(call, m, p);

  HestonMonteCarloPricer::Settings ms;
  ms.n_paths = 250'000;
  ms.n_steps = 200;
  ms.seed = 999;
  ms.antithetic = true;
  ms.variance_scheme = HestonMonteCarloPricer::VarianceScheme::QE;

  HestonMonteCarloPricer mc(ms);

  auto res = mc.price_with_error(call, m, p);

  const double diff = std::abs(res.price - ref);
  CHECK(diff <= 5.0 * res.std_error + 0.3);
}

// ------------------------------------------------------------
// (4) Convergence sanity: more paths => closer to Fourier
// ------------------------------------------------------------
TEST_CASE("MC (QE): convergence improves with more paths (CI-based)") {
  Market m = standard_market();
  HestonParams p = standard_params();
  VanillaOption call{OptionType::Call, 100.0, 1.0};

  HestonFourierPricer fourier;
  const double ref = fourier.price(call, m, p);

  // Base settings
  HestonMonteCarloPricer::Settings ms1;
  ms1.n_paths = 50'000;
  ms1.n_steps = 200;
  ms1.seed = 42;
  ms1.antithetic = true;
  ms1.variance_scheme = HestonMonteCarloPricer::VarianceScheme::QE;

  // 4x more paths (same seed => nested sampling would require code changes,
  // so we treat it as a separate estimate with reduced standard error).
  HestonMonteCarloPricer::Settings ms2 = ms1;
  ms2.n_paths = 200'000;

  HestonMonteCarloPricer mc1(ms1);
  HestonMonteCarloPricer mc2(ms2);

  const auto r1 = mc1.price_with_error(call, m, p);
  const auto r2 = mc2.price_with_error(call, m, p);

  const double err1 = std::abs(r1.price - ref);
  const double err2 = std::abs(r2.price - ref);

  // Allow non-monotonicity, but err2 shouldn't be meaningfully worse
  // given its (smaller) Monte Carlo noise.
  const double k = 4.0;  // ~4-sigma cushion, stable in CI
  CHECK(err2 <= err1 + k * r2.std_error);
}