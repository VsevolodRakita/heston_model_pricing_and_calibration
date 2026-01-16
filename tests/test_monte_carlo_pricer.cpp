#include "third_party/doctest/doctest.h"

#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "pricers/mc/heston_monte_carlo_pricer.hpp"

#include <cmath>

TEST_CASE("Monte Carlo (QE): agrees with Fourier within tolerance (call)") {
  using namespace heston;

  Market m{100.0, 0.02, 0.01};
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);
  VanillaOption call{OptionType::Call, 100.0, 1.0};

  HestonFourierPricer::Settings fs;
  fs.alpha = 1.5;
  fs.u_max = 250.0;
  fs.n_intervals_even = 12'000;
  HestonFourierPricer fourier(fs);

  HestonMonteCarloPricer::Settings ms;
  ms.n_paths = 200'000;
  ms.n_steps = 250;
  ms.seed = 123;
  ms.antithetic = true;
  ms.variance_scheme = HestonMonteCarloPricer::VarianceScheme::QE;
  ms.psi_c = 1.5;

  HestonMonteCarloPricer mc(ms);

  const double ref = fourier.price(call, m, p);
  const auto res = mc.price_with_error(call, m, p);

  const double diff = std::abs(res.price - ref);

  // sampling noise + small discretization cushion
  const double discretization_buffer = 0.35;
  CHECK(diff <= 4.0 * res.std_error + discretization_buffer);
}

TEST_CASE("Monte Carlo (QE): put-call parity holds approximately") {
  using namespace heston;

  Market m{100.0, 0.02, 0.01};
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  VanillaOption call{OptionType::Call, 100.0, 1.0};
  VanillaOption put {OptionType::Put,  100.0, 1.0};

  HestonMonteCarloPricer::Settings ms;
  ms.n_paths = 250'000;
  ms.n_steps = 250;
  ms.seed = 7;
  ms.antithetic = true;
  ms.variance_scheme = HestonMonteCarloPricer::VarianceScheme::QE;
  ms.psi_c = 1.5;

  HestonMonteCarloPricer mc(ms);

  const auto c = mc.price_with_error(call, m, p);
  const auto pp = mc.price_with_error(put, m, p);

  const double disc_r = std::exp(-m.r * call.T);
  const double disc_q = std::exp(-m.q * call.T);

  const double parity = (c.price - pp.price) - (m.s0 * disc_q - call.K * disc_r);

  const double tol = 6.0 * (c.std_error + pp.std_error) + 0.35;
  CHECK(std::abs(parity) <= tol);
}