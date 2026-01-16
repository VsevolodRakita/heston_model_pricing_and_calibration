#include "third_party/doctest/doctest.h"

#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"

#include <cmath>

TEST_CASE("Fourier pricer: put-call parity") {
  using namespace heston;

  Market m{100.0, 0.02, 0.01};
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  HestonFourierPricer::Settings s;
  s.alpha = 1.5;
  s.u_max = 250.0;
  s.n_intervals_even = 12'000;

  HestonFourierPricer pricer(s);

  VanillaOption call{OptionType::Call, 100.0, 1.0};
  VanillaOption put {OptionType::Put,  100.0, 1.0};

  const double c = pricer.price(call, m, p);
  const double pp = pricer.price(put, m, p);

  const double disc_r = std::exp(-m.r * call.T);
  const double disc_q = std::exp(-m.q * call.T);

  const double residual = (c - pp) - (m.s0 * disc_q - call.K * disc_r);
  CHECK(residual == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("Fourier pricer: call decreases with strike") {
  using namespace heston;

  Market m{100.0, 0.02, 0.0};
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  HestonFourierPricer pricer;

  VanillaOption c1{OptionType::Call, 80.0,  1.0};
  VanillaOption c2{OptionType::Call, 100.0, 1.0};
  VanillaOption c3{OptionType::Call, 120.0, 1.0};

  const double p1 = pricer.price(c1, m, p);
  const double p2 = pricer.price(c2, m, p);
  const double p3 = pricer.price(c3, m, p);

  CHECK(p1 + 1e-10 >= p2);
  CHECK(p2 + 1e-10 >= p3);
  CHECK(p3 >= 0.0);
}