#include "third_party/doctest/doctest.h"

#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "vol/black_scholes.hpp"

#include <cmath>
#include <initializer_list>

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

// Absolute price check (guards the e^{-rT} discounting): as vol-of-vol -> 0 and
// v0 == theta, the variance is deterministically constant, so the Heston price
// must collapse onto Black–Scholes with vol = sqrt(theta). A non-zero rate is
// used on purpose so a missing discount factor would show up.
TEST_CASE("Fourier pricer: reduces to Black-Scholes in the low-vol-of-vol limit") {
  using namespace heston;

  Market m{100.0, 0.03, 0.01};

  HestonFourierPricer::Settings s;
  s.u_max = 300.0;
  s.n_intervals_even = 20'000;
  HestonFourierPricer pricer(s);

  const double theta = 0.04;               // => BS vol = 0.20
  HestonParams p(theta, 1.5, theta, 1e-3, 0.0);

  for (const double K : {80.0, 100.0, 125.0}) {
    for (const auto type : {OptionType::Call, OptionType::Put}) {
      VanillaOption opt{type, K, 1.0};
      const double heston_px = pricer.price(opt, m, p);
      const double bs_px     = black_scholes_price(opt, m, std::sqrt(theta));
      CHECK(heston_px == doctest::Approx(bs_px).epsilon(1e-3));
    }
  }
}