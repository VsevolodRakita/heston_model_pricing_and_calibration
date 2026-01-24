#include "third_party/doctest/doctest.h"

#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "models/heston/characteristic_function.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "products/vanilla_option.hpp"

#include "bs_helpers.hpp"

#include <complex>
#include <cmath>

using namespace heston;

static Market mkt() { return Market{100.0, 0.02, 0.01}; }

TEST_CASE("CF stability: log_spot_cf returns finite values on a grid") {
  Market m = mkt();
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  HestonCharacteristicFunction cf({true}); // use_little_trap = true

  const double T = 1.0;
  for (double re_u : {0.0, 0.1, 1.0, 5.0, 10.0, 50.0}) {
    for (double im_u : {0.0, -0.5, -1.0}) {
      std::complex<double> u(re_u, im_u);
      auto val = cf.log_spot_cf(u, T, m, p);
      CHECK(std::isfinite(val.real()));
      CHECK(std::isfinite(val.imag()));
    }
  }
}

TEST_CASE("Fourier stability: price is finite and within loose arbitrage bounds") {
  Market m = mkt();
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  VanillaOption call{OptionType::Call, 100.0, 1.0};
  HestonFourierPricer pricer;

  const double px = pricer.price(call, m, p);
  CHECK(std::isfinite(px));
  CHECK(px >= 0.0);

  // Upper bound for a call: S0*e^{-qT}
  const double upper = m.s0 * std::exp(-m.q * call.T);
  CHECK(px <= upper + 1e-6);
}

TEST_CASE("BS limit case: short maturity matches BS with vol = sqrt(v0) (call)") {
  Market m = mkt();

  // General, non-degenerate Heston params
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  // Short maturity
  VanillaOption call{OptionType::Call, 100.0, 1.0 / 365.0};

  HestonFourierPricer pricer;
  const double heston_px = pricer.price(call, m, p);

  const double bs_vol = std::sqrt(p.v0);
  const double bs_px = test_helpers::bs_call(m.s0, call.K, call.T, m.r, m.q, bs_vol);

  // For tiny T this should be close (Fourier has numerical error too)
  CHECK(heston_px == doctest::Approx(bs_px).epsilon(5e-3));
}

TEST_CASE("BS limit case: short maturity matches BS with vol = sqrt(v0) (call+put)") {
  Market m = mkt();
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  const double T = 1.0 / 365.0;
  const double K = 110.0;

  VanillaOption call{OptionType::Call, K, T};
  VanillaOption put {OptionType::Put,  K, T};

  HestonFourierPricer pricer;
  const double c = pricer.price(call, m, p);
  const double pp = pricer.price(put, m, p);

  const double bs_vol = std::sqrt(p.v0);
  const double bs_c = test_helpers::bs_call(m.s0, K, T, m.r, m.q, bs_vol);
  const double bs_p = test_helpers::bs_put (m.s0, K, T, m.r, m.q, bs_vol);

  CHECK(c  == doctest::Approx(bs_c).epsilon(5e-3));
  CHECK(pp == doctest::Approx(bs_p).epsilon(5e-3));

  // Parity should still hold very tightly for Fourier pricer
  const double disc_r = std::exp(-m.r * T);
  const double disc_q = std::exp(-m.q * T);
  const double parity = (c - pp) - (m.s0 * disc_q - K * disc_r);
  CHECK(std::abs(parity) <= 1e-6);
}

TEST_CASE("Stability: call price increases with spot (Fourier)") {
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);
  VanillaOption call{OptionType::Call, 100.0, 1.0};

  Market m1{90.0, 0.02, 0.01};
  Market m2{110.0, 0.02, 0.01};

  HestonFourierPricer pricer;
  CHECK(pricer.price(call, m2, p) > pricer.price(call, m1, p));
}