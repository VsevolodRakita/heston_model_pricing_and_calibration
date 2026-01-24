#include "third_party/doctest/doctest.h"

#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"
#include "vol/black_scholes.hpp"
#include "vol/implied_vol.hpp"
#include "utils/errors.hpp"

using namespace heston;

TEST_CASE("Implied vol: BS price -> IV -> price round trip (call)") {
  Market m{100.0, 0.02, 0.01};
  VanillaOption call{OptionType::Call, 100.0, 1.0};

  const double vol_true = 0.25;
  const double px = black_scholes_price(call, m, vol_true);
  const double iv = implied_vol_black_scholes(call, m, px);

  CHECK(iv == doctest::Approx(vol_true).epsilon(1e-6));

  const double px2 = black_scholes_price(call, m, iv);
  CHECK(px2 == doctest::Approx(px).epsilon(1e-6));
}

TEST_CASE("Implied vol: BS price -> IV -> price round trip (put)") {
  Market m{100.0, 0.01, 0.00};
  VanillaOption put{OptionType::Put, 110.0, 0.5};

  const double vol_true = 0.40;
  const double px = black_scholes_price(put, m, vol_true);
  const double iv = implied_vol_black_scholes(put, m, px);

  CHECK(iv == doctest::Approx(vol_true).epsilon(1e-6));

  const double px2 = black_scholes_price(put, m, iv);
  CHECK(px2 == doctest::Approx(px).epsilon(1e-6));
}

TEST_CASE("Implied vol: rejects arbitrage-violating prices") {
  Market m{100.0, 0.02, 0.01};
  VanillaOption call{OptionType::Call, 100.0, 1.0};

  // Upper bound for call is S0*e^{-qT} ~ 99, so this is impossible.
  CHECK_THROWS_AS(implied_vol_black_scholes(call, m, 500.0), InvalidInput);
}
