#pragma once

#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"

namespace heston {

// Black–Scholes price for European vanilla option with continuous dividend yield q.
double black_scholes_price(
    const VanillaOption& opt,
    const Market& mkt,
    double vol);

// Black–Scholes vega (dPrice/dVol).
double black_scholes_vega(
    const VanillaOption& opt,
    const Market& mkt,
    double vol);

} // namespace heston