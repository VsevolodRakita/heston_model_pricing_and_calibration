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

// Closed-form Black–Scholes Greeks (with continuous dividend yield q).
// These serve as the analytic reference the finite-difference Greeks are
// validated against. All require vol > 0.
double black_scholes_delta(   // dPrice/dS0
    const VanillaOption& opt, const Market& mkt, double vol);
double black_scholes_gamma(   // d2Price/dS0^2
    const VanillaOption& opt, const Market& mkt, double vol);
double black_scholes_theta(   // dPrice/dt = -dPrice/dT, per year
    const VanillaOption& opt, const Market& mkt, double vol);
double black_scholes_rho(     // dPrice/dr
    const VanillaOption& opt, const Market& mkt, double vol);

} // namespace heston