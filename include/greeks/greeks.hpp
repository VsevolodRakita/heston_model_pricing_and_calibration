#pragma once
#include "pricers/pricer.hpp"   // IVanillaPricer, VanillaOption, Market, HestonParams

namespace heston {

// First- and second-order sensitivities with respect to *market observables*
// (spot, time, rate). These are model-agnostic: they bump the Market / option,
// not the Heston parameters, so they mean the same thing for any pricer.
struct Greeks {
  double price = 0.0;   // the base price (returned for convenience)
  double delta = 0.0;   // dV/dS0
  double gamma = 0.0;   // d2V/dS0^2
  double theta = 0.0;   // dV/dt = -dV/dT   (per year; "time decay")
  double rho   = 0.0;   // dV/dr
};

// Sensitivities with respect to the five Heston parameters ("the vega bucket").
// d_v0 is the Heston analogue of vega: the sensitivity to the initial variance.
// (Sensitivity to initial *volatility* is d_v0 * 2*sqrt(v0).)
struct HestonParamSensitivities {
  double d_v0    = 0.0;
  double d_kappa = 0.0;
  double d_theta = 0.0;
  double d_sigma = 0.0;
  double d_rho   = 0.0;
};

// Finite-difference step sizes. Spot / kappa / sigma use *relative* bumps
// (scaled by the parameter); the rest use *absolute* bumps. Defaults are chosen
// to balance truncation vs. round-off for the deterministic Fourier pricer.
// For a Monte Carlo pricer, use common random numbers (a fixed seed) and expect
// noisier results — larger bumps help there.
struct GreeksFdSettings {
  double spot_rel_bump  = 1e-3;   // delta, gamma
  double r_abs_bump     = 1e-4;   // rho
  double t_abs_bump     = 1e-4;   // theta

  double v0_abs_bump    = 1e-4;
  double kappa_rel_bump = 1e-3;
  double theta_abs_bump = 1e-4;
  double sigma_rel_bump = 1e-3;
  double rho_abs_bump   = 1e-3;
};

// Market Greeks (delta, gamma, theta, rho) by central finite differences.
// gamma reuses the spot-bumped prices, so this costs 6 pricer evaluations.
[[nodiscard]] Greeks compute_greeks_fd(
    const IVanillaPricer& pricer,
    const VanillaOption& opt,
    const Market& mkt,
    const HestonParams& params,
    const GreeksFdSettings& settings = GreeksFdSettings{});

// Sensitivities to the five Heston parameters by central finite differences,
// falling back to one-sided differences at the edge of the valid region
// (e.g. sigma near 0, rho near +/-1).
[[nodiscard]] HestonParamSensitivities compute_param_sensitivities_fd(
    const IVanillaPricer& pricer,
    const VanillaOption& opt,
    const Market& mkt,
    const HestonParams& params,
    const GreeksFdSettings& settings = GreeksFdSettings{});

} // namespace heston
