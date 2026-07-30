#include "greeks/greeks.hpp"

#include "utils/errors.hpp"

namespace heston {

namespace {

// Central difference when both bumped points are usable, otherwise a one-sided
// difference toward whichever side is valid. `price_of` maps params -> price.
template <class PriceOf>
[[nodiscard]] double diff_(
    PriceOf&& price_of,
    double base,
    const HestonParams& p_up, bool up_ok,
    const HestonParams& p_dn, bool dn_ok,
    double h)
{
  if (up_ok && dn_ok) return (price_of(p_up) - price_of(p_dn)) / (2.0 * h);
  if (up_ok)          return (price_of(p_up) - base) / h;
  if (dn_ok)          return (base - price_of(p_dn)) / h;
  throw NumericFailure("compute_param_sensitivities_fd: no valid bump direction");
}

void validate_settings_(const GreeksFdSettings& s) {
  const bool ok =
      s.spot_rel_bump > 0.0 && s.r_abs_bump > 0.0 && s.t_abs_bump > 0.0 &&
      s.v0_abs_bump > 0.0 && s.kappa_rel_bump > 0.0 && s.theta_abs_bump > 0.0 &&
      s.sigma_rel_bump > 0.0 && s.rho_abs_bump > 0.0;
  if (!ok) throw InvalidInput("Greeks: all finite-difference bump sizes must be > 0");
}

} // namespace

Greeks compute_greeks_fd(
    const IVanillaPricer& pricer,
    const VanillaOption& opt,
    const Market& mkt,
    const HestonParams& params,
    const GreeksFdSettings& settings)
{
  if (!opt.is_valid_basic() || !mkt.is_valid_basic() || !params.is_valid_basic()) {
    throw InvalidInput("compute_greeks_fd: invalid inputs");
  }
  validate_settings_(settings);

  Greeks g;
  const double base = pricer.price(opt, mkt, params);
  g.price = base;

  // Delta & Gamma: bump the spot.
  {
    const double h = settings.spot_rel_bump * mkt.s0;
    if (!(mkt.s0 - h > 0.0)) {
      throw InvalidInput("compute_greeks_fd: spot bump drives S0 <= 0");
    }
    Market up = mkt; up.s0 = mkt.s0 + h;
    Market dn = mkt; dn.s0 = mkt.s0 - h;
    const double v_up = pricer.price(opt, up, params);
    const double v_dn = pricer.price(opt, dn, params);
    g.delta = (v_up - v_dn) / (2.0 * h);
    g.gamma = (v_up - 2.0 * base + v_dn) / (h * h);
  }

  // Rho: bump the risk-free rate.
  {
    const double h = settings.r_abs_bump;
    Market up = mkt; up.r = mkt.r + h;
    Market dn = mkt; dn.r = mkt.r - h;
    const double v_up = pricer.price(opt, up, params);
    const double v_dn = pricer.price(opt, dn, params);
    g.rho = (v_up - v_dn) / (2.0 * h);
  }

  // Theta = dV/dt = -dV/dT. Shrink the bump if it would cross expiry.
  {
    double h = settings.t_abs_bump;
    if (!(opt.T - h > 0.0)) h = 0.5 * opt.T;
    VanillaOption up = opt; up.T = opt.T + h;
    VanillaOption dn = opt; dn.T = opt.T - h;
    const double v_up = pricer.price(up, mkt, params);
    const double v_dn = pricer.price(dn, mkt, params);
    const double dV_dT = (v_up - v_dn) / (2.0 * h);
    g.theta = -dV_dT;
  }

  return g;
}

HestonParamSensitivities compute_param_sensitivities_fd(
    const IVanillaPricer& pricer,
    const VanillaOption& opt,
    const Market& mkt,
    const HestonParams& params,
    const GreeksFdSettings& settings)
{
  if (!opt.is_valid_basic() || !mkt.is_valid_basic() || !params.is_valid_basic()) {
    throw InvalidInput("compute_param_sensitivities_fd: invalid inputs");
  }
  validate_settings_(settings);

  const auto price_of = [&](const HestonParams& p) { return pricer.price(opt, mkt, p); };
  const double base = price_of(params);

  HestonParamSensitivities out;

  // d/dv0 : variance floors at 0.
  {
    const double h = settings.v0_abs_bump;
    HestonParams up = params; up.v0 = params.v0 + h;
    HestonParams dn = params; dn.v0 = params.v0 - h;
    out.d_v0 = diff_(price_of, base, up, true, dn, (params.v0 - h) > 0.0, h);
  }

  // d/dkappa : kappa > 0 strictly, and a relative bump keeps it positive.
  {
    const double h = settings.kappa_rel_bump * params.kappa;
    HestonParams up = params; up.kappa = params.kappa + h;
    HestonParams dn = params; dn.kappa = params.kappa - h;
    out.d_kappa = diff_(price_of, base, up, true, dn, true, h);
  }

  // d/dtheta : long-run variance floors at 0.
  {
    const double h = settings.theta_abs_bump;
    HestonParams up = params; up.theta = params.theta + h;
    HestonParams dn = params; dn.theta = params.theta - h;
    out.d_theta = diff_(price_of, base, up, true, dn, (params.theta - h) > 0.0, h);
  }

  // d/dsigma : vol-of-vol floors at 0; use an absolute floor if sigma == 0.
  {
    const double h = (params.sigma > 0.0)
        ? settings.sigma_rel_bump * params.sigma
        : settings.sigma_rel_bump;
    HestonParams up = params; up.sigma = params.sigma + h;
    HestonParams dn = params; dn.sigma = params.sigma - h;
    out.d_sigma = diff_(price_of, base, up, true, dn, (params.sigma - h) > 0.0, h);
  }

  // d/drho : rho lives in the open interval (-1, 1).
  {
    const double h = settings.rho_abs_bump;
    HestonParams up = params; up.rho = params.rho + h;
    HestonParams dn = params; dn.rho = params.rho - h;
    out.d_rho = diff_(price_of, base, up, (params.rho + h) < 1.0, dn, (params.rho - h) > -1.0, h);
  }

  return out;
}

} // namespace heston
