#include "vol/black_scholes.hpp"

#include <algorithm>
#include <cmath>

#include "utils/errors.hpp"

namespace heston {

namespace bs_detail {

static double norm_cdf_(double x) {
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

static double norm_pdf_(double x) {
  static constexpr double inv_sqrt_2pi = 0.39894228040143267794;
  return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

} // namespace bs_detail

static double discounted_intrinsic_forward_(OptionType type, const Market& m, double K, double T) {
  const double disc_r = std::exp(-m.r * T);
  const double disc_q = std::exp(-m.q * T);
  const double fwd_disc = m.s0 * disc_q;
  const double k_disc = K * disc_r;

  if (type == OptionType::Call) return std::max(fwd_disc - k_disc, 0.0);
  return std::max(k_disc - fwd_disc, 0.0);
}

double black_scholes_price(
    const VanillaOption& opt,
    const Market& mkt,
    double vol) {

  if (!mkt.is_valid_basic()) throw InvalidInput("black_scholes_price: invalid market");
  if (!opt.is_valid_basic()) throw InvalidInput("black_scholes_price: invalid option (need K>0, T>0)");
  if (!std::isfinite(vol) || vol < 0.0) throw InvalidInput("black_scholes_price: vol must be finite and >= 0");

  // vol == 0 => deterministic forward payoff discounted
  if (vol == 0.0) {
    return discounted_intrinsic_forward_(opt.type, mkt, opt.K, opt.T);
  }

  const double sqrtT = std::sqrt(opt.T);
  const double sig_sqrtT = vol * sqrtT;

  const double disc_r = std::exp(-mkt.r * opt.T);
  const double disc_q = std::exp(-mkt.q * opt.T);

  const double d1 =
      (std::log(mkt.s0 / opt.K) + (mkt.r - mkt.q + 0.5 * vol * vol) * opt.T) / sig_sqrtT;
  const double d2 = d1 - sig_sqrtT;

  if (opt.type == OptionType::Call) {
    return mkt.s0 * disc_q * bs_detail::norm_cdf_(d1) - opt.K * disc_r * bs_detail::norm_cdf_(d2);
  } else {
    return opt.K * disc_r * bs_detail::norm_cdf_(-d2) - mkt.s0 * disc_q * bs_detail::norm_cdf_(-d1);
  }
}

double black_scholes_vega(
    const VanillaOption& opt,
    const Market& mkt,
    double vol) {

  if (!mkt.is_valid_basic()) throw InvalidInput("black_scholes_vega: invalid market");
  if (!opt.is_valid_basic()) throw InvalidInput("black_scholes_vega: invalid option (need K>0, T>0)");
  if (!std::isfinite(vol) || vol < 0.0) throw InvalidInput("black_scholes_vega: vol must be finite and >= 0");

  if (vol == 0.0) return 0.0;

  const double sqrtT = std::sqrt(opt.T);
  const double sig_sqrtT = vol * sqrtT;
  const double disc_q = std::exp(-mkt.q * opt.T);

  const double d1 =
      (std::log(mkt.s0 / opt.K) + (mkt.r - mkt.q + 0.5 * vol * vol) * opt.T) / sig_sqrtT;

  return mkt.s0 * disc_q * bs_detail::norm_pdf_(d1) * sqrtT;
}

} // namespace heston