#include "vol/implied_vol.hpp"

#include <algorithm>
#include <cmath>

#include "utils/errors.hpp"
#include "vol/black_scholes.hpp"

namespace heston {

static bool is_finite_(double x) { return std::isfinite(x); }

double implied_vol_black_scholes(
    const VanillaOption& opt,
    const Market& mkt,
    double price,
    const ImpliedVolSettings& settings) {

  if (!mkt.is_valid_basic()) throw InvalidInput("implied_vol_black_scholes: invalid market");
  if (!opt.is_valid_basic()) throw InvalidInput("implied_vol_black_scholes: invalid option (need K>0, T>0)");
  if (!is_finite_(price) || price < 0.0) throw InvalidInput("implied_vol_black_scholes: price must be finite and >= 0");

  const double disc_r = std::exp(-mkt.r * opt.T);
  const double disc_q = std::exp(-mkt.q * opt.T);

  const double lower = (opt.type == OptionType::Call)
      ? std::max(mkt.s0 * disc_q - opt.K * disc_r, 0.0)
      : std::max(opt.K * disc_r - mkt.s0 * disc_q, 0.0);

  const double upper = (opt.type == OptionType::Call)
      ? mkt.s0 * disc_q
      : opt.K * disc_r;

  if (price < lower - 1e-10 || price > upper + 1e-10) {
    throw InvalidInput("implied_vol_black_scholes: price violates no-arbitrage bounds");
  }

  double lo = settings.vol_lower;
  double hi = settings.vol_upper;

  double f_lo = black_scholes_price(opt, mkt, lo) - price;
  double f_hi = black_scholes_price(opt, mkt, hi) - price;

  // Expand hi to get a bracket (bounded)
  std::size_t expand = 0;
  while (f_lo * f_hi > 0.0 && expand < 20) {
    hi *= 2.0;
    if (hi > 20.0) break;
    f_hi = black_scholes_price(opt, mkt, hi) - price;
    ++expand;
  }

  if (f_lo * f_hi > 0.0) {
    throw NumericFailure("implied_vol_black_scholes: failed to bracket implied vol");
  }

  // Bisection
  double mid = 0.5 * (lo + hi);
  for (std::size_t it = 0; it < settings.max_iter; ++it) {
    mid = 0.5 * (lo + hi);
    const double f_mid = black_scholes_price(opt, mkt, mid) - price;

    if (std::abs(f_mid) <= settings.tol_abs) break;

    if (f_lo * f_mid <= 0.0) {
      hi = mid;
      f_hi = f_mid;
    } else {
      lo = mid;
      f_lo = f_mid;
    }
  }

  double vol = mid;

  // Newton polish (stays within bracket)
  if (settings.newton_polish) {
    for (std::size_t it = 0; it < settings.newton_max_iter; ++it) {
      const double f = black_scholes_price(opt, mkt, vol) - price;
      if (std::abs(f) <= settings.tol_abs) break;

      const double vega = black_scholes_vega(opt, mkt, vol);
      if (vega <= 1e-14) break;

      const double next = std::clamp(vol - f / vega, lo, hi);
      if (std::abs(next - vol) <= 1e-14) break;
      vol = next;
    }
  }

  if (!is_finite_(vol) || vol < 0.0) throw NumericFailure("implied_vol_black_scholes: non-finite vol");
  return vol;
}

} // namespace heston