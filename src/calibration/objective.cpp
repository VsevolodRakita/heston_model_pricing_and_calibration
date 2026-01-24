#include "calibration/objective.hpp"

#include <cmath>
#include "utils/errors.hpp"
#include "vol/implied_vol.hpp"

namespace heston {

ObjectiveResult evaluate_iv_objective(
    const std::vector<IvQuote>& quotes,
    const Market& mkt,
    const HestonParams& p,
    const HestonFourierPricer& fourier_pricer) {

  if (!mkt.is_valid_basic()) throw InvalidInput("evaluate_iv_objective: invalid market");
  if (!p.is_finite()) throw InvalidInput("evaluate_iv_objective: non-finite params");
  if (quotes.empty()) throw InvalidInput("evaluate_iv_objective: no quotes");

  ObjectiveResult out;
  out.residuals.reserve(quotes.size());

  // IV inversion settings for calibration use
  ImpliedVolSettings ivs;
  ivs.tol_abs = 1e-8;
  ivs.max_iter = 200;
  ivs.newton_polish = true;

  for (const auto& q : quotes) {
    if (!q.is_valid_basic()) throw InvalidInput("evaluate_iv_objective: invalid quote in set");

    // Model price via Fourier
    const double model_px = fourier_pricer.price(q.opt, mkt, p);

    // Convert model price -> model implied vol
    const double model_iv = implied_vol_black_scholes(q.opt, mkt, model_px, ivs);

    const double r = model_iv - q.iv;
    out.residuals.push_back(r);
    out.loss += q.weight * r * r;
  }

  if (!std::isfinite(out.loss)) throw NumericFailure("evaluate_iv_objective: non-finite loss");
  return out;
}

} // namespace heston