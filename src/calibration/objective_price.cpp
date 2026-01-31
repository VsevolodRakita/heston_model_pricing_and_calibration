#include "calibration/objective_price.hpp"

#include <cmath>

#include "utils/errors.hpp"

namespace heston {

ObjectiveResult evaluate_price_objective(
    const std::vector<PriceQuote>& quotes,
    const Market& mkt,
    const HestonParams& p,
    const HestonFourierPricer& pricer) {

  if (!mkt.is_valid_basic()) throw InvalidInput("evaluate_price_objective: invalid market");
  if (quotes.empty()) throw InvalidInput("evaluate_price_objective: empty quotes");
  if (!p.is_finite()) throw InvalidInput("evaluate_price_objective: non-finite params");

  ObjectiveResult out;
  out.residuals.reserve(quotes.size());

  for (const auto& q : quotes) {
    if (!q.is_valid_basic()) throw InvalidInput("evaluate_price_objective: invalid quote");

    const double model_px = pricer.price(q.opt, mkt, p);
    if (!std::isfinite(model_px)) throw NumericFailure("evaluate_price_objective: non-finite model price");

    const double r = model_px - q.market_price;
    out.residuals.push_back(r);
    out.loss += q.weight * r * r;
  }

  if (!std::isfinite(out.loss)) throw NumericFailure("evaluate_price_objective: non-finite loss");
  return out;
}

} // namespace heston