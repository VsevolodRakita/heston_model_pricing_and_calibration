#include "calibration/prepare_quotes.hpp"

#include "utils/errors.hpp"
#include "vol/black_scholes.hpp"

namespace heston {

std::vector<PriceQuote> prepare_price_quotes_from_iv(
    const std::vector<IvQuote>& iv_quotes,
    const Market& mkt) {

  if (!mkt.is_valid_basic()) throw InvalidInput("prepare_price_quotes_from_iv: invalid market");

  std::vector<PriceQuote> out;
  out.reserve(iv_quotes.size());

  for (const auto& q : iv_quotes) {
    if (!q.is_valid_basic()) throw InvalidInput("prepare_price_quotes_from_iv: invalid iv quote");

    const double px = black_scholes_price(q.opt, mkt, q.iv);
    if (!std::isfinite(px) || px < 0.0) throw NumericFailure("prepare_price_quotes_from_iv: non-finite/negative BS price");

    PriceQuote pq;
    pq.opt = q.opt;
    pq.market_price = px;
    pq.weight = q.weight;
    out.push_back(pq);
  }

  if (out.empty()) throw InvalidInput("prepare_price_quotes_from_iv: empty quote set");
  return out;
}

} // namespace heston