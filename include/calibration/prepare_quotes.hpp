#pragma once
#include <vector>

#include "calibration/iv_quote.hpp"
#include "calibration/price_quote.hpp"
#include "models/heston/market.hpp"

namespace heston {

// Converts market IV quotes into market prices using Black–Scholes once.
std::vector<PriceQuote> prepare_price_quotes_from_iv(
    const std::vector<IvQuote>& iv_quotes,
    const Market& mkt);

} // namespace heston