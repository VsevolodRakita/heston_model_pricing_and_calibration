#pragma once
#include <cmath>

#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"

namespace heston {

struct PriceQuote {
  VanillaOption opt;
  double market_price;     // computed once (e.g., from BS using market IV)
  double weight = 1.0;     // optional weights

  [[nodiscard]] bool is_valid_basic() const noexcept {
    return opt.is_valid_basic()
        && std::isfinite(market_price)
        && (market_price >= 0.0)
        && std::isfinite(weight)
        && (weight > 0.0);
  }
};

} // namespace heston