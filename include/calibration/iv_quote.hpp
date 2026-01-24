#pragma once
#include <cmath>
#include "products/vanilla_option.hpp"

namespace heston {

struct IvQuote {
  VanillaOption opt;   // type, K, T
  double iv;           // market implied vol
  double weight = 1.0; // optional (e.g. vega weights later)

  [[nodiscard]] bool is_valid_basic() const noexcept {
    return opt.is_valid_basic() && std::isfinite(iv) && (iv > 0.0) && std::isfinite(weight) && (weight > 0.0);
  }
};

} // namespace heston