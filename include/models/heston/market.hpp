#pragma once
#include <cmath>

namespace heston {

struct Market {
  double s0 ;  // spot
  double r;   // continuously-compounded risk-free rate
  double q;   // continuously-compounded dividend yield / carry

  [[nodiscard]] bool is_valid_basic() const noexcept {
    return std::isfinite(s0) && std::isfinite(r) && std::isfinite(q) && (s0 > 0.0);
  }
};

} // namespace heston