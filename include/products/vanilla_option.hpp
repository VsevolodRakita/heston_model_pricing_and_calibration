#pragma once
#include <cmath>

namespace heston {

enum class OptionType { Call, Put };

struct VanillaOption {
  OptionType type = OptionType::Call;
  double K;   // strike
  double T;     // maturity in years

  [[nodiscard]] bool is_valid_basic() const noexcept {
    return std::isfinite(K) && std::isfinite(T) && (K > 0.0) && (T > 0.0);
  }
};

} // namespace heston