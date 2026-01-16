#pragma once
#include <cstddef>
#include <random>
#include <cmath>
#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "utils/errors.hpp"

namespace heston {

enum class VarianceScheme {
  FullTruncationEuler,   // simple & robust
  QE                  // Andersen QE (better accuracy, a bit more code)
};

struct SimulationSettings {
  std::size_t nPaths;
  std::size_t nSteps;     // time steps per path
  bool antithetic = true;
  VarianceScheme varScheme = VarianceScheme::FullTruncationEuler;

  // Optional: deterministic seeding for reproducibility
  std::uint64_t seed = 42;

  [[nodiscard]] bool is_valid_basic() const noexcept {
    return nPaths > 0 && nSteps > 0;
  }
};

// Simple POD for MC summary
struct MCResult {
  double price = 0.0;
  double stdError = 0.0;  // standard error of discounted payoff mean
};

} // namespace heston