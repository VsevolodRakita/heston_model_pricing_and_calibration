#pragma once
#include <random>
#include <cstdint>

namespace heston {

class Rng {
public:
  explicit Rng(std::uint64_t seed) : eng_(seed), norm_(0.0, 1.0) {}

  // standard normal
  [[nodiscard]] double z() { return norm_(eng_); }

private:
  std::mt19937_64 eng_;
  std::normal_distribution<double> norm_;
};

} // namespace heston