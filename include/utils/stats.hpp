#pragma once
#include <cstddef>
#include <cmath>

namespace heston {

// Welford online mean/variance
class OnlineStats {
public:
  void add(double x) {
    ++n_;
    const double delta = x - mean_;
    mean_ += delta / double(n_);
    const double delta2 = x - mean_;
    m2_ += delta * delta2;
  }

  [[nodiscard]] std::size_t n() const noexcept { return n_; }
  [[nodiscard]] double mean() const noexcept { return mean_; }

  // sample variance
  [[nodiscard]] double var() const noexcept {
    return (n_ > 1) ? (m2_ / double(n_ - 1)) : 0.0;
  }

  [[nodiscard]] double stddev() const noexcept { return std::sqrt(var()); }

private:
  std::size_t n_ = 0;
  double mean_ = 0.0;
  double m2_ = 0.0;
};

}