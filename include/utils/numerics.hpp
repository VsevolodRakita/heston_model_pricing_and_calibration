#pragma once
#include <vector>
#include <complex>
#include <cstddef>
#include <cmath>

namespace heston {

// Uniform grid: x_j = x0 + j*dx
struct UniformGrid {
  double x0 = 0.0;
  double dx = 1.0;
  std::size_t n = 0;

  [[nodiscard]] double operator[](std::size_t j) const noexcept { return x0 + dx * double(j); }
};

// Basic Simpson integration for smooth integrands on [a, b] with even n intervals.
template <typename F>
double simpson(F&& f, double a, double b, std::size_t nEven) {
  if (nEven < 2) nEven = 2;
  if (nEven % 2 == 1) ++nEven;
  const double h = (b - a) / double(nEven);

  double s = f(a) + f(b);
  for (std::size_t i = 1; i < nEven; ++i) {
    const double x = a + h * double(i);
    s += (i % 2 ? 4.0 : 2.0) * f(x);
  }
  return s * h / 3.0;
}

}// namespace heston