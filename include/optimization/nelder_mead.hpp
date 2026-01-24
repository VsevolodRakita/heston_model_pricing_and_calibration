#pragma once
#include <cstddef>
#include <functional>
#include <vector>

namespace heston {

struct NelderMeadSettings {
  std::size_t max_iter = 500;
  double tol_f = 1e-10;
  double tol_x = 1e-8;

  // standard coefficients
  double alpha = 1.0; // reflection
  double gamma = 2.0; // expansion
  double rho   = 0.5; // contraction
  double sigma = 0.5; // shrink
};

struct OptimizeResult {
  std::vector<double> x_best;
  double f_best = 0.0;
  std::size_t iters = 0;
};

OptimizeResult nelder_mead_minimize(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x0,
    const NelderMeadSettings& s);

} // namespace heston