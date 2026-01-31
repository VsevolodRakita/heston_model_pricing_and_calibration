#pragma once
#include <cstddef>
#include <functional>
#include <vector>

namespace heston {

struct CmaEsSettings {
  std::size_t max_iter = 300;
  std::size_t lambda = 0;            // if 0 => auto: 4 + floor(3*log(n))
  double sigma0 = 0.3;               // initial step size
  std::size_t seed = 123;            // deterministic
  double tol_f = 1e-12;              // stop if best improves less than this (approx)
  double tol_x = 1e-8;               // stop if step size is tiny
};

struct OptimizeResult {
  std::vector<double> x_best;
  double f_best = 0.0;
  std::size_t iters = 0;
};

OptimizeResult cma_es_minimize(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x0,
    const CmaEsSettings& s);

} // namespace heston