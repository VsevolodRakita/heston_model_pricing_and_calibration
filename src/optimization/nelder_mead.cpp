#include "optimization/nelder_mead.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "utils/errors.hpp"

namespace heston {

static bool is_finite_vec_(const std::vector<double>& x) {
  for (double v : x) {
    if (!std::isfinite(v)) return false;
  }
  return true;
}

static double max_abs_diff_(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

static std::vector<double> add_scaled_(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double scale_b) {
  // returns a + scale_b * b
  std::vector<double> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + scale_b * b[i];
  return out;
}

static std::vector<double> sub_(
    const std::vector<double>& a,
    const std::vector<double>& b) {
  std::vector<double> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] - b[i];
  return out;
}

OptimizeResult nelder_mead_minimize(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x0,
    const NelderMeadSettings& s) {

  if (!f) throw InvalidInput("nelder_mead_minimize: objective function is empty");
  if (x0.empty()) throw InvalidInput("nelder_mead_minimize: x0 is empty");
  if (!is_finite_vec_(x0)) throw InvalidInput("nelder_mead_minimize: x0 contains non-finite values");
  if (s.max_iter == 0) throw InvalidInput("nelder_mead_minimize: max_iter must be > 0");
  if (!(s.tol_f > 0.0) || !(s.tol_x > 0.0)) throw InvalidInput("nelder_mead_minimize: tolerances must be > 0");

  const std::size_t n = x0.size();
  const std::size_t m = n + 1; // simplex size

  // Build initial simplex:
  // x0 and x0 + step_i * e_i
  std::vector<std::vector<double>> X(m, x0);
  for (std::size_t i = 0; i < n; ++i) {
    double step = 0.05 * (std::abs(x0[i]) + 1.0);
    X[i + 1][i] += step;
  }

  std::vector<double> F(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    const double fi = f(X[i]);
    if (!std::isfinite(fi)) throw NumericFailure("nelder_mead_minimize: objective returned non-finite value");
    F[i] = fi;
  }

  auto sort_simplex_ = [&]() {
    std::vector<std::size_t> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return F[a] < F[b]; });

    std::vector<std::vector<double>> X2;
    std::vector<double> F2;
    X2.reserve(m);
    F2.reserve(m);
    for (std::size_t k = 0; k < m; ++k) {
      X2.push_back(X[idx[k]]);
      F2.push_back(F[idx[k]]);
    }
    X.swap(X2);
    F.swap(F2);
  };

  auto centroid_excluding_worst_ = [&]() {
    // centroid of best n points (exclude worst at index n)
    std::vector<double> c(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) { // 0..n-1
      for (std::size_t j = 0; j < n; ++j) c[j] += X[i][j];
    }
    for (double& v : c) v /= static_cast<double>(n);
    return c;
  };

  auto f_range_ = [&]() {
    auto [min_it, max_it] = std::minmax_element(F.begin(), F.end());
    return *max_it - *min_it;
  };

  auto simplex_diameter_ = [&]() {
    // max distance from best point in infinity norm
    double d = 0.0;
    for (std::size_t i = 1; i < m; ++i) d = std::max(d, max_abs_diff_(X[i], X[0]));
    return d;
  };

  sort_simplex_();

  std::size_t it = 0;
  for (; it < s.max_iter; ++it) {
    // Check convergence
    if (f_range_() <= s.tol_f && simplex_diameter_() <= s.tol_x) break;

    const std::vector<double> x_best = X[0];
    const std::vector<double> x_worst = X[n];
    const double f_best = F[0];
    const double f_second_worst = F[n - 1];
    const double f_worst = F[n];

    const std::vector<double> c = centroid_excluding_worst_();

    // Reflection: xr = c + alpha * (c - x_worst)
    const std::vector<double> xr = add_scaled_(c, sub_(c, x_worst), s.alpha);
    double fr = f(xr);
    if (!std::isfinite(fr)) throw NumericFailure("nelder_mead_minimize: objective returned non-finite value");

    if (fr < f_best) {
      // Expansion: xe = c + gamma * (xr - c)
      const std::vector<double> xe = add_scaled_(c, sub_(xr, c), s.gamma);
      double fe = f(xe);
      if (!std::isfinite(fe)) throw NumericFailure("nelder_mead_minimize: objective returned non-finite value");

      if (fe < fr) {
        X[n] = xe;
        F[n] = fe;
      } else {
        X[n] = xr;
        F[n] = fr;
      }
      sort_simplex_();
      continue;
    }

    if (fr < f_second_worst) {
      // Accept reflection
      X[n] = xr;
      F[n] = fr;
      sort_simplex_();
      continue;
    }

    // Contraction
    if (fr < f_worst) {
      // Outside contraction: xoc = c + rho * (xr - c)
      const std::vector<double> xoc = add_scaled_(c, sub_(xr, c), s.rho);
      double foc = f(xoc);
      if (!std::isfinite(foc)) throw NumericFailure("nelder_mead_minimize: objective returned non-finite value");

      if (foc <= fr) {
        X[n] = xoc;
        F[n] = foc;
        sort_simplex_();
        continue;
      }
    } else {
      // Inside contraction: xic = c - rho * (c - x_worst) = c + rho * (x_worst - c)
      const std::vector<double> xic = add_scaled_(c, sub_(x_worst, c), s.rho);
      double fic = f(xic);
      if (!std::isfinite(fic)) throw NumericFailure("nelder_mead_minimize: objective returned non-finite value");

      if (fic < f_worst) {
        X[n] = xic;
        F[n] = fic;
        sort_simplex_();
        continue;
      }
    }

    // Shrink towards best: Xi = x_best + sigma * (Xi - x_best)
    for (std::size_t i = 1; i < m; ++i) {
      const std::vector<double> diff = sub_(X[i], x_best);
      X[i] = add_scaled_(x_best, diff, s.sigma);
      const double fi = f(X[i]);
      if (!std::isfinite(fi)) throw NumericFailure("nelder_mead_minimize: objective returned non-finite value");
      F[i] = fi;
    }
    sort_simplex_();
  }

  OptimizeResult out;
  out.x_best = X[0];
  out.f_best = F[0];
  out.iters = it;
  return out;
}

} // namespace heston