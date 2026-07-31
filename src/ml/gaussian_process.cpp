#include "ml/gaussian_process.hpp"

#include <algorithm>
#include <cmath>

#include "optimization/cma_es.hpp"
#include "utils/errors.hpp"

namespace heston {

namespace {
constexpr double kLog2Pi = 1.8378770664093453;  // log(2*pi)

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}
}  // namespace

double GaussianProcessRegressor::kernel_(const std::vector<double>& a, const std::vector<double>& b,
                                         const std::vector<double>& ell, double sigma_f) const {
  double q = 0.0;
  for (std::size_t k = 0; k < d_; ++k) {
    const double diff = (a[k] - b[k]) / ell[k];
    q += diff * diff;
  }
  return sigma_f * sigma_f * std::exp(-0.5 * q);
}

GaussianProcessRegressor::Factorization GaussianProcessRegressor::factor_(
    const std::vector<double>& ell, double sigma_f, double sigma_n) const {
  linalg::Matrix K(n_);
  const double diag = sigma_n * sigma_n + settings_.jitter;
  for (std::size_t i = 0; i < n_; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double kij = kernel_(Xs_[i], Xs_[j], ell, sigma_f);
      if (i == j) kij += diag;
      K(i, j) = kij;
      K(j, i) = kij;
    }
  }
  Factorization f;
  f.L = linalg::cholesky(K);            // throws NumericFailure if not SPD
  f.alpha = linalg::chol_solve(f.L, ys_);
  f.logdet = linalg::chol_logdet(f.L);
  return f;
}

std::vector<double> GaussianProcessRegressor::standardize_x_(const std::vector<double>& x) const {
  std::vector<double> xs(d_);
  for (std::size_t k = 0; k < d_; ++k) xs[k] = (x[k] - x_mean_[k]) / x_std_[k];
  return xs;
}

void GaussianProcessRegressor::init_hyperparameters_() {
  // Signal std of the standardized target is 1 by construction.
  sigma_f_ = 1.0;
  sigma_n_ = settings_.noise;

  // Median-heuristic length scale per dimension, computed on a capped random
  // sample of pairs of standardized inputs (inputs already have unit variance,
  // so this lands near 1 but adapts to the actual spread).
  ell_.assign(d_, 1.0);
  if (n_ >= 2) {
    const std::size_t max_pairs = 2000;
    for (std::size_t k = 0; k < d_; ++k) {
      std::vector<double> diffs;
      diffs.reserve(std::min(max_pairs, n_ * (n_ - 1) / 2));
      std::size_t count = 0;
      for (std::size_t i = 0; i < n_ && count < max_pairs; ++i) {
        for (std::size_t j = i + 1; j < n_ && count < max_pairs; ++j) {
          diffs.push_back(std::abs(Xs_[i][k] - Xs_[j][k]));
          ++count;
        }
      }
      if (!diffs.empty()) {
        std::nth_element(diffs.begin(), diffs.begin() + diffs.size() / 2, diffs.end());
        const double med = diffs[diffs.size() / 2];
        if (med > 1e-6) ell_[k] = med;
      }
    }
  }
}

void GaussianProcessRegressor::optimize_hyperparameters_() {
  // Optimize in log space so all hyperparameters stay strictly positive and the
  // search is unconstrained. Reuses the library's own CMA-ES optimizer.
  std::vector<double> x0(d_ + 2);
  for (std::size_t k = 0; k < d_; ++k) x0[k] = std::log(ell_[k]);
  x0[d_] = std::log(sigma_f_);
  x0[d_ + 1] = std::log(std::max(sigma_n_, 1e-8));

  auto objective = [this](const std::vector<double>& log_theta) {
    return this->neg_log_marginal_likelihood(log_theta);
  };

  CmaEsSettings s;
  s.max_iter = settings_.max_opt_iter;
  s.seed = static_cast<std::size_t>(settings_.opt_seed);
  s.sigma0 = 0.5;

  const OptimizeResult res = cma_es_minimize(objective, x0, s);

  // Adopt the fitted hyperparameters.
  for (std::size_t k = 0; k < d_; ++k) ell_[k] = std::exp(res.x_best[k]);
  sigma_f_ = std::exp(res.x_best[d_]);
  sigma_n_ = std::exp(res.x_best[d_ + 1]);
}

double GaussianProcessRegressor::neg_log_marginal_likelihood(
    const std::vector<double>& log_theta) const {
  std::vector<double> ell(d_);
  for (std::size_t k = 0; k < d_; ++k) ell[k] = std::exp(log_theta[k]);
  const double sigma_f = std::exp(log_theta[d_]);
  const double sigma_n = std::exp(log_theta[d_ + 1]);

  try {
    const Factorization f = factor_(ell, sigma_f, sigma_n);
    // NLML = 0.5 yᵀ K_y^{-1} y + 0.5 log|K_y| + (n/2) log(2π).
    return 0.5 * dot(ys_, f.alpha) + 0.5 * f.logdet + 0.5 * double(n_) * kLog2Pi;
  } catch (const NumericFailure&) {
    return 1e300;  // non-PD at these hyperparameters: steer the optimizer away
  }
}

void GaussianProcessRegressor::fit(const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& y) {
  if (X.empty() || X.size() != y.size())
    throw InvalidInput("GaussianProcessRegressor::fit: X and y sizes disagree or are empty");

  n_ = X.size();
  d_ = X[0].size();
  if (d_ == 0) throw InvalidInput("GaussianProcessRegressor::fit: zero-dimensional inputs");

  // Per-dimension input standardization.
  x_mean_.assign(d_, 0.0);
  x_std_.assign(d_, 0.0);
  for (const auto& row : X)
    for (std::size_t k = 0; k < d_; ++k) x_mean_[k] += row[k];
  for (std::size_t k = 0; k < d_; ++k) x_mean_[k] /= double(n_);
  for (const auto& row : X)
    for (std::size_t k = 0; k < d_; ++k) {
      const double dm = row[k] - x_mean_[k];
      x_std_[k] += dm * dm;
    }
  for (std::size_t k = 0; k < d_; ++k) {
    x_std_[k] = std::sqrt(x_std_[k] / double(n_));
    if (!(x_std_[k] > 1e-12)) x_std_[k] = 1.0;  // constant dimension
  }

  // Target standardization.
  y_mean_ = 0.0;
  for (double v : y) y_mean_ += v;
  y_mean_ /= double(n_);
  double yv = 0.0;
  for (double v : y) {
    const double dm = v - y_mean_;
    yv += dm * dm;
  }
  y_std_ = std::sqrt(yv / double(n_));
  if (!(y_std_ > 1e-12)) y_std_ = 1.0;  // constant target

  Xs_.assign(n_, std::vector<double>(d_));
  for (std::size_t i = 0; i < n_; ++i)
    for (std::size_t k = 0; k < d_; ++k) Xs_[i][k] = (X[i][k] - x_mean_[k]) / x_std_[k];
  ys_.assign(n_, 0.0);
  for (std::size_t i = 0; i < n_; ++i) ys_[i] = (y[i] - y_mean_) / y_std_;

  init_hyperparameters_();
  if (settings_.optimize_hyperparameters) optimize_hyperparameters_();

  // Final factorization at the chosen hyperparameters.
  const Factorization f = factor_(ell_, sigma_f_, sigma_n_);
  L_ = f.L;
  alpha_ = f.alpha;
}

GaussianProcessRegressor::Prediction GaussianProcessRegressor::predict(
    const std::vector<double>& x) const {
  if (alpha_.empty()) throw InvalidInput("GaussianProcessRegressor::predict: model is not fitted");
  if (x.size() != d_) throw InvalidInput("GaussianProcessRegressor::predict: wrong input dimension");

  const std::vector<double> xs = standardize_x_(x);

  std::vector<double> kstar(n_);
  for (std::size_t i = 0; i < n_; ++i) kstar[i] = kernel_(xs, Xs_[i], ell_, sigma_f_);

  const double mean_s = dot(kstar, alpha_);

  // Predictive variance: k(x*,x*) - kstarᵀ K_y^{-1} kstar, via v = L^{-1} kstar.
  const std::vector<double> v = linalg::forward_subst(L_, kstar);
  double var_s = sigma_f_ * sigma_f_ - dot(v, v);
  if (var_s < 0.0) var_s = 0.0;  // guard tiny negative from round-off

  Prediction p;
  p.mean = y_mean_ + y_std_ * mean_s;
  p.variance = y_std_ * y_std_ * var_s;
  return p;
}

}  // namespace heston
