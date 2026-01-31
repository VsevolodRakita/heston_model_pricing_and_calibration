#include "optimization/cma_es.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "utils/errors.hpp"

namespace heston {

static bool is_finite_vec_(const std::vector<double>& x) {
  for (double v : x) if (!std::isfinite(v)) return false;
  return true;
}

static std::vector<double> zeros_(std::size_t n) { return std::vector<double>(n, 0.0); }

static double dot_(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

static double norm2_(const std::vector<double>& a) { return std::sqrt(dot_(a, a)); }

OptimizeResult cma_es_minimize(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x0,
    const CmaEsSettings& s) {

  if (!f) throw InvalidInput("cma_es_minimize: objective empty");
  if (x0.empty()) throw InvalidInput("cma_es_minimize: x0 empty");
  if (!is_finite_vec_(x0)) throw InvalidInput("cma_es_minimize: x0 non-finite");
  if (!(s.sigma0 > 0.0)) throw InvalidInput("cma_es_minimize: sigma0 must be > 0");
  if (s.max_iter == 0) throw InvalidInput("cma_es_minimize: max_iter must be > 0");

  const std::size_t n = x0.size();
  const std::size_t lambda = (s.lambda == 0) ? (4u + static_cast<std::size_t>(std::floor(3.0 * std::log(static_cast<double>(n))))) : s.lambda;
  const std::size_t mu = lambda / 2;

  // recombination weights
  std::vector<double> w(mu, 0.0);
  for (std::size_t i = 0; i < mu; ++i) w[i] = std::log((mu + 0.5)) - std::log(i + 1.0);
  const double w_sum = std::accumulate(w.begin(), w.end(), 0.0);
  for (double& wi : w) wi /= w_sum;
  const double mu_eff = 1.0 / std::accumulate(w.begin(), w.end(), 0.0, [](double acc, double wi){ return acc + wi*wi; });

  // Strategy parameters (diagonal CMA-ES)
  const double c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0);
  const double d_sigma = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma;
  const double c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);
  const double c1 = 2.0 / ((n + 1.3) * (n + 1.3) + mu_eff);
  const double c_mu = std::min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0)*(n + 2.0) + mu_eff));

  // expectation of ||N(0,I)||
  const double chi_n = std::sqrt(static_cast<double>(n)) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n*n));

  std::mt19937_64 rng(s.seed);
  std::normal_distribution<double> nd(0.0, 1.0);

  std::vector<double> mean = x0;
  std::vector<double> pc = zeros_(n);
  std::vector<double> ps = zeros_(n);
  std::vector<double> diagC(n, 1.0);   // diagonal covariance
  std::vector<double> diagD(n, 1.0);   // sqrt(diagC)
  double sigma = s.sigma0;

  // Track best
  OptimizeResult best;
  best.x_best = x0;
  best.f_best = f(x0);
  if (!std::isfinite(best.f_best)) throw NumericFailure("cma_es_minimize: objective non-finite at x0");

  double prev_best = best.f_best;

  struct Candidate { std::vector<double> x; std::vector<double> z; double fx; };
  std::vector<Candidate> pop;
  pop.reserve(lambda);

  for (std::size_t iter = 0; iter < s.max_iter; ++iter) {
    // eigenvalues for diagonal are just diagD = sqrt(diagC)
    for (std::size_t i = 0; i < n; ++i) diagD[i] = std::sqrt(std::max(1e-18, diagC[i]));

    pop.clear();
    for (std::size_t k = 0; k < lambda; ++k) {
      std::vector<double> z(n, 0.0);
      std::vector<double> x(n, 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        z[i] = nd(rng);
        x[i] = mean[i] + sigma * diagD[i] * z[i];
      }

      double fx = f(x);
      if (!std::isfinite(fx)) fx = 1e50; // penalty (important)
      pop.push_back({std::move(x), std::move(z), fx});
    }

    std::sort(pop.begin(), pop.end(), [](const Candidate& a, const Candidate& b){ return a.fx < b.fx; });

    if (pop[0].fx < best.f_best) {
      best.f_best = pop[0].fx;
      best.x_best = pop[0].x;
    }

    // Recombine to new mean: mean += sigma * sum w_i * D*z_i (diag)
    std::vector<double> y_w(n, 0.0); // y_w = sum w_i * (D * z_i)
    for (std::size_t i = 0; i < mu; ++i) {
      for (std::size_t j = 0; j < n; ++j) y_w[j] += w[i] * (diagD[j] * pop[i].z[j]);
    }

    // Update mean
    for (std::size_t j = 0; j < n; ++j) mean[j] += sigma * y_w[j];

    // Update evolution path ps
    // ps = (1-c_sigma) ps + sqrt(c_sigma(2-c_sigma) mu_eff) * (C^{-1/2} * y_w)
    // For diagonal: C^{-1/2} is 1/diagD
    const double ps_coef = std::sqrt(c_sigma * (2.0 - c_sigma) * mu_eff);
    for (std::size_t j = 0; j < n; ++j) {
      const double c_inv_sqrt = 1.0 / std::max(1e-18, diagD[j]);
      ps[j] = (1.0 - c_sigma) * ps[j] + ps_coef * (c_inv_sqrt * y_w[j]);
    }

    const double ps_norm = norm2_(ps);
    const bool h_sigma = ps_norm / std::sqrt(1.0 - std::pow(1.0 - c_sigma, 2.0 * (iter + 1))) < (1.4 + 2.0/(n + 1.0)) * chi_n;

    // Update evolution path pc
    const double pc_coef = std::sqrt(c_c * (2.0 - c_c) * mu_eff);
    for (std::size_t j = 0; j < n; ++j) {
      pc[j] = (1.0 - c_c) * pc[j] + (h_sigma ? pc_coef * y_w[j] : 0.0);
    }

    // Rank-one + rank-mu update of diagonal C
    // diagC = (1 - c1 - c_mu) diagC + c1 * pc^2 + c_mu * sum w_i * (D*z_i)^2
    std::vector<double> rank_mu(n, 0.0);
    for (std::size_t i = 0; i < mu; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const double dz = diagD[j] * pop[i].z[j];
        rank_mu[j] += w[i] * (dz * dz);
      }
    }

    for (std::size_t j = 0; j < n; ++j) {
      const double old = diagC[j];
      const double pc2 = pc[j] * pc[j];
      diagC[j] = (1.0 - c1 - c_mu) * old + c1 * pc2 + c_mu * rank_mu[j];
      diagC[j] = std::max(1e-18, diagC[j]);
    }

    // Step-size control
    sigma *= std::exp((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0));
    sigma = std::max(1e-12, sigma);

    // stopping checks
    if (std::abs(prev_best - best.f_best) < s.tol_f) {
      best.iters = iter + 1;
      return best;
    }
    prev_best = best.f_best;

    if (sigma < s.tol_x) {
      best.iters = iter + 1;
      return best;
    }
  }

  best.iters = s.max_iter;
  return best;
}

} // namespace heston