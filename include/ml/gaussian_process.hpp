#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "utils/linalg.hpp"

namespace heston {

// Gaussian Process Regression with an Automatic-Relevance-Determination (ARD)
// squared-exponential kernel:
//
//   k(x, x') = sigma_f^2 * exp( -0.5 * sum_d (x_d - x'_d)^2 / ell_d^2 )
//
// Inputs and the target are standardized internally (zero mean / unit variance,
// per dimension). Standardization is what keeps the kernel matrix well
// conditioned when the raw features live on very different scales
// (e.g. strike ~ 100 alongside rho ~ -0.7).
//
// The regressor is deliberately dimension-agnostic and reusable: the surrogate
// pricer is one client, but the same object can later back calibration or
// implied-vol-surface fitting.
class GaussianProcessRegressor {
public:
  struct Settings {
    // Fit {ell_1..ell_d, sigma_f, sigma_n} by maximizing the marginal
    // log-likelihood. If false, a median-heuristic length scale and the fixed
    // `noise` below are used (much cheaper — no O(n^3)-per-iteration search).
    bool optimize_hyperparameters = true;

    std::size_t max_opt_iter = 200;   // CMA-ES budget for the hyperparameter search
    std::uint64_t opt_seed = 7;       // deterministic hyperparameter search
    double noise = 1e-6;              // observation noise std (standardized units)
    double jitter = 1e-10;           // extra diagonal for numerical stability
  };

  GaussianProcessRegressor() = default;
  explicit GaussianProcessRegressor(const Settings& s) : settings_(s) {}

  // Fit to training inputs X (n rows, each of dimension d) and targets y (n).
  void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

  struct Prediction {
    double mean = 0.0;
    double variance = 0.0;  // predictive variance (>= 0), in original target units^2
  };

  [[nodiscard]] Prediction predict(const std::vector<double>& x) const;

  // Negative log marginal likelihood at hyperparameters encoded in log space:
  //   log_theta = [log ell_1, ..., log ell_d, log sigma_f, log sigma_n].
  // Exposed for the optimizer and for testing. Returns a large finite penalty
  // if the kernel matrix is not positive definite at those hyperparameters.
  [[nodiscard]] double neg_log_marginal_likelihood(const std::vector<double>& log_theta) const;

  [[nodiscard]] std::size_t dim() const noexcept { return d_; }
  [[nodiscard]] std::size_t n_train() const noexcept { return n_; }
  [[nodiscard]] const std::vector<double>& length_scales() const noexcept { return ell_; }
  [[nodiscard]] double signal_stddev() const noexcept { return sigma_f_; }
  [[nodiscard]] double noise_stddev() const noexcept { return sigma_n_; }

private:
  Settings settings_{};

  // Standardization state.
  std::size_t d_ = 0, n_ = 0;
  std::vector<double> x_mean_, x_std_;  // per-dimension (size d)
  double y_mean_ = 0.0, y_std_ = 1.0;

  // Standardized training data.
  std::vector<std::vector<double>> Xs_;  // n x d
  std::vector<double> ys_;               // n

  // Hyperparameters (in standardized space).
  std::vector<double> ell_;  // d length scales
  double sigma_f_ = 1.0;     // signal std
  double sigma_n_ = 1e-3;    // noise std

  // Trained state: Cholesky of K_y = K + sigma_n^2 I, and alpha = K_y^{-1} ys_.
  linalg::Matrix L_;
  std::vector<double> alpha_;

  // A factorization of K_y for a given hyperparameter set.
  struct Factorization {
    linalg::Matrix L;
    std::vector<double> alpha;
    double logdet = 0.0;
  };

  [[nodiscard]] double kernel_(const std::vector<double>& a, const std::vector<double>& b,
                               const std::vector<double>& ell, double sigma_f) const;
  [[nodiscard]] Factorization factor_(const std::vector<double>& ell, double sigma_f,
                                      double sigma_n) const;
  [[nodiscard]] std::vector<double> standardize_x_(const std::vector<double>& x) const;
  void init_hyperparameters_();
  void optimize_hyperparameters_();
};

}  // namespace heston
