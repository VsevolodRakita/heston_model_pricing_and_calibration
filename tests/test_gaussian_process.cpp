#include "third_party/doctest/doctest.h"

#include <cmath>
#include <vector>

#include "ml/gaussian_process.hpp"

using namespace heston;

namespace {
// A smooth, non-trivial 2-D target.
double target(double x, double y) { return std::sin(0.9 * x) * std::cos(0.7 * y) + 0.5 * x; }

// 8x8 grid of training points on [0,3]^2.
void make_training(std::vector<std::vector<double>>& X, std::vector<double>& y) {
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      const double x = 3.0 * i / 7.0;
      const double yy = 3.0 * j / 7.0;
      X.push_back({x, yy});
      y.push_back(target(x, yy));
    }
}
}  // namespace

TEST_CASE("GPR: interpolates training points as noise -> 0") {
  std::vector<std::vector<double>> X;
  std::vector<double> y;
  make_training(X, y);

  GaussianProcessRegressor::Settings s;
  s.optimize_hyperparameters = false;  // fixed median-heuristic length scales
  s.noise = 1e-7;                      // near-interpolation
  GaussianProcessRegressor gp(s);
  gp.fit(X, y);

  // At a training input the posterior mean reproduces the target (any length
  // scale, because noise -> 0), and the predictive variance collapses.
  const std::vector<double> xq = X[20];
  const auto pred = gp.predict(xq);
  CHECK(pred.mean == doctest::Approx(y[20]).epsilon(1e-3));
  CHECK(pred.variance < 1e-3);
}

TEST_CASE("GPR: accurate out-of-sample after fitting hyperparameters") {
  std::vector<std::vector<double>> X;
  std::vector<double> y;
  make_training(X, y);

  GaussianProcessRegressor::Settings s;
  s.optimize_hyperparameters = true;
  s.max_opt_iter = 150;
  GaussianProcessRegressor gp(s);
  gp.fit(X, y);

  // Interior points that are not on the training grid.
  for (const auto& q : {std::vector<double>{1.55, 1.35}, std::vector<double>{0.8, 2.1},
                        std::vector<double>{2.2, 0.55}}) {
    const auto pred = gp.predict(q);
    CHECK(pred.mean == doctest::Approx(target(q[0], q[1])).epsilon(0.05));
  }
}

TEST_CASE("GPR: predictive variance grows away from the data") {
  std::vector<std::vector<double>> X;
  std::vector<double> y;
  make_training(X, y);

  GaussianProcessRegressor::Settings s;
  s.optimize_hyperparameters = false;
  s.noise = 1e-4;
  GaussianProcessRegressor gp(s);
  gp.fit(X, y);

  const auto near = gp.predict({1.5, 1.5});     // inside the training region
  const auto far = gp.predict({20.0, 20.0});    // far outside
  CHECK(far.variance > near.variance);
}
