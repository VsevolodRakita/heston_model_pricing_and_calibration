#pragma once
#include <cstddef>
#include <cmath>
#include <vector>

#include "utils/errors.hpp"

// Minimal dense linear algebra for the Gaussian-process regressor.
// Header-only, standard-library only — no third-party runtime dependency,
// in keeping with the rest of the library.
namespace heston::linalg {

// Row-major dense square matrix.
struct Matrix {
  std::size_t n = 0;
  std::vector<double> a;  // n*n, row-major

  Matrix() = default;
  explicit Matrix(std::size_t n_) : n(n_), a(n_ * n_, 0.0) {}

  [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept { return a[i * n + j]; }
  [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept { return a[i * n + j]; }
};

// Cholesky factorization of a symmetric positive-definite matrix A: returns the
// lower-triangular L with A = L Lᵀ. Throws NumericFailure on a non-positive
// pivot (A not SPD — e.g. a near-singular kernel matrix with too little jitter).
[[nodiscard]] inline Matrix cholesky(const Matrix& A) {
  const std::size_t n = A.n;
  Matrix L(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double sum = A(i, j);
      for (std::size_t k = 0; k < j; ++k) sum -= L(i, k) * L(j, k);
      if (i == j) {
        if (!(sum > 0.0)) throw NumericFailure("cholesky: matrix is not positive definite");
        L(i, j) = std::sqrt(sum);
      } else {
        L(i, j) = sum / L(j, j);
      }
    }
  }
  return L;
}

// Solve L y = b by forward substitution (L lower-triangular).
[[nodiscard]] inline std::vector<double> forward_subst(const Matrix& L, const std::vector<double>& b) {
  const std::size_t n = L.n;
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = b[i];
    for (std::size_t k = 0; k < i; ++k) sum -= L(i, k) * y[k];
    y[i] = sum / L(i, i);
  }
  return y;
}

// Solve Lᵀ x = y by back substitution (L lower-triangular, so Lᵀ upper).
[[nodiscard]] inline std::vector<double> back_subst_transpose(const Matrix& L, const std::vector<double>& y) {
  const std::size_t n = L.n;
  std::vector<double> x(n);
  for (std::size_t ii = 0; ii < n; ++ii) {
    const std::size_t i = n - 1 - ii;
    double sum = y[i];
    for (std::size_t k = i + 1; k < n; ++k) sum -= L(k, i) * x[k];
    x[i] = sum / L(i, i);
  }
  return x;
}

// Solve A x = b given the Cholesky factor L of A (A = L Lᵀ):
//   A x = b  <=>  L (Lᵀ x) = b  =>  forward-solve then back-solve.
[[nodiscard]] inline std::vector<double> chol_solve(const Matrix& L, const std::vector<double>& b) {
  return back_subst_transpose(L, forward_subst(L, b));
}

// log|A| from the Cholesky factor: log det(A) = 2 * sum_i log(L_ii).
[[nodiscard]] inline double chol_logdet(const Matrix& L) {
  double s = 0.0;
  for (std::size_t i = 0; i < L.n; ++i) s += std::log(L(i, i));
  return 2.0 * s;
}

}  // namespace heston::linalg
