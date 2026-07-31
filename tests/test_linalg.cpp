#include "third_party/doctest/doctest.h"

#include <cmath>
#include <vector>

#include "utils/errors.hpp"
#include "utils/linalg.hpp"

using namespace heston;
using heston::linalg::Matrix;

namespace {
// Classic textbook SPD matrix with an exact integer Cholesky factor:
//   A = [[4,12,-16],[12,37,-43],[-16,-43,98]],  L = [[2,0,0],[6,1,0],[-8,5,3]].
Matrix make_A() {
  Matrix A(3);
  const double vals[3][3] = {{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) A(i, j) = vals[i][j];
  return A;
}
}  // namespace

TEST_CASE("linalg: Cholesky matches the known factor and reconstructs A") {
  const Matrix A = make_A();
  const Matrix L = linalg::cholesky(A);

  CHECK(L(0, 0) == doctest::Approx(2.0));
  CHECK(L(1, 0) == doctest::Approx(6.0));
  CHECK(L(1, 1) == doctest::Approx(1.0));
  CHECK(L(2, 0) == doctest::Approx(-8.0));
  CHECK(L(2, 1) == doctest::Approx(5.0));
  CHECK(L(2, 2) == doctest::Approx(3.0));

  // Reconstruct A = L Lᵀ.
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < 3; ++k) s += L(i, k) * L(j, k);
      CHECK(s == doctest::Approx(A(i, j)));
    }
}

TEST_CASE("linalg: chol_solve recovers a known solution") {
  const Matrix A = make_A();
  const Matrix L = linalg::cholesky(A);

  // x_true = (1, 2, 3); b = A x_true = (-20, -43, 192).
  const std::vector<double> b = {-20.0, -43.0, 192.0};
  const std::vector<double> x = linalg::chol_solve(L, b);

  CHECK(x[0] == doctest::Approx(1.0));
  CHECK(x[1] == doctest::Approx(2.0));
  CHECK(x[2] == doctest::Approx(3.0));
}

TEST_CASE("linalg: log-determinant from the Cholesky factor") {
  const Matrix A = make_A();
  const Matrix L = linalg::cholesky(A);
  // det(A) = det(L)^2 = (2*1*3)^2 = 36.
  CHECK(linalg::chol_logdet(L) == doctest::Approx(std::log(36.0)));
}

TEST_CASE("linalg: non-positive-definite input throws") {
  Matrix A(2);
  A(0, 0) = 1.0;  A(0, 1) = 2.0;
  A(1, 0) = 2.0;  A(1, 1) = 1.0;  // indefinite (eigenvalues 3 and -1)
  CHECK_THROWS_AS((void)linalg::cholesky(A), NumericFailure);
}
