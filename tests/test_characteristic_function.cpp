#include "third_party/doctest/doctest.h"

#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "models/heston/characteristic_function.hpp"

#include <complex>
#include <cmath>

TEST_CASE("heston CF: phi(0) == 1") {
  using namespace heston;
  Market m{100.0, 0.02, 0.0};
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  HestonCharacteristicFunction cf;
  const std::complex<double> u(0.0, 0.0);

  auto phi = cf.log_spot_cf(u, 1.0, m, p);
  CHECK(phi.real() == doctest::Approx(1.0));
  CHECK(phi.imag() == doctest::Approx(0.0));
}

TEST_CASE("heston CF: conjugate symmetry for real u") {
  using namespace heston;
  Market m{100.0, 0.02, 0.0};
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  HestonCharacteristicFunction cf;
  const double ur = 1.234;

  auto phi_u  = cf.log_spot_cf({ur, 0.0}, 0.5, m, p);
  auto phi_mu = cf.log_spot_cf({-ur, 0.0}, 0.5, m, p);

  // For real-valued X, phi(-u) = conj(phi(u))
  CHECK(phi_mu.real() == doctest::Approx(phi_u.real()).epsilon(1e-10));
  CHECK(phi_mu.imag() == doctest::Approx(-phi_u.imag()).epsilon(1e-10));
}

TEST_CASE("heston CF: finite on a small grid") {
  using namespace heston;
  Market m{100.0, 0.02, 0.0};
  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  HestonCharacteristicFunction cf;

  for (double ur : {0.1, 0.5, 1.0, 2.0, 5.0}) {
    auto phi = cf.log_spot_cf({ur, 0.0}, 1.0, m, p);
    CHECK(std::isfinite(phi.real()));
    CHECK(std::isfinite(phi.imag()));
  }
}