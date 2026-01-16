#include "third_party/doctest/doctest.h"

#include "models/heston/heston_params.hpp"

#include <limits>
#include <string>

TEST_CASE("HestonParams: basic validity and Feller condition") {
  using namespace heston;

  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);

  CHECK(p.is_finite());
  CHECK(p.is_valid_basic());

  CHECK(p.feller_lhs_minus_rhs() == doctest::Approx(-0.09));
  CHECK_FALSE(p.satisfies_feller());

  SUBCASE("Invalid rho fails validity") {
    HestonParams bad(0.04, 2.0, 0.04, 0.5, 1.5);
    CHECK_FALSE(bad.is_valid_basic());
  }

  SUBCASE("Negative v0 fails validity") {
    HestonParams bad(-0.001, 2.0, 0.04, 0.5, -0.7);
    CHECK_FALSE(bad.is_valid_basic());
  }

  SUBCASE("A parameter set satisfying Feller is detected") {
    HestonParams ok(0.04, 3.0, 0.09, 0.5, -0.7);
    CHECK(ok.feller_lhs_minus_rhs() == doctest::Approx(0.29));
    CHECK(ok.satisfies_feller());
  }
}

TEST_CASE("HestonParams: to_string includes key fields") {
  using namespace heston;

  HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);
  const std::string s = p.to_string();

  CHECK(!s.empty());
  CHECK(s.find("HestonParams") != std::string::npos);
  CHECK(s.find("v0=") != std::string::npos);
  CHECK(s.find("kappa=") != std::string::npos);
  CHECK(s.find("theta=") != std::string::npos);
  CHECK(s.find("sigma=") != std::string::npos);
  CHECK(s.find("rho=") != std::string::npos);
  CHECK(s.find("feller") != std::string::npos);
}