#pragma once
#include <string>
#include <cmath>

namespace heston {

struct HestonParams {
  double v0;
  double kappa;
  double theta;
  double sigma;
  double rho;

  HestonParams() {};

  constexpr HestonParams(double v0_,
                         double kappa_,
                         double theta_,
                         double sigma_,
                         double rho_) noexcept
    : v0(v0_), kappa(kappa_), theta(theta_), sigma(sigma_), rho(rho_) {}

  [[nodiscard]] bool is_finite() const noexcept {
    return std::isfinite(v0) && std::isfinite(kappa) && std::isfinite(theta) &&
           std::isfinite(sigma) && std::isfinite(rho);
  }

  [[nodiscard]] bool is_valid_basic() const noexcept {
    if (!is_finite()) return false;
    if (v0 < 0.0 || kappa <= 0.0 || theta < 0.0 || sigma < 0.0) return false;
    if (rho < -1.0 || rho > 1.0) return false;
    return true;
  }

  [[nodiscard]] double feller_lhs_minus_rhs() const noexcept {
    return 2.0 * kappa * theta - sigma * sigma;
  }

  [[nodiscard]] bool satisfies_feller() const noexcept {
    return feller_lhs_minus_rhs() >= 0.0;
  }

  [[nodiscard]] std::string to_string() const;
};

}