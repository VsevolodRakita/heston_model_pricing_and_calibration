#pragma once
#include <complex>
#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "utils/errors.hpp"

namespace heston {

class HestonCharacteristicFunction {
public:
  struct Settings {
    // Numerical and formulation choices for the Heston CF.
    // Currently only the Little Heston Trap formulation is implemented.
    bool use_little_trap = true;
  };

  HestonCharacteristicFunction() = default;
  explicit HestonCharacteristicFunction(const Settings& s) : settings_(s) {}

  [[nodiscard]] std::complex<double> log_spot_cf(
      std::complex<double> u,
      double t,
      const Market& mkt,
      const HestonParams& p) const;

private:
  Settings settings_;
};

} // namespace heston