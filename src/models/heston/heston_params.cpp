#include "models/heston/heston_params.hpp"

#include <sstream>
#include <iomanip>

namespace heston {

std::string HestonParams::to_string() const {
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(6);

  oss << "HestonParams{"
      << "v0=" << v0
      << ", kappa=" << kappa
      << ", theta=" << theta
      << ", sigma=" << sigma
      << ", rho=" << rho
      << ", feller(2*kappa*theta - sigma^2)=" << feller_lhs_minus_rhs()
      << ", feller_ok=" << (satisfies_feller() ? "true" : "false")
      << "}";

  return oss.str();
}

}