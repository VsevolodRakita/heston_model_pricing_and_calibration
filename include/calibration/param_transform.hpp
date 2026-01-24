#pragma once
#include <cmath>
#include <vector>
#include "models/heston/heston_params.hpp"
#include "utils/errors.hpp"

namespace heston {

// x = (x0,x1,x2,x3,x4) in R^5
// v0     = exp(x0)
// kappa  = exp(x1)
// theta  = exp(x2)
// sigma  = exp(x3)
// rho    = tanh(x4)
inline HestonParams heston_from_unconstrained(const std::vector<double>& x) {
  if (x.size() != 5) throw InvalidInput("heston_from_unconstrained: expected x.size()==5");

  HestonParams p = HestonParams(std::exp(x[0]), std::exp(x[1]), std::exp(x[2]), std::exp(x[3]),std::tanh(x[4]));
  return p;
}

// Rough inverse (only for constructing x0 from an initial guess)
inline std::vector<double> unconstrained_from_heston(const HestonParams& p) {
  if (!p.is_finite()) throw InvalidInput("unconstrained_from_heston: params not finite");
  if (!(p.v0 > 0.0 && p.kappa > 0.0 && p.theta > 0.0 && p.sigma > 0.0)) {
    throw InvalidInput("unconstrained_from_heston: expected positive v0,kappa,theta,sigma");
  }
  if (!(p.rho > -1.0 && p.rho < 1.0)) throw InvalidInput("unconstrained_from_heston: rho must be in (-1,1)");

  std::vector<double> x(5);
  x[0] = std::log(p.v0);
  x[1] = std::log(p.kappa);
  x[2] = std::log(p.theta);
  x[3] = std::log(p.sigma);

  // atanh(rho) = 0.5*log((1+rho)/(1-rho))
  x[4] = 0.5 * std::log((1.0 + p.rho) / (1.0 - p.rho));
  return x;
}

} // namespace heston