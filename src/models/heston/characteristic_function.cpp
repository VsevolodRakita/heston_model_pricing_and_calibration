#include "models/heston/characteristic_function.hpp"

#include <cmath>
#include <complex>

namespace heston {

namespace {
  using cd = std::complex<double>;
}

std::complex<double> HestonCharacteristicFunction::log_spot_cf(
    std::complex<double> u,
    double t,
    const Market& mkt,
    const HestonParams& p) const
{
  if (!(t > 0.0) || !mkt.is_valid_basic() || !p.is_valid_basic()) {
    throw InvalidInput("log_spot_cf: invalid inputs");
  }

  const double s0 = mkt.s0;
  if (!(s0 > 0.0) || !std::isfinite(s0)) {
    throw InvalidInput("log_spot_cf: s0 must be finite and > 0");
  }

  const double x0 = std::log(s0);
  const double kappa = p.kappa;
  const double theta = p.theta;
  const double sigma = p.sigma;
  const double rho   = p.rho;
  const double v0    = p.v0;
  const double r = mkt.r;
  const double q = mkt.q;

  const cd i(0.0, 1.0);
  const cd iu = i * u;

  const double a = kappa * theta;
  const double b = kappa; // lambda = 0

  const cd beta = rho * sigma * iu - b;
  const cd d = std::sqrt(beta * beta + sigma * sigma * (iu + u * u));

  const cd denom = (b - rho * sigma * iu + d);
  const cd numer = (b - rho * sigma * iu - d);
  const cd g = numer / denom;

  const cd exp_minus_dt = std::exp(-d * t);
  const cd one_minus_g_exp = (cd(1.0, 0.0) - g * exp_minus_dt);
  const cd one_minus_g     = (cd(1.0, 0.0) - g);

  const double eps = 1e-14;
  if (std::abs(one_minus_g_exp) < eps ||
      std::abs(one_minus_g) < eps ||
      std::abs(denom) < eps) {
    throw NumericFailure("log_spot_cf: numerical instability");
  }

  const cd C = (r - q) * iu * t
             + (a / (sigma * sigma)) *
               ((b - rho * sigma * iu - d) * t
                - cd(2.0, 0.0) * std::log(one_minus_g_exp / one_minus_g));

  const cd D = ((b - rho * sigma * iu - d) / (sigma * sigma))
             * ((cd(1.0, 0.0) - exp_minus_dt) / one_minus_g_exp);

  return std::exp(C + D * v0 + iu * x0);
}

} // namespace heston