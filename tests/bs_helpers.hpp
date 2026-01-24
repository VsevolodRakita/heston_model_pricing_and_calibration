#pragma once
#include <algorithm>
#include <cmath>

namespace test_helpers {

inline double norm_cdf(double x) {
  // Abramowitz-Stegun via erf
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double bs_call(double S0, double K, double T, double r, double q, double vol) {
  if (T <= 0.0) return std::max(S0 - K, 0.0);
  if (vol <= 0.0) {
    const double fwd = S0 * std::exp((r - q) * T);
    return std::exp(-r * T) * std::max(fwd - K, 0.0);
  }
  const double sig_sqrtT = vol * std::sqrt(T);
  const double d1 = (std::log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sig_sqrtT;
  const double d2 = d1 - sig_sqrtT;
  return S0 * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

inline double bs_put(double S0, double K, double T, double r, double q, double vol) {
  const double call = bs_call(S0, K, T, r, q, vol);
  // put-call parity
  return call - (S0 * std::exp(-q * T) - K * std::exp(-r * T));
}

} // namespace test_helpers