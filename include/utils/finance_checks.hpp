#pragma once
#include <cmath>
#include "products/vanilla_option.hpp"
#include "models/heston/Market.hpp"

namespace heston {

// Put-call parity residual: (C - P) - (S e^{-qT} - K e^{-rT})
inline double putCallParityResidual(
    double callPrice,
    double putPrice,
    const VanillaOption& opt, // same K,T
    const Market& mkt)
{
  const double discR = std::exp(-mkt.r * opt.T);
  const double discQ = std::exp(-mkt.q * opt.T);
  return (callPrice - putPrice) - (mkt.s0 * discQ - opt.K * discR);
}

} // namespace heston