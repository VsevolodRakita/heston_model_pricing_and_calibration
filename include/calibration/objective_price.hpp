#pragma once
#include <vector>

#include "calibration/price_quote.hpp"
#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"

namespace heston {

struct ObjectiveResult {
  double loss = 0.0;
  std::vector<double> residuals; // model_price - market_price
};

ObjectiveResult evaluate_price_objective(
    const std::vector<PriceQuote>& quotes,
    const Market& mkt,
    const HestonParams& p,
    const HestonFourierPricer& pricer);

} // namespace heston