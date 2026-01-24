#pragma once
#include <vector>
#include "calibration/iv_quote.hpp"
#include "models/heston/market.hpp"
#include "models/heston/heston_params.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"

namespace heston {

struct ObjectiveResult {
  double loss = 0.0;
  std::vector<double> residuals; // model_iv - mkt_iv
};

ObjectiveResult evaluate_iv_objective(
    const std::vector<IvQuote>& quotes,
    const Market& mkt,
    const HestonParams& p,
    const HestonFourierPricer& fourier_pricer);

} // namespace heston