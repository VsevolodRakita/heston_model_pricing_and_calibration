#pragma once
#include <vector>

#include "calibration/iv_quote.hpp"
#include "models/heston/market.hpp"
#include "models/heston/heston_params.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "optimization/nelder_mead.hpp"

namespace heston {

struct CalibrationSettings {
  NelderMeadSettings nm;
  HestonFourierPricer fourier_pricer = HestonFourierPricer{};
};

struct CalibrationResult {
  HestonParams params;
  double loss = 0.0;
  std::size_t iters = 0;

  CalibrationResult(const HestonParams& p, double l, std::size_t its): params(p),loss(l),iters(its){}
};

CalibrationResult calibrate_heston_to_iv(
    const std::vector<IvQuote>& quotes,
    const Market& mkt,
    const HestonParams& initial_guess,
    const CalibrationSettings& settings);

} // namespace heston 