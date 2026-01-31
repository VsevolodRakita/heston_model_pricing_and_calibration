#pragma once
#include <vector>

#include "calibration/calibration_report.hpp"
#include "calibration/price_quote.hpp"
#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "optimization/cma_es.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"

namespace heston {

struct PriceCalibrationSettings {
  CmaEsSettings cma;
  HestonFourierPricer pricer = HestonFourierPricer{};

  bool enforce_basic_param_validity = true;

  // Trace controls
  bool enable_trace = true;
  std::size_t trace_every = 1; // record every N iterations (1 = every iter)
};

CalibrationReport calibrate_heston_to_prices_cmaes(
    const std::vector<PriceQuote>& quotes,
    const Market& mkt,
    const HestonParams& initial_guess,
    const PriceCalibrationSettings& settings = PriceCalibrationSettings{});

} // namespace heston