#pragma once
#include <cstddef>
#include <vector>

#include "models/heston/heston_params.hpp"

namespace heston {

struct CalibrationTracePoint {
  std::size_t iter = 0;
  HestonParams params;
  double loss = 0.0;
};

struct CalibrationReport {
  HestonParams params;
  double loss = 0.0;
  std::size_t iters = 0;

  std::vector<CalibrationTracePoint> trace;
  std::vector<double> final_residuals; // objective residuals at best params
};

} // namespace heston