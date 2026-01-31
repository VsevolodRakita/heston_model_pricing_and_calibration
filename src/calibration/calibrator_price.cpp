#include "calibration/calibrator_price.hpp"

#include <cmath>

#include "calibration/objective_price.hpp"
#include "calibration/param_transform.hpp"
#include "utils/errors.hpp"

namespace heston {

CalibrationReport calibrate_heston_to_prices_cmaes(
    const std::vector<PriceQuote>& quotes,
    const Market& mkt,
    const HestonParams& initial_guess,
    const PriceCalibrationSettings& settings) {

  if (!mkt.is_valid_basic()) throw InvalidInput("calibrate_heston_to_prices_cmaes: invalid market");
  if (quotes.empty()) throw InvalidInput("calibrate_heston_to_prices_cmaes: empty quotes");
  for (const auto& q : quotes) {
    if (!q.is_valid_basic()) throw InvalidInput("calibrate_heston_to_prices_cmaes: invalid quote");
  }

  if (!initial_guess.is_finite()) throw InvalidInput("calibrate_heston_to_prices_cmaes: initial guess not finite");
  if (settings.enforce_basic_param_validity && !initial_guess.is_valid_basic()) {
    throw InvalidInput("calibrate_heston_to_prices_cmaes: initial guess fails basic validity");
  }

  const std::vector<double> x0 = unconstrained_from_heston(initial_guess);

  CalibrationReport rep{};

  std::size_t iter_counter = 0;

  auto loss_fn = [&](const std::vector<double>& x) -> double {
    ++iter_counter;

    // penalty-safe objective
    HestonParams p{0.04, 1.5, 0.04, 0.6, -0.7};
    try {
      p = heston_from_unconstrained(x);
    } catch (...) {
      return 1e50;
    }

    if (!p.is_finite()) return 1e50;
    if (settings.enforce_basic_param_validity && !p.is_valid_basic()) return 1e50;

    double loss = 1e50;
    try {
      loss = evaluate_price_objective(quotes, mkt, p, settings.pricer).loss;
      if (!std::isfinite(loss)) loss = 1e50;
    } catch (...) {
      loss = 1e50;
    }

    if (settings.enable_trace && settings.trace_every > 0 && (iter_counter % settings.trace_every == 0)) {
      CalibrationTracePoint tp{};
      tp.iter = iter_counter;
      tp.params = p;
      tp.loss = loss;
      rep.trace.push_back(tp);
    }

    return loss;
  };

  const auto opt = cma_es_minimize(loss_fn, x0, settings.cma);

  rep.params = heston_from_unconstrained(opt.x_best);
  rep.loss = opt.f_best;
  rep.iters = opt.iters;

  // Final residuals at best
  const auto final_obj = evaluate_price_objective(quotes, mkt, rep.params, settings.pricer);
  rep.final_residuals = final_obj.residuals;

  return rep;
}

} // namespace heston