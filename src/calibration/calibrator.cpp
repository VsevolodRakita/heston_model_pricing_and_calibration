#include "calibration/calibrator.hpp"

#include "calibration/objective.hpp"
#include "calibration/param_transform.hpp"
#include "utils/errors.hpp"

namespace heston {

CalibrationResult calibrate_heston_to_iv(
    const std::vector<IvQuote>& quotes,
    const Market& mkt,
    const HestonParams& initial_guess,
    const CalibrationSettings& settings) {

  if (!mkt.is_valid_basic()) throw InvalidInput("calibrate_heston_to_iv: invalid market");
  if (quotes.empty()) throw InvalidInput("calibrate_heston_to_iv: empty quotes");
  if (!initial_guess.is_finite()) throw InvalidInput("calibrate_heston_to_iv: initial guess not finite");

  // Convert to unconstrained x0
  const std::vector<double> x0 = unconstrained_from_heston(initial_guess);

  // Objective function in unconstrained variables
  auto f = [&](const std::vector<double>& x) -> double {
    const HestonParams p = heston_from_unconstrained(x);
    const auto res = evaluate_iv_objective(quotes, mkt, p, settings.fourier_pricer);
    return res.loss;
  };

  const auto opt = nelder_mead_minimize(f, x0, settings.nm);

  CalibrationResult out(heston_from_unconstrained(opt.x_best), opt.f_best, opt.iters);
  return out;
}

} // namespace heston