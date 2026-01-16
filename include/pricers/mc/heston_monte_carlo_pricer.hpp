#pragma once
#include <cstddef>
#include <cstdint>

#include "pricers/pricer.hpp"      // IVanillaPricer, VanillaOption, OptionType
#include "utils/errors.hpp"        // InvalidInput, NumericFailure

namespace heston {

// Small helper type used by QE moment matching.
struct VarianceMoments {
  double m = 0.0;
  double s2 = 0.0;
};

class HestonMonteCarloPricer final : public IVanillaPricer {
public:
  enum class VarianceScheme {
    FullTruncationEuler,
    MilsteinFullTruncation,
    QE
  };

  struct Settings {
    std::size_t n_paths = 200'000;
    std::size_t n_steps = 200;
    std::uint64_t seed = 42;

    bool antithetic = true;
    VarianceScheme variance_scheme = VarianceScheme::QE;

    // QE regime switch threshold (common default 1.5)
    double psi_c = 1.5;

    // For Euler/Milstein only
    double min_variance = 0.0;
  };

  HestonMonteCarloPricer() = default;
  explicit HestonMonteCarloPricer(const Settings& s) : settings_(s) {}

  [[nodiscard]] double price(
      const VanillaOption& opt,
      const Market& mkt,
      const HestonParams& p) const override;

  struct Result {
    double price = 0.0;
    double std_error = 0.0;
    std::size_t n_used = 0;
  };

  [[nodiscard]] Result price_with_error(
      const VanillaOption& opt,
      const Market& mkt,
      const HestonParams& p) const;

  [[nodiscard]] const Settings& settings() const noexcept { return settings_; }

private:
  Settings settings_;

  [[nodiscard]] static VarianceMoments variance_moments_(
      double v, double dt, const HestonParams& p);

  // QE step for variance; uses zv as the “driver” normal.
  [[nodiscard]] static double qe_variance_step_(
      double v,
      double dt,
      const HestonParams& p,
      double psi_c,
      double zv,
      double zu); // independent extra normal used only when needed

  // Log stock update using integrated variance over the step.
  [[nodiscard]] static double log_spot_step_(
      double x,
      double integrated_var,
      double dt,
      double r,
      double q,
      double z_s);
};

} // namespace heston