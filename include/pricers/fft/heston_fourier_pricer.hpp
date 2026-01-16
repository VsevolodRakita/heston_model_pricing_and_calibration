#pragma once
#include <cstddef>
#include <complex>

#include "pricers/pricer.hpp"
#include "models/heston/characteristic_function.hpp"
#include "utils/errors.hpp"

namespace heston {

class HestonFourierPricer final : public IVanillaPricer {
public:
  struct Settings {
    double alpha = 1.5;                    // damping factor alpha > 0
    double u_max = 200.0;                  // integrate u in [0, u_max]
    std::size_t n_intervals_even = 10'000; // Simpson intervals (even)

    // Optional numerical guardrails
    double min_price = 0.0;
    double max_price_multiple_of_spot = 10.0;
  };

  HestonFourierPricer();
  explicit HestonFourierPricer(const Settings& s);
  HestonFourierPricer(const Settings& s, const HestonCharacteristicFunction::Settings& cfs);

  [[nodiscard]] double price(
      const VanillaOption& opt,
      const Market& mkt,
      const HestonParams& p) const override;

  [[nodiscard]] const Settings& settings() const noexcept { return settings_; }

private:
  Settings settings_;
  HestonCharacteristicFunction cf_;

  [[nodiscard]] double call_price_carr_madan_(
      double log_k, double t,
      const Market& mkt,
      const HestonParams& p) const;

  [[nodiscard]] std::complex<double> psi_(
      std::complex<double> u,
      double log_k, double t,
      const Market& mkt,
      const HestonParams& p) const;
};

} // namespace heston