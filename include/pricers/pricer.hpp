#pragma once
#include <vector>
#include "products/vanilla_option.hpp"
#include "models/heston/market.hpp"
#include "models/heston/heston_params.hpp"

namespace heston {

// Optional extra info a pricer may provide.
// Deterministic pricers (FFT) can leave this at defaults.
// Monte Carlo can fill stdError, etc.
struct PriceDiagnostics {
  double stdError = 0.0;          // standard error of estimated price (MC)
  std::string note = {};          // optional human-readable note/warning
};

// Common interface so calibration/surrogates can swap pricers.
class IVanillaPricer {
public:
  virtual ~IVanillaPricer() = default;

  // Price one vanilla option.
  [[nodiscard]] virtual double price(
      const VanillaOption& opt,
      const Market& mkt,
      const HestonParams& p) const = 0;

  // Convenience batch pricing (default loops; override for vectorization/FFT batching).
  [[nodiscard]] virtual std::vector<double> priceBatch(
      const std::vector<VanillaOption>& opts,
      const Market& mkt,
      const HestonParams& p) const
  {
    std::vector<double> out;
    out.reserve(opts.size());
    for (const auto& o : opts) out.push_back(price(o, mkt, p));
    return out;
  }

  // Optional: diagnostics for the most recent pricing call.
  // Default is "no extra info". Monte Carlo pricers can override this.
  [[nodiscard]] virtual PriceDiagnostics diagnostics() const { return {}; }
};

}