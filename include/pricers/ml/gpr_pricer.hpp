#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

#include "ml/gaussian_process.hpp"
#include "models/heston/market.hpp"
#include "pricers/pricer.hpp"

namespace heston {

// Gaussian-process surrogate for the Heston European price, following
// De Spiegeleer, Madan, Reyners & Schoutens (2018), §3.2.1.
//
// Trained offline against any IVanillaPricer over a box in the seven features
//   (K, T, v0, kappa, theta, sigma, rho),
// it then prices new points with a cheap kernel-vector product. Because it
// implements IVanillaPricer, it is a drop-in engine for calibration or client
// code. The market (s0, r, q) is held at a fixed reference supplied in the
// config; puts are returned from the (call-trained) surrogate via exact
// put-call parity, so the model stays arbitrage-consistent with a single GP.
//
// The predictive standard deviation from the GP is reported through the
// standard PriceDiagnostics.stdError channel — mirroring the Monte Carlo pricer.
class GprPricer final : public IVanillaPricer {
public:
  // Inclusive [lo, hi] ranges for the seven trained features.
  struct Box {
    double K_lo = 0.0, K_hi = 0.0;
    double T_lo = 0.0, T_hi = 0.0;
    double v0_lo = 0.0, v0_hi = 0.0;
    double kappa_lo = 0.0, kappa_hi = 0.0;
    double theta_lo = 0.0, theta_hi = 0.0;
    double sigma_lo = 0.0, sigma_hi = 0.0;
    double rho_lo = 0.0, rho_hi = 0.0;
  };

  struct Config {
    Box box{};
    Market reference{};                  // fixed (s0, r, q) the surrogate is trained for
    std::size_t n_train = 1500;          // number of sampled training points
    std::uint64_t sample_seed = 20240607;
    double market_tol = 1e-9;            // tolerance validating the passed-in market
    GaussianProcessRegressor::Settings gp{};
  };

  explicit GprPricer(const Config& cfg);

  // Train the surrogate by pricing sampled box points with `truth`
  // (e.g. a HestonFourierPricer). Must be called before price().
  void train(const IVanillaPricer& truth);

  [[nodiscard]] double price(const VanillaOption& opt, const Market& mkt,
                             const HestonParams& p) const override;

  // Predictive std (and any extrapolation note) for the most recent price() call.
  [[nodiscard]] PriceDiagnostics diagnostics() const override { return last_diag_; }

  [[nodiscard]] bool is_trained() const noexcept { return trained_; }
  [[nodiscard]] const Config& config() const noexcept { return cfg_; }

private:
  Config cfg_;
  GaussianProcessRegressor gp_;
  bool trained_ = false;
  mutable PriceDiagnostics last_diag_{};

  [[nodiscard]] std::array<double, 7> features_(const VanillaOption& opt,
                                                const HestonParams& p) const;
  [[nodiscard]] bool in_box_(const std::array<double, 7>& f) const;
};

}  // namespace heston
