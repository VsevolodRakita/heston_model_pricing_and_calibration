#include "pricers/ml/gpr_pricer.hpp"

#include <cmath>
#include <vector>

#include "utils/errors.hpp"
#include "utils/rng.hpp"

namespace heston {

GprPricer::GprPricer(const Config& cfg) : cfg_(cfg), gp_(cfg.gp) {}

std::array<double, 7> GprPricer::features_(const VanillaOption& opt, const HestonParams& p) const {
  return {opt.K, opt.T, p.v0, p.kappa, p.theta, p.sigma, p.rho};
}

bool GprPricer::in_box_(const std::array<double, 7>& f) const {
  const Box& b = cfg_.box;
  const double lo[7] = {b.K_lo, b.T_lo, b.v0_lo, b.kappa_lo, b.theta_lo, b.sigma_lo, b.rho_lo};
  const double hi[7] = {b.K_hi, b.T_hi, b.v0_hi, b.kappa_hi, b.theta_hi, b.sigma_hi, b.rho_hi};
  for (std::size_t k = 0; k < 7; ++k) {
    const double slack = 1e-9 * (std::abs(hi[k]) + std::abs(lo[k]) + 1.0);
    if (f[k] < lo[k] - slack || f[k] > hi[k] + slack) return false;
  }
  return true;
}

void GprPricer::train(const IVanillaPricer& truth) {
  const Box& b = cfg_.box;
  Rng rng(cfg_.sample_seed);

  std::vector<std::vector<double>> X;
  std::vector<double> y;
  X.reserve(cfg_.n_train);
  y.reserve(cfg_.n_train);

  for (std::size_t i = 0; i < cfg_.n_train; ++i) {
    const double K = rng.uniform(b.K_lo, b.K_hi);
    const double T = rng.uniform(b.T_lo, b.T_hi);
    const double v0 = rng.uniform(b.v0_lo, b.v0_hi);
    const double kappa = rng.uniform(b.kappa_lo, b.kappa_hi);
    const double theta = rng.uniform(b.theta_lo, b.theta_hi);
    const double sigma = rng.uniform(b.sigma_lo, b.sigma_hi);
    const double rho = rng.uniform(b.rho_lo, b.rho_hi);

    const VanillaOption call{OptionType::Call, K, T};
    const HestonParams p(v0, kappa, theta, sigma, rho);

    // Train on calls only; puts are recovered analytically via parity.
    const double px = truth.price(call, cfg_.reference, p);

    X.push_back({K, T, v0, kappa, theta, sigma, rho});
    y.push_back(px);
  }

  gp_.fit(X, y);
  trained_ = true;
}

double GprPricer::price(const VanillaOption& opt, const Market& mkt, const HestonParams& p) const {
  if (!trained_) throw InvalidInput("GprPricer::price: surrogate has not been trained");

  // The surrogate is only valid for the market it was trained on.
  const Market& ref = cfg_.reference;
  if (std::abs(mkt.s0 - ref.s0) > cfg_.market_tol || std::abs(mkt.r - ref.r) > cfg_.market_tol ||
      std::abs(mkt.q - ref.q) > cfg_.market_tol) {
    throw InvalidInput(
        "GprPricer::price: market differs from the reference the surrogate was trained for");
  }

  const std::array<double, 7> f = features_(opt, p);
  last_diag_ = PriceDiagnostics{};
  if (!in_box_(f)) {
    last_diag_.note = "GprPricer: query is outside the training box (extrapolation).";
  }

  const std::vector<double> x(f.begin(), f.end());
  const GaussianProcessRegressor::Prediction pred = gp_.predict(x);

  double call_px = pred.mean;
  if (call_px < 0.0) call_px = 0.0;  // prices are non-negative

  last_diag_.stdError = std::sqrt(pred.variance);

  if (opt.type == OptionType::Call) return call_px;

  // Put via put-call parity: P = C - (S e^{-qT} - K e^{-rT}).
  const double disc_r = std::exp(-mkt.r * opt.T);
  const double disc_q = std::exp(-mkt.q * opt.T);
  double put_px = call_px - (mkt.s0 * disc_q - opt.K * disc_r);
  if (put_px < 0.0) put_px = 0.0;
  return put_px;
}

}  // namespace heston
