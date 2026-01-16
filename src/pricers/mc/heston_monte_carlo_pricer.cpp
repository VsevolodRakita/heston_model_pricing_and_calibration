#include "pricers/mc/heston_monte_carlo_pricer.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace heston {

namespace {

inline double discount(double r, double t) { return std::exp(-r * t); }

// Standard normal CDF using erfc (stable, no <numbers> required).
inline double normal_cdf(double z) {
  return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

struct OnlineStats {
  std::size_t n = 0;
  double mean = 0.0;
  double m2 = 0.0;

  void add(double x) {
    ++n;
    const double delta = x - mean;
    mean += delta / static_cast<double>(n);
    const double delta2 = x - mean;
    m2 += delta * delta2;
  }

  double variance() const {
    return (n > 1) ? (m2 / static_cast<double>(n - 1)) : 0.0;
  }

  double std_error() const {
    if (n == 0) return 0.0;
    return std::sqrt(variance() / static_cast<double>(n));
  }
};

} // namespace

double HestonMonteCarloPricer::log_spot_step_(
    double x,
    double integrated_var,
    double dt,
    double r,
    double q,
    double z_s)
{
  // x_{t+dt} = x_t + (r-q)dt - 0.5*I + sqrt(I)*Z
  const double I = std::max(integrated_var, 0.0);
  const double drift = (r - q) * dt - 0.5 * I;
  const double diff  = std::sqrt(I) * z_s;
  return x + drift + diff;
}

VarianceMoments HestonMonteCarloPricer::variance_moments_(
    double v, double dt, const HestonParams& p)
{
  const double kappa = p.kappa;
  const double theta = p.theta;
  const double sigma = p.sigma;

  const double ex = std::exp(-kappa * dt);

  VarianceMoments out;
  out.m = theta + (v - theta) * ex;

  out.s2 =
      (v * sigma * sigma * ex / kappa) * (1.0 - ex)
    + (theta * sigma * sigma / (2.0 * kappa)) * (1.0 - ex) * (1.0 - ex);

  return out;
}

double HestonMonteCarloPricer::qe_variance_step_(
    double v,
    double dt,
    const HestonParams& p,
    double psi_c,
    double zv,
    double zu)
{
  const auto mom = variance_moments_(v, dt, p);
  const double m = mom.m;
  const double s2 = mom.s2;

  if (!std::isfinite(m) || !std::isfinite(s2) || s2 < 0.0) {
    throw NumericFailure("qe_variance_step_: invalid moments");
  }
  if (m <= 0.0) return 0.0;

  const double psi = s2 / (m * m);

  // Regime 1: quadratic form in a normal (depends on zv)
  if (psi <= psi_c) {
    const double term = 2.0 / psi;
    const double root = std::sqrt(std::max(term - 1.0, 0.0));
    const double b2 = term - 1.0 + std::sqrt(term) * root;
    const double a  = m / (1.0 + b2);
    const double b  = std::sqrt(std::max(b2, 0.0));

    const double v_next = a * (b + zv) * (b + zv);
    return std::max(v_next, 0.0);
  }

  // Regime 2: mixture (0 with prob p0, else exponential)
  // IMPORTANT: we derive the uniform used for the branch decision from zv via Φ(zv),
  // so the variance step still depends on the same “variance shock” normal.
  const double p0 = (psi - 1.0) / (psi + 1.0);
  const double beta = (1.0 - p0) / m;

  const double U = normal_cdf(zv);  // in (0,1)
  if (U <= p0) return 0.0;

  // Use independent zu for the exponential draw, also via Φ(zu) for antithetic symmetry.
  double U2 = normal_cdf(zu);
  // Protect against log(0)
  U2 = std::clamp(U2, 1e-16, 1.0 - 1e-16);

  const double v_next = -std::log(U2) / beta;
  return std::max(v_next, 0.0);
}

HestonMonteCarloPricer::Result HestonMonteCarloPricer::price_with_error(
    const VanillaOption& opt,
    const Market& mkt,
    const HestonParams& p) const
{
  if (!opt.is_valid_basic() || !mkt.is_valid_basic() || !p.is_valid_basic()) {
    throw InvalidInput("HestonMonteCarloPricer::price_with_error: invalid inputs");
  }
  if (settings_.n_paths < 1 || settings_.n_steps < 1) {
    throw InvalidInput("HestonMonteCarloPricer::price_with_error: invalid settings");
  }
  if (!(opt.K > 0.0) || !(opt.T > 0.0)) {
    throw InvalidInput("HestonMonteCarloPricer::price_with_error: K and T must be > 0");
  }

  const double dt = opt.T / static_cast<double>(settings_.n_steps);

  const double rho = p.rho;
  const double sqrt_one_minus_rho2 = std::sqrt(std::max(0.0, 1.0 - rho * rho));

  std::mt19937_64 rng(settings_.seed);
  std::normal_distribution<double> norm(0.0, 1.0);

  OnlineStats stats;

  const std::size_t n_outer = settings_.antithetic ? (settings_.n_paths / 2) : settings_.n_paths;

  for (std::size_t path = 0; path < n_outer; ++path) {
    // Pre-draw normals so antithetic pair is truly +/- on the same draws.
    std::vector<double> z_v(settings_.n_steps);
    std::vector<double> z_ind(settings_.n_steps);
    std::vector<double> z_u(settings_.n_steps); // extra independent normal for QE mixture regime

    for (std::size_t step = 0; step < settings_.n_steps; ++step) {
      z_v[step] = norm(rng);
      z_ind[step] = norm(rng);
      z_u[step] = norm(rng);
    }

    const int n_rep = settings_.antithetic ? 2 : 1;

    for (int rep = 0; rep < n_rep; ++rep) {
      const double sign = (rep == 0) ? 1.0 : -1.0;

      double v = p.v0;
      double x = std::log(mkt.s0);

      for (std::size_t step = 0; step < settings_.n_steps; ++step) {
        const double zv = sign * z_v[step];
        const double zi = sign * z_ind[step];
        const double zu = sign * z_u[step];

        const double z_s = rho * zv + sqrt_one_minus_rho2 * zi;

        double v_next = v;

        switch (settings_.variance_scheme) {
          case VarianceScheme::QE: {
            v_next = qe_variance_step_(v, dt, p, settings_.psi_c, zv, zu);
            break;
          }
          case VarianceScheme::MilsteinFullTruncation: {
            const double v_pos = std::max(v, 0.0);
            const double dW = std::sqrt(dt) * zv;
            v_next =
                v
              + p.kappa * (p.theta - v_pos) * dt
              + p.sigma * std::sqrt(v_pos) * dW
              + 0.25 * p.sigma * p.sigma * dt * (zv * zv - 1.0);

            v_next = std::max(v_next, settings_.min_variance);
            break;
          }
          case VarianceScheme::FullTruncationEuler: {
            const double v_pos = std::max(v, 0.0);
            v_next =
                v
              + p.kappa * (p.theta - v_pos) * dt
              + p.sigma * std::sqrt(v_pos * dt) * zv;

            v_next = std::max(v_next, settings_.min_variance);
            break;
          }
        }

        // Euler stock
        const double v_pos = std::max(v, 0.0);
        x += (mkt.r - mkt.q - 0.5 * v_pos) * dt
          + std::sqrt(v_pos * dt) * z_s;

        v = v_next;
      }

      const double s_t = std::exp(x);

      double payoff = 0.0;
      if (opt.type == OptionType::Call) payoff = std::max(s_t - opt.K, 0.0);
      else payoff = std::max(opt.K - s_t, 0.0);

      stats.add(payoff);
    }
  }

  const double disc = discount(mkt.r, opt.T);

  Result res;
  res.price = disc * stats.mean;
  res.std_error = disc * stats.std_error();
  res.n_used = stats.n;
  return res;
}

double HestonMonteCarloPricer::price(
    const VanillaOption& opt,
    const Market& mkt,
    const HestonParams& p) const
{
  return price_with_error(opt, mkt, p).price;
}

} // namespace heston