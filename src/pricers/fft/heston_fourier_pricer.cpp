#include "pricers/fft/heston_fourier_pricer.hpp"

#include <cmath>
#include <complex>

#include "utils/numerics.hpp"

namespace heston {

namespace {
  using cd = std::complex<double>;
  constexpr double pi = 3.141592653589793238462643383279502884;

  [[nodiscard]] inline double discount(double r, double t) {
    return std::exp(-r * t);
  }

  [[nodiscard]] inline double discount_div(double q, double t) {
    return std::exp(-q * t);
  }
} // namespace

HestonFourierPricer::HestonFourierPricer()
  : settings_(Settings{}), cf_(HestonCharacteristicFunction::Settings{}) {}

HestonFourierPricer::HestonFourierPricer(const Settings& s)
  : settings_(s), cf_(HestonCharacteristicFunction::Settings{}) {}

HestonFourierPricer::HestonFourierPricer(
    const Settings& s,
    const HestonCharacteristicFunction::Settings& cfs)
  : settings_(s), cf_(cfs) {}

double HestonFourierPricer::price(
    const VanillaOption& opt,
    const Market& mkt,
    const HestonParams& p) const
{
  if (!opt.is_valid_basic() || !mkt.is_valid_basic() || !p.is_valid_basic()) {
    throw InvalidInput("HestonFourierPricer::price: invalid inputs");
  }

  if (!(settings_.alpha > 0.0) || !(settings_.u_max > 0.0) || settings_.n_intervals_even < 2) {
    throw InvalidInput("HestonFourierPricer::price: invalid settings");
  }

  const double t = opt.T;
  const double k = opt.K;

  if (!(k > 0.0) || !(t > 0.0)) {
    throw InvalidInput("HestonFourierPricer::price: K and T must be > 0");
  }

  const double log_k = std::log(k);
  const double call = call_price_carr_madan_(log_k, t, mkt, p);

  double out = 0.0;
  if (opt.type == OptionType::Call) {
    out = call;
  } else {
    const double rhs = mkt.s0 * discount_div(mkt.q, t) - k * discount(mkt.r, t);
    out = call - rhs;
  }

  if (out < settings_.min_price) out = settings_.min_price;

  if (settings_.max_price_multiple_of_spot > 0.0) {
    const double max_price = settings_.max_price_multiple_of_spot * mkt.s0;
    if (out > max_price) out = max_price;
  }

  return out;
}

double HestonFourierPricer::call_price_carr_madan_(
    double log_k,
    double t,
    const Market& mkt,
    const HestonParams& p) const
{
  if (!(t > 0.0) || !std::isfinite(log_k)) {
    throw InvalidInput("call_price_carr_madan_: invalid inputs");
  }

  std::size_t n = settings_.n_intervals_even;
  if (n % 2 == 1) ++n;

  const double alpha = settings_.alpha;
  const double u_max = settings_.u_max;

  auto integrand = [&](double u_real) -> double {
    const cd u(u_real, 0.0);
    return psi_(u, log_k, t, mkt, p).real();
  };

  const double integral = simpson(integrand, 0.0, u_max, n);

  // Carr–Madan call price. The characteristic function carries the (r - q)
  // forward drift but NOT discounting, so the e^{-rT} factor is applied here.
  double price = std::exp(-mkt.r * t) * std::exp(-alpha * log_k) * (integral / pi);

  if (price < 0.0 && price > -1e-10) price = 0.0;
  if (!std::isfinite(price)) {
    throw NumericFailure("call_price_carr_madan_: non-finite price");
  }

  return price;
}

std::complex<double> HestonFourierPricer::psi_(
    std::complex<double> u,
    double log_k,
    double t,
    const Market& mkt,
    const HestonParams& p) const
{
  const double alpha = settings_.alpha;
  const cd i(0.0, 1.0);

  const cd u_shift = u - i * (alpha + 1.0);
  const cd phi = cf_.log_spot_cf(u_shift, t, mkt, p);

  const cd denom = (alpha * alpha + alpha - u * u) + i * (2.0 * alpha + 1.0) * u;
  if (std::abs(denom) < 1e-16) {
    throw NumericFailure("psi_: denominator too small");
  }

  const cd expo = std::exp(-i * u * log_k);
  return (expo * phi) / denom;
}

HestonFourierPricer::AnalyticGreeks HestonFourierPricer::analytic_greeks(
    const VanillaOption& opt,
    const Market& mkt,
    const HestonParams& p) const
{
  if (!opt.is_valid_basic() || !mkt.is_valid_basic() || !p.is_valid_basic()) {
    throw InvalidInput("HestonFourierPricer::analytic_greeks: invalid inputs");
  }
  if (!(settings_.alpha > 0.0) || !(settings_.u_max > 0.0) || settings_.n_intervals_even < 2) {
    throw InvalidInput("HestonFourierPricer::analytic_greeks: invalid settings");
  }

  const double t = opt.T;
  const double k = opt.K;
  if (!(k > 0.0) || !(t > 0.0)) {
    throw InvalidInput("HestonFourierPricer::analytic_greeks: K and T must be > 0");
  }

  const double log_k = std::log(k);
  const double alpha = settings_.alpha;
  const double u_max = settings_.u_max;
  const double s0    = mkt.s0;

  std::size_t n = settings_.n_intervals_even;
  if (n % 2 == 1) ++n;
  const double h = u_max / static_cast<double>(n);

  const cd i(0.0, 1.0);

  // One composite-Simpson pass accumulates three integrals that share a single
  // characteristic-function evaluation per node:
  //   I0 : integrand                 -> call price
  //   I1 : integrand * (i w)         -> dC/d(log S0)
  //   I2 : integrand * (-(w^2))      -> d2C/d(log S0)^2
  // where w = u - i(alpha+1) is the (shifted) transform variable, and the log
  // spot enters the integrand only through the characteristic function's
  // exp(i w log S0) factor.
  double I0 = 0.0, I1 = 0.0, I2 = 0.0;

  for (std::size_t j = 0; j <= n; ++j) {
    const double u = static_cast<double>(j) * h;
    const cd w(u, -(alpha + 1.0));

    const cd denom = (alpha * alpha + alpha - u * u) + i * (2.0 * alpha + 1.0) * u;
    if (std::abs(denom) < 1e-16) {
      throw NumericFailure("analytic_greeks: denominator too small");
    }

    const cd phi  = cf_.log_spot_cf(w, t, mkt, p);
    const cd expo = std::exp(-i * u * log_k);
    const cd f0   = (expo * phi) / denom;
    const cd f1   = (i * w) * f0;
    const cd f2   = (-(w * w)) * f0;

    double sw;  // composite-Simpson weight
    if (j == 0 || j == n) sw = 1.0;
    else if (j % 2 == 1)  sw = 4.0;
    else                  sw = 2.0;

    I0 += sw * f0.real();
    I1 += sw * f1.real();
    I2 += sw * f2.real();
  }

  const double scale = h / 3.0;
  I0 *= scale; I1 *= scale; I2 *= scale;

  const double pref   = std::exp(-mkt.r * t) * std::exp(-alpha * log_k) / pi;
  const double C      = pref * I0;   // call price
  const double dCdx0  = pref * I1;   // dC/d(log S0) = S0 * delta_call
  const double d2Cdx0 = pref * I2;   // d2C/d(log S0)^2

  const double delta_call = dCdx0 / s0;
  const double gamma_call = (d2Cdx0 - dCdx0) / (s0 * s0);
  const double rho_call   = t * (dCdx0 - C);   // = t*(S0*delta_call - C)

  AnalyticGreeks g;
  if (opt.type == OptionType::Call) {
    g.price = C;
    g.delta = delta_call;
    g.gamma = gamma_call;
    g.rho   = rho_call;
  } else {
    // Put via put–call parity: P = C - (S0 e^{-qT} - K e^{-rT}).
    const double disc_q = discount_div(mkt.q, t);
    const double disc_r = discount(mkt.r, t);
    g.price = C - (s0 * disc_q - k * disc_r);
    g.delta = delta_call - disc_q;
    g.gamma = gamma_call;
    g.rho   = rho_call - k * t * disc_r;
  }
  return g;
}

} // namespace heston