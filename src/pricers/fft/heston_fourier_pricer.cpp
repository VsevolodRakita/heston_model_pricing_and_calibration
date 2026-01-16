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

  double price = std::exp(-alpha * log_k) * (integral / pi);

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

} // namespace heston