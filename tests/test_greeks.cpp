#include "third_party/doctest/doctest.h"

#include <cmath>
#include <initializer_list>

#include "greeks/greeks.hpp"
#include "vol/black_scholes.hpp"
#include "models/heston/heston_params.hpp"
#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"

using namespace heston;

namespace {

// Adapter that exposes constant-volatility Black–Scholes through the
// IVanillaPricer interface. The Heston params are ignored, which lets us run
// the generic finite-difference engine against a pricer whose Greeks we know
// in closed form.
struct BsPricer final : IVanillaPricer {
  double vol;
  explicit BsPricer(double v) : vol(v) {}
  [[nodiscard]] double price(
      const VanillaOption& opt, const Market& mkt, const HestonParams&) const override {
    return black_scholes_price(opt, mkt, vol);
  }
};

// Closeness with an absolute floor, so the check stays meaningful when the
// reference value is small (the finite-difference / quadrature noise floor of
// the Fourier pricer is a few 1e-3 on these parameter sensitivities).
bool close_scaled(double a, double b, double rel = 2e-2, double abs_floor = 5e-3) {
  return std::abs(a - b) <= abs_floor + rel * std::abs(b);
}

} // namespace

TEST_CASE("BS closed-form Greeks match a manual finite difference") {
  const Market m{100.0, 0.02, 0.01};
  const double vol = 0.2;
  const VanillaOption call{OptionType::Call, 100.0, 1.0};

  auto price_at_spot = [&](double s) {
    Market mm = m; mm.s0 = s;
    return black_scholes_price(call, mm, vol);
  };

  const double h = 1e-3 * m.s0;
  const double delta_fd = (price_at_spot(m.s0 + h) - price_at_spot(m.s0 - h)) / (2.0 * h);
  const double gamma_fd =
      (price_at_spot(m.s0 + h) - 2.0 * price_at_spot(m.s0) + price_at_spot(m.s0 - h)) / (h * h);

  CHECK(black_scholes_delta(call, m, vol) == doctest::Approx(delta_fd).epsilon(1e-4));
  CHECK(black_scholes_gamma(call, m, vol) == doctest::Approx(gamma_fd).epsilon(1e-2));
}

TEST_CASE("BS put/call Greek parity relations") {
  const Market m{100.0, 0.03, 0.01};
  const double vol = 0.25;
  const VanillaOption call{OptionType::Call, 95.0, 0.75};
  const VanillaOption put {OptionType::Put,  95.0, 0.75};

  const double disc_q = std::exp(-m.q * call.T);
  const double disc_r = std::exp(-m.r * call.T);

  // C - P = S e^{-qT} - K e^{-rT}  =>  differentiate both sides.
  CHECK(black_scholes_delta(call, m, vol) - black_scholes_delta(put, m, vol)
        == doctest::Approx(disc_q).epsilon(1e-10));
  CHECK(black_scholes_gamma(call, m, vol) == doctest::Approx(black_scholes_gamma(put, m, vol)));
  CHECK(black_scholes_rho(call, m, vol) - black_scholes_rho(put, m, vol)
        == doctest::Approx(call.K * call.T * disc_r).epsilon(1e-10));
}

TEST_CASE("Finite-difference engine reproduces analytic BS Greeks") {
  const Market m{100.0, 0.02, 0.01};
  const double vol = 0.2;
  BsPricer bs(vol);
  const HestonParams ignored(0.04, 1.0, 0.04, 0.5, -0.5);

  for (const auto type : {OptionType::Call, OptionType::Put}) {
    const VanillaOption opt{type, 100.0, 1.0};
    const Greeks g = compute_greeks_fd(bs, opt, m, ignored);

    CHECK(g.price == doctest::Approx(black_scholes_price(opt, m, vol)));
    CHECK(g.delta == doctest::Approx(black_scholes_delta(opt, m, vol)).epsilon(1e-3));
    CHECK(g.gamma == doctest::Approx(black_scholes_gamma(opt, m, vol)).epsilon(5e-2));
    CHECK(g.theta == doctest::Approx(black_scholes_theta(opt, m, vol)).epsilon(1e-2));
    CHECK(g.rho   == doctest::Approx(black_scholes_rho(opt, m, vol)).epsilon(1e-3));
  }
}

TEST_CASE("Heston delta/gamma are well behaved for an ATM call") {
  const Market m{100.0, 0.02, 0.0};
  const HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);
  const VanillaOption call{OptionType::Call, 100.0, 1.0};

  HestonFourierPricer pricer;
  const Greeks g = compute_greeks_fd(pricer, call, m, p);

  const double disc_q = std::exp(-m.q * call.T);
  CHECK(g.delta > 0.0);
  CHECK(g.delta < disc_q);      // a call delta cannot exceed e^{-qT}
  CHECK(g.gamma > 0.0);         // vanilla convexity
  CHECK(g.theta < 0.0);         // long option loses value with the passage of time here
}

TEST_CASE("Heston Greeks obey put/call parity") {
  const Market m{100.0, 0.02, 0.01};
  const HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);
  const VanillaOption call{OptionType::Call, 100.0, 1.0};
  const VanillaOption put {OptionType::Put,  100.0, 1.0};

  HestonFourierPricer::Settings s;
  s.u_max = 250.0;
  s.n_intervals_even = 12'000;
  HestonFourierPricer pricer(s);

  const Greeks gc = compute_greeks_fd(pricer, call, m, p);
  const Greeks gp = compute_greeks_fd(pricer, put,  m, p);

  const double disc_q = std::exp(-m.q * call.T);
  const double disc_r = std::exp(-m.r * call.T);

  CHECK(gc.delta - gp.delta == doctest::Approx(disc_q).epsilon(1e-3));
  CHECK(std::abs(gc.gamma - gp.gamma) < 1e-3);
  CHECK(gc.rho - gp.rho == doctest::Approx(call.K * call.T * disc_r).epsilon(1e-3));
}

TEST_CASE("Analytic Fourier Greeks agree with finite differences") {
  const Market m{100.0, 0.03, 0.01};
  const HestonParams p(0.04, 1.5, 0.04, 0.5, -0.7);

  HestonFourierPricer::Settings s;
  s.u_max = 300.0;
  s.n_intervals_even = 20'000;
  HestonFourierPricer pricer(s);

  for (const double K : {80.0, 100.0, 120.0}) {
    for (const auto type : {OptionType::Call, OptionType::Put}) {
      const VanillaOption opt{type, K, 1.0};

      const HestonFourierPricer::AnalyticGreeks a = pricer.analytic_greeks(opt, m, p);
      const Greeks fd = compute_greeks_fd(pricer, opt, m, p);

      CHECK(a.price == doctest::Approx(pricer.price(opt, m, p)).epsilon(1e-6));
      CHECK(a.delta == doctest::Approx(fd.delta).epsilon(1e-4));
      CHECK(a.gamma == doctest::Approx(fd.gamma).epsilon(1e-3));
      CHECK(a.rho   == doctest::Approx(fd.rho).epsilon(1e-4));
    }
  }
}

TEST_CASE("Analytic Fourier Greeks reduce to Black-Scholes in the low-vol-of-vol limit") {
  const Market m{100.0, 0.03, 0.01};
  const double theta = 0.04;               // BS vol = 0.20
  const HestonParams p(theta, 1.5, theta, 1e-3, 0.0);

  HestonFourierPricer::Settings s;
  s.u_max = 300.0;
  s.n_intervals_even = 20'000;
  HestonFourierPricer pricer(s);

  const double vol = std::sqrt(theta);
  for (const auto type : {OptionType::Call, OptionType::Put}) {
    const VanillaOption opt{type, 100.0, 1.0};
    const HestonFourierPricer::AnalyticGreeks a = pricer.analytic_greeks(opt, m, p);

    CHECK(a.delta == doctest::Approx(black_scholes_delta(opt, m, vol)).epsilon(2e-3));
    CHECK(a.gamma == doctest::Approx(black_scholes_gamma(opt, m, vol)).epsilon(2e-3));
    CHECK(a.rho   == doctest::Approx(black_scholes_rho(opt, m, vol)).epsilon(2e-3));
  }
}

TEST_CASE("Heston parameter sensitivities: vega positive and parity-invariant") {
  const Market m{100.0, 0.02, 0.01};
  const HestonParams p(0.04, 2.0, 0.04, 0.5, -0.7);
  const VanillaOption call{OptionType::Call, 100.0, 1.0};
  const VanillaOption put {OptionType::Put,  100.0, 1.0};

  HestonFourierPricer::Settings s;
  s.u_max = 250.0;
  s.n_intervals_even = 12'000;
  HestonFourierPricer pricer(s);

  const HestonParamSensitivities sc = compute_param_sensitivities_fd(pricer, call, m, p);
  const HestonParamSensitivities sp = compute_param_sensitivities_fd(pricer, put,  m, p);

  // More initial variance is worth more to a vanilla option: the Heston "vega".
  CHECK(sc.d_v0 > 0.0);

  // C - P is independent of the Heston parameters, so every sensitivity must
  // agree between the call and the put.
  CHECK(close_scaled(sc.d_v0,    sp.d_v0));
  CHECK(close_scaled(sc.d_kappa, sp.d_kappa));
  CHECK(close_scaled(sc.d_theta, sp.d_theta));
  CHECK(close_scaled(sc.d_sigma, sp.d_sigma));
  CHECK(close_scaled(sc.d_rho,   sp.d_rho));
}
