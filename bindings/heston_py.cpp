// Python bindings for the Heston pricing/calibration library.
//
// Build with -DHESTON_BUILD_PYTHON=ON; produces an importable module `heston`.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <cstdint>
#include <utility>

#include "models/heston/market.hpp"
#include "models/heston/heston_params.hpp"
#include "products/vanilla_option.hpp"
#include "pricers/fft/heston_fourier_pricer.hpp"
#include "pricers/mc/heston_monte_carlo_pricer.hpp"
#include "pricers/ml/gpr_pricer.hpp"
#include "vol/black_scholes.hpp"
#include "vol/implied_vol.hpp"
#include "greeks/greeks.hpp"
#include "calibration/price_quote.hpp"
#include "calibration/calibrator_price.hpp"

namespace py = pybind11;
using namespace heston;

PYBIND11_MODULE(heston, m) {
  m.doc() = "Heston stochastic-volatility option pricing, Greeks and calibration";

  py::enum_<OptionType>(m, "OptionType")
      .value("Call", OptionType::Call)
      .value("Put", OptionType::Put);

  py::class_<Market>(m, "Market")
      .def(py::init([](double s0, double r, double q) { return Market{s0, r, q}; }),
           py::arg("s0"), py::arg("r") = 0.0, py::arg("q") = 0.0)
      .def_readwrite("s0", &Market::s0)
      .def_readwrite("r", &Market::r)
      .def_readwrite("q", &Market::q)
      .def("__repr__", [](const Market& m) {
        return "Market(s0=" + std::to_string(m.s0) + ", r=" + std::to_string(m.r) +
               ", q=" + std::to_string(m.q) + ")";
      });

  py::class_<HestonParams>(m, "HestonParams")
      .def(py::init([](double v0, double kappa, double theta, double sigma, double rho) {
             return HestonParams(v0, kappa, theta, sigma, rho);
           }),
           py::arg("v0"), py::arg("kappa"), py::arg("theta"), py::arg("sigma"), py::arg("rho"))
      .def_readwrite("v0", &HestonParams::v0)
      .def_readwrite("kappa", &HestonParams::kappa)
      .def_readwrite("theta", &HestonParams::theta)
      .def_readwrite("sigma", &HestonParams::sigma)
      .def_readwrite("rho", &HestonParams::rho)
      .def("is_valid_basic", &HestonParams::is_valid_basic)
      .def("satisfies_feller", &HestonParams::satisfies_feller)
      .def("__repr__", [](const HestonParams& p) { return p.to_string(); });

  py::class_<VanillaOption>(m, "VanillaOption")
      .def(py::init([](OptionType type, double K, double T) {
             return VanillaOption{type, K, T};
           }),
           py::arg("type"), py::arg("K"), py::arg("T"))
      .def_readwrite("type", &VanillaOption::type)
      .def_readwrite("K", &VanillaOption::K)
      .def_readwrite("T", &VanillaOption::T);

  // Common pricer base + diagnostics, so engines are interchangeable in Python
  // (e.g. any engine can be handed to GprPricer.train).
  py::class_<IVanillaPricer>(m, "IVanillaPricer");

  py::class_<PriceDiagnostics>(m, "PriceDiagnostics")
      .def_readonly("std_error", &PriceDiagnostics::stdError)
      .def_readonly("note", &PriceDiagnostics::note);

  // ---- Fourier pricer -------------------------------------------------------
  py::class_<HestonFourierPricer::AnalyticGreeks>(m, "FourierGreeks")
      .def_readonly("price", &HestonFourierPricer::AnalyticGreeks::price)
      .def_readonly("delta", &HestonFourierPricer::AnalyticGreeks::delta)
      .def_readonly("gamma", &HestonFourierPricer::AnalyticGreeks::gamma)
      .def_readonly("rho", &HestonFourierPricer::AnalyticGreeks::rho);

  py::class_<HestonFourierPricer, IVanillaPricer>(m, "HestonFourierPricer")
      .def(py::init([](double alpha, double u_max, std::size_t n_intervals_even) {
             HestonFourierPricer::Settings s;
             s.alpha = alpha;
             s.u_max = u_max;
             s.n_intervals_even = n_intervals_even;
             return HestonFourierPricer(s);
           }),
           py::arg("alpha") = 1.5, py::arg("u_max") = 200.0,
           py::arg("n_intervals_even") = 10000)
      .def("price", &HestonFourierPricer::price,
           py::arg("option"), py::arg("market"), py::arg("params"))
      .def("analytic_greeks", &HestonFourierPricer::analytic_greeks,
           py::arg("option"), py::arg("market"), py::arg("params"));

  // ---- Monte Carlo pricer ---------------------------------------------------
  py::class_<HestonMonteCarloPricer::Result>(m, "MCResult")
      .def_readonly("price", &HestonMonteCarloPricer::Result::price)
      .def_readonly("std_error", &HestonMonteCarloPricer::Result::std_error)
      .def_readonly("n_used", &HestonMonteCarloPricer::Result::n_used);

  py::class_<HestonMonteCarloPricer, IVanillaPricer>(m, "HestonMonteCarloPricer")
      .def(py::init([](std::size_t n_paths, std::size_t n_steps, std::uint64_t seed) {
             HestonMonteCarloPricer::Settings s;
             s.n_paths = n_paths;
             s.n_steps = n_steps;
             s.seed = seed;
             return HestonMonteCarloPricer(s);
           }),
           py::arg("n_paths") = 200000, py::arg("n_steps") = 200, py::arg("seed") = 42)
      .def("price", &HestonMonteCarloPricer::price,
           py::arg("option"), py::arg("market"), py::arg("params"))
      .def("price_with_error", &HestonMonteCarloPricer::price_with_error,
           py::arg("option"), py::arg("market"), py::arg("params"));

  // ---- Black–Scholes / implied vol -----------------------------------------
  m.def("black_scholes_price", &black_scholes_price, py::arg("option"), py::arg("market"), py::arg("vol"));
  m.def("black_scholes_delta", &black_scholes_delta, py::arg("option"), py::arg("market"), py::arg("vol"));
  m.def("black_scholes_gamma", &black_scholes_gamma, py::arg("option"), py::arg("market"), py::arg("vol"));
  m.def("black_scholes_vega", &black_scholes_vega, py::arg("option"), py::arg("market"), py::arg("vol"));
  m.def("black_scholes_theta", &black_scholes_theta, py::arg("option"), py::arg("market"), py::arg("vol"));
  m.def("black_scholes_rho", &black_scholes_rho, py::arg("option"), py::arg("market"), py::arg("vol"));
  m.def("implied_vol", [](const VanillaOption& opt, const Market& mkt, double price) {
    return implied_vol_black_scholes(opt, mkt, price);
  }, py::arg("option"), py::arg("market"), py::arg("price"),
     "Black–Scholes implied volatility from a price.");

  // ---- Finite-difference Greeks --------------------------------------------
  py::class_<Greeks>(m, "Greeks")
      .def_readonly("price", &Greeks::price)
      .def_readonly("delta", &Greeks::delta)
      .def_readonly("gamma", &Greeks::gamma)
      .def_readonly("theta", &Greeks::theta)
      .def_readonly("rho", &Greeks::rho);

  py::class_<HestonParamSensitivities>(m, "HestonParamSensitivities")
      .def_readonly("d_v0", &HestonParamSensitivities::d_v0)
      .def_readonly("d_kappa", &HestonParamSensitivities::d_kappa)
      .def_readonly("d_theta", &HestonParamSensitivities::d_theta)
      .def_readonly("d_sigma", &HestonParamSensitivities::d_sigma)
      .def_readonly("d_rho", &HestonParamSensitivities::d_rho);

  m.def("compute_greeks_fd",
        [](const HestonFourierPricer& pricer, const VanillaOption& opt,
           const Market& mkt, const HestonParams& p) {
          return compute_greeks_fd(pricer, opt, mkt, p);
        },
        py::arg("pricer"), py::arg("option"), py::arg("market"), py::arg("params"));

  m.def("compute_param_sensitivities_fd",
        [](const HestonFourierPricer& pricer, const VanillaOption& opt,
           const Market& mkt, const HestonParams& p) {
          return compute_param_sensitivities_fd(pricer, opt, mkt, p);
        },
        py::arg("pricer"), py::arg("option"), py::arg("market"), py::arg("params"));

  // ---- Calibration (prices, CMA-ES) ----------------------------------------
  py::class_<PriceQuote>(m, "PriceQuote")
      .def(py::init([](const VanillaOption& opt, double market_price, double weight) {
             return PriceQuote{opt, market_price, weight};
           }),
           py::arg("option"), py::arg("market_price"), py::arg("weight") = 1.0)
      .def_readwrite("opt", &PriceQuote::opt)
      .def_readwrite("market_price", &PriceQuote::market_price)
      .def_readwrite("weight", &PriceQuote::weight);

  py::class_<CalibrationReport>(m, "CalibrationReport")
      .def_readonly("params", &CalibrationReport::params)
      .def_readonly("loss", &CalibrationReport::loss)
      .def_readonly("iters", &CalibrationReport::iters)
      .def_readonly("final_residuals", &CalibrationReport::final_residuals);

  m.def("calibrate_heston_to_prices_cmaes",
        [](const std::vector<PriceQuote>& quotes, const Market& mkt,
           const HestonParams& guess) {
          return calibrate_heston_to_prices_cmaes(quotes, mkt, guess);
        },
        py::arg("quotes"), py::arg("market"), py::arg("initial_guess"),
        "Calibrate the five Heston parameters to price quotes via CMA-ES.");

  // ---- GPR surrogate pricer -------------------------------------------------
  py::class_<GprPricer, IVanillaPricer>(m, "GprPricer")
      .def(py::init([](const Market& reference, std::pair<double, double> K,
                       std::pair<double, double> T, std::pair<double, double> v0,
                       std::pair<double, double> kappa, std::pair<double, double> theta,
                       std::pair<double, double> sigma, std::pair<double, double> rho,
                       std::size_t n_train, std::uint64_t seed, bool optimize_hyperparameters,
                       std::size_t max_opt_iter) {
             GprPricer::Config cfg;
             cfg.reference = reference;
             cfg.box = GprPricer::Box{K.first,     K.second,     T.first,     T.second,
                                      v0.first,    v0.second,    kappa.first, kappa.second,
                                      theta.first, theta.second, sigma.first, sigma.second,
                                      rho.first,   rho.second};
             cfg.n_train = n_train;
             cfg.sample_seed = seed;
             cfg.gp.optimize_hyperparameters = optimize_hyperparameters;
             cfg.gp.max_opt_iter = max_opt_iter;
             return GprPricer(cfg);
           }),
           py::arg("reference"), py::arg("K"), py::arg("T"), py::arg("v0"), py::arg("kappa"),
           py::arg("theta"), py::arg("sigma"), py::arg("rho"), py::arg("n_train") = 1500,
           py::arg("seed") = 20240607, py::arg("optimize_hyperparameters") = true,
           py::arg("max_opt_iter") = 200,
           "GPR surrogate for the Heston price over a box in (K, T, v0, kappa, theta, sigma, rho). "
           "Each range is a (low, high) tuple; the market (reference) is held fixed.")
      .def("train", &GprPricer::train, py::arg("truth"),
           "Train the surrogate by pricing sampled box points with a reference engine "
           "(e.g. a HestonFourierPricer).")
      .def("price", &GprPricer::price, py::arg("option"), py::arg("market"), py::arg("params"))
      .def("diagnostics", &GprPricer::diagnostics,
           "Predictive std (std_error) and any extrapolation note for the most recent price().")
      .def("is_trained", &GprPricer::is_trained);
}
