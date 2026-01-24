#pragma once
#include <cstddef>

#include "models/heston/market.hpp"
#include "products/vanilla_option.hpp"

namespace heston {

struct ImpliedVolSettings {
  double vol_lower = 1e-8;
  double vol_upper = 5.0;
  double tol_abs = 1e-8;
  std::size_t max_iter = 200;

  bool newton_polish = true;
  std::size_t newton_max_iter = 10;
};

double implied_vol_black_scholes(
    const VanillaOption& opt,
    const Market& mkt,
    double price,
    const ImpliedVolSettings& settings = ImpliedVolSettings{});

} // namespace heston