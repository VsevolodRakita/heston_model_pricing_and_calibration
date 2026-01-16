#pragma once
#include <stdexcept>
#include <string>

namespace heston {

struct HestonError : std::runtime_error {
  explicit HestonError(const std::string& msg) : std::runtime_error(msg) {}
};

struct InvalidInput : HestonError {
  explicit InvalidInput(const std::string& msg) : HestonError("InvalidInput: " + msg) {}
};

struct NumericFailure : HestonError {
  explicit NumericFailure(const std::string& msg) : HestonError("NumericFailure: " + msg) {}
};

} // namespace heston