add_library(heston_core)

target_sources(heston_core
  PRIVATE
    models/heston/heston_params.cpp
    models/heston/characteristic_function.cpp
    pricers/fft/heston_fourier_pricer.cpp
    pricers/mc/heston_monte_carlo_pricer.cpp
    vol/black_scholes.cpp
    vol/implied_vol.cpp
    calibration/objective.cpp
    calibration/calibrator.cpp
    optimization/nelder_mead.cpp
)

target_include_directories(heston_core
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Good warnings (GCC/Clang/MSVC)
if(MSVC)
  target_compile_options(heston_core PRIVATE /W4)
else()
  target_compile_options(heston_core PRIVATE -Wall -Wextra -Wpedantic)
endif()