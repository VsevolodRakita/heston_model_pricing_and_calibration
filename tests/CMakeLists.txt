add_executable(test_heston_params
  test_main.cpp
  test_heston_params.cpp
)

target_include_directories(test_heston_params PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_heston_params PRIVATE heston_core)
add_test(NAME test_heston_params COMMAND test_heston_params)


add_executable(test_characteristic_function
  test_main.cpp
  test_characteristic_function.cpp
)

target_include_directories(test_characteristic_function PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_characteristic_function PRIVATE heston_core)
add_test(NAME test_characteristic_function COMMAND test_characteristic_function)

add_executable(test_fourier_pricer
  test_main.cpp
  test_fourier_pricer.cpp
)

target_include_directories(test_fourier_pricer PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_fourier_pricer PRIVATE heston_core)

add_test(NAME test_fourier_pricer COMMAND test_fourier_pricer)

add_executable(test_monte_carlo_pricer
  test_main.cpp
  test_monte_carlo_pricer.cpp
)

target_include_directories(test_monte_carlo_pricer PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_monte_carlo_pricer PRIVATE heston_core)

add_test(NAME test_monte_carlo_pricer COMMAND test_monte_carlo_pricer)

add_executable(test_monte_carlo_pricer_slow
    test_main.cpp
    test_monte_carlo_pricer_slow.cpp
)
target_link_libraries(test_monte_carlo_pricer_slow PRIVATE heston_core)

add_test(
    NAME test_monte_carlo_pricer_slow
    COMMAND test_monte_carlo_pricer_slow
)

set_tests_properties(
    test_monte_carlo_pricer_slow
    PROPERTIES
        LABELS "monte_carlo;slow"
)

add_executable(test_limit_cases_and_stability
    test_main.cpp
    test_limit_cases_and_stability.cpp
)
target_link_libraries(test_limit_cases_and_stability PRIVATE heston_core)

add_test(NAME test_limit_cases_and_stability COMMAND test_limit_cases_and_stability)
set_tests_properties(test_limit_cases_and_stability PROPERTIES LABELS "fast;stability")

add_executable(test_implied_vol
  test_main.cpp
  test_implied_vol.cpp
)

target_include_directories(test_implied_vol PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_implied_vol PRIVATE heston_core)
add_test(NAME test_implied_vol COMMAND test_implied_vol)
set_tests_properties(test_implied_vol PROPERTIES LABELS "fast;vol")

add_executable(test_calibration_synthetic
  test_main.cpp
  test_calibration_synthetic.cpp
)

target_include_directories(test_calibration_synthetic PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_calibration_synthetic PRIVATE heston_core)

add_test(NAME test_calibration_synthetic COMMAND test_calibration_synthetic)
set_tests_properties(test_calibration_synthetic PROPERTIES LABELS "slow;calibration")

add_executable(test_calibration_extended
  test_main.cpp
  test_calibration_extended.cpp
)

target_include_directories(test_calibration_extended PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_calibration_extended PRIVATE heston_core)

add_test(NAME test_calibration_extended COMMAND test_calibration_extended)
set_tests_properties(test_calibration_extended PROPERTIES LABELS "slow;calibration")

add_executable(test_calibration_price_cmaes
  test_main.cpp
  test_calibration_price_cmaes.cpp
)

target_include_directories(test_calibration_price_cmaes PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_calibration_price_cmaes PRIVATE heston_core)

add_test(NAME test_calibration_price_cmaes COMMAND test_calibration_price_cmaes)
set_tests_properties(test_calibration_price_cmaes PROPERTIES LABELS "slow;calibration")