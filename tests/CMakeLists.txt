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