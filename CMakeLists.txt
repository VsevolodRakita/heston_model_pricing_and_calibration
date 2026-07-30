cmake_minimum_required(VERSION 3.20)

project(heston_pricing
  VERSION 0.1.0
  LANGUAGES CXX
)

# Options
option(HESTON_BUILD_TESTS "Build unit tests" ON)
option(HESTON_BUILD_PYTHON "Build the pybind11 Python module" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Allow the static core library to be linked into the Python shared module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Helpful defaults
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Core library (minimal, grows over time)
add_subdirectory(src)

# Tests
if(HESTON_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Python bindings (pybind11)
if(HESTON_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()