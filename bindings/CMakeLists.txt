find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(heston_py heston_py.cpp)

# Import name is `heston` (the compiled file is heston<PYTHON_EXT_SUFFIX>).
set_target_properties(heston_py PROPERTIES OUTPUT_NAME heston)

target_link_libraries(heston_py PRIVATE heston_core)
