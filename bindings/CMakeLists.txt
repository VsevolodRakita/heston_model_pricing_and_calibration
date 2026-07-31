find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(heston_py heston_py.cpp)

# Import name is `heston` (the compiled file is heston<PYTHON_EXT_SUFFIX>).
set_target_properties(heston_py PROPERTIES OUTPUT_NAME heston)

target_link_libraries(heston_py PRIVATE heston_core)

# On MinGW, link the GCC/libstdc++ runtime statically into the module. This both
# avoids a binutils import-table crash seen on some ld versions and makes the
# resulting .pyd self-contained (no libstdc++-6 / libgcc_s DLL needed at import).
if(MINGW)
  target_link_options(heston_py PRIVATE -static-libgcc -static-libstdc++)
endif()
