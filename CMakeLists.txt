cmake_minimum_required(VERSION 3.18)

project(tensorstudio VERSION 2.1.0 LANGUAGES CXX)

option(
  TENSORSTUDIO_ENABLE_CUDA_BACKEND_DESCRIPTOR
  "Mark CUDA backend descriptor hooks as compiled without enabling CUDA tensor execution"
  OFF)
option(
  TENSORSTUDIO_ENABLE_METAL_BACKEND_DESCRIPTOR
  "Mark Metal backend descriptor hooks as compiled without enabling Metal tensor execution"
  OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(TENSORSTUDIO_CORE_SOURCES
  src/core/tensor.cpp
  src/core/dtype.cpp
  src/core/shape.cpp
  src/core/device.cpp
  src/core/storage.cpp
  src/core/ops.cpp
  src/core/autograd.cpp
  src/core/module.cpp
  src/core/optim.cpp
  src/core/random.cpp
  src/core/serialization.cpp
  src/core/errors.cpp
)

set(TENSORSTUDIO_BINDING_SOURCES
  src/bindings/bindings.cpp
  src/bindings/bind_tensor.cpp
  src/bindings/bind_ops.cpp
  src/bindings/bind_autograd.cpp
  src/bindings/bind_nn.cpp
  src/bindings/bind_optim.cpp
)

pybind11_add_module(_C MODULE
  ${TENSORSTUDIO_CORE_SOURCES}
  ${TENSORSTUDIO_BINDING_SOURCES}
)

target_include_directories(_C PRIVATE include src/bindings)
target_compile_features(_C PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(_C PRIVATE /W4 /permissive-)
else()
  target_compile_options(_C PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(_C PRIVATE TENSORSTUDIO_VERSION="${PROJECT_VERSION}")
if(TENSORSTUDIO_ENABLE_CUDA_BACKEND_DESCRIPTOR)
  target_compile_definitions(_C PRIVATE TENSORSTUDIO_HAS_CUDA=1)
endif()
if(TENSORSTUDIO_ENABLE_METAL_BACKEND_DESCRIPTOR)
  target_compile_definitions(_C PRIVATE TENSORSTUDIO_HAS_METAL=1)
endif()

install(TARGETS _C
  LIBRARY DESTINATION tensorstudio
  RUNTIME DESTINATION tensorstudio
  ARCHIVE DESTINATION tensorstudio
)
