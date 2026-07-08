cmake_minimum_required(VERSION 3.18)

project(tensorstudio VERSION 1.15.0 LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

option(TENSORSTUDIO_ENABLE_BLAS "Enable optional CBLAS-backed matmul when available" ON)
option(TENSORSTUDIO_ENABLE_CUDA "Enable experimental CUDA backend metadata hooks" OFF)
option(TENSORSTUDIO_ENABLE_METAL "Enable experimental Metal backend metadata hooks on Apple platforms" OFF)

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
  src/core/perf.cpp
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

if(TENSORSTUDIO_ENABLE_BLAS)
  find_package(BLAS)
  if(BLAS_FOUND)
    include(CheckIncludeFileCXX)
    check_include_file_cxx(cblas.h TENSORSTUDIO_HAVE_CBLAS_H)
    target_link_libraries(_C PRIVATE ${BLAS_LIBRARIES})
    if(APPLE)
      target_compile_definitions(_C PRIVATE TENSORSTUDIO_HAS_ACCELERATE=1)
    elseif(TENSORSTUDIO_HAVE_CBLAS_H)
      target_compile_definitions(_C PRIVATE TENSORSTUDIO_HAS_CBLAS=1)
    endif()
  endif()
endif()

if(TENSORSTUDIO_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    target_compile_definitions(_C PRIVATE TENSORSTUDIO_HAS_CUDA=1)
  else()
    message(WARNING "TENSORSTUDIO_ENABLE_CUDA was requested, but CUDAToolkit was not found")
  endif()
endif()

if(TENSORSTUDIO_ENABLE_METAL)
  if(APPLE)
    target_compile_definitions(_C PRIVATE TENSORSTUDIO_HAS_METAL=1)
  else()
    message(WARNING "TENSORSTUDIO_ENABLE_METAL is only supported on Apple platforms")
  endif()
endif()

if(MSVC)
  target_compile_options(_C PRIVATE /W4 /permissive-)
else()
  target_compile_options(_C PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(_C PRIVATE TENSORSTUDIO_VERSION="${PROJECT_VERSION}")

install(TARGETS _C
  LIBRARY DESTINATION tensorstudio
  RUNTIME DESTINATION tensorstudio
  ARCHIVE DESTINATION tensorstudio
)
