#pragma once

#include <stddef.h>
#include <stdint.h>

#define TENSORSTUDIO_KERNEL_ABI_VERSION_MAJOR 1
#define TENSORSTUDIO_KERNEL_ABI_VERSION_MINOR 0
#define TENSORSTUDIO_KERNEL_ABI_NAME "tensorstudio.kernel_abi.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum TensorStudioKernelDType {
  TENSORSTUDIO_KERNEL_BOOL = 0,
  TENSORSTUDIO_KERNEL_INT32 = 1,
  TENSORSTUDIO_KERNEL_INT64 = 2,
  TENSORSTUDIO_KERNEL_FLOAT32 = 3,
  TENSORSTUDIO_KERNEL_FLOAT64 = 4,
} TensorStudioKernelDType;

typedef enum TensorStudioKernelDevice {
  TENSORSTUDIO_KERNEL_CPU = 0,
  TENSORSTUDIO_KERNEL_CUDA = 1,
  TENSORSTUDIO_KERNEL_METAL = 2,
  TENSORSTUDIO_KERNEL_PLUGIN = 3,
} TensorStudioKernelDevice;

typedef enum TensorStudioKernelStatusCode {
  TENSORSTUDIO_KERNEL_OK = 0,
  TENSORSTUDIO_KERNEL_INVALID_ARGUMENT = 1,
  TENSORSTUDIO_KERNEL_UNSUPPORTED_DTYPE = 2,
  TENSORSTUDIO_KERNEL_UNSUPPORTED_DEVICE = 3,
  TENSORSTUDIO_KERNEL_RUNTIME_ERROR = 4,
} TensorStudioKernelStatusCode;

typedef struct TensorStudioKernelTensorView {
  void* data;
  const int64_t* shape;
  const int64_t* strides;
  int64_t ndim;
  int64_t offset;
  TensorStudioKernelDType dtype;
  TensorStudioKernelDevice device;
  const char* backend;
} TensorStudioKernelTensorView;

typedef struct TensorStudioKernelStatus {
  TensorStudioKernelStatusCode code;
  const char* message;
} TensorStudioKernelStatus;

typedef struct TensorStudioKernelContext {
  uint32_t abi_major;
  uint32_t abi_minor;
  void* user_data;
} TensorStudioKernelContext;

typedef TensorStudioKernelStatus (*TensorStudioKernelFn)(
    TensorStudioKernelContext* context,
    const TensorStudioKernelTensorView* inputs,
    size_t input_count,
    TensorStudioKernelTensorView* outputs,
    size_t output_count);

typedef struct TensorStudioKernelRegistration {
  const char* name;
  const char* backend;
  const char* abi_name;
  TensorStudioKernelFn function;
} TensorStudioKernelRegistration;

#ifdef __cplusplus
}
#endif
