#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tensorstudio/dtype.hpp"

namespace tensorstudio {

enum class DeviceType {
  CPU,
  CUDA,
  Metal,
  Plugin,
};

enum class MemorySpace {
  Host,
  Device,
  Unified,
};

enum class KernelExecutionMode {
  Synchronous,
  Asynchronous,
};

struct BackendInfo {
  DeviceType type{DeviceType::CPU};
  std::string name{};
  bool compiled{false};
  bool available{false};
  int64_t device_count{0};
  std::string reason{};
};

class Device {
 public:
  Device();
  explicit Device(DeviceType type, int64_t index = 0, std::string backend = {});

  DeviceType type() const;
  const std::string& backend() const;
  int64_t index() const;
  std::string str() const;
  bool is_cpu() const;
  bool is_accelerator() const;

  bool operator==(const Device& other) const;
  bool operator!=(const Device& other) const;

 private:
  DeviceType type_;
  int64_t index_;
  std::string backend_;
};

struct DeviceTransferInfo {
  Device source{};
  Device target{};
  bool supported{false};
  bool direct{false};
  std::string reason{};
};

struct BackendAllocatorInfo {
  Device device{};
  MemorySpace memory_space{MemorySpace::Host};
  bool can_allocate{false};
  bool supports_pinned_host{false};
  bool supports_unified_memory{false};
  bool supports_streams{false};
  int64_t alignment{64};
  std::string reason{};
};

struct BackendRuntimeInfo {
  Device device{};
  bool executable{false};
  bool supports_eager{false};
  bool supports_graph{false};
  bool supports_compiler{false};
  bool supports_streams{false};
  bool supports_events{false};
  bool supports_peer_access{false};
  bool supports_host_fallback{false};
  std::string compiler{};
  std::string reason{};
};

struct BackendDeviceProperties {
  Device device{};
  std::string name{};
  std::string vendor{};
  std::string architecture{};
  int64_t total_memory_bytes{0};
  int64_t free_memory_bytes{0};
  int64_t max_threads_per_block{0};
  int64_t max_shared_memory_bytes{0};
  bool supports_fp16{false};
  bool supports_bf16{false};
  bool supports_tf32{false};
  bool unified_addressing{false};
  bool available{false};
  std::string reason{};
};

struct LogicalDeviceInfo {
  Device physical_device{};
  Device logical_device{};
  int64_t memory_limit_bytes{0};
  int64_t priority{0};
  bool default_device{false};
  std::string reason{};
};

struct BackendOpInfo {
  std::string op{};
  int64_t min_inputs{1};
  int64_t max_inputs{1};
  bool differentiable{false};
  bool shape_inference{true};
  bool stateful{false};
  std::string category{};
  std::string notes{};
};

struct BackendKernelInfo {
  DeviceType type{DeviceType::CPU};
  std::string backend{};
  std::string op{};
  std::vector<DType> dtypes{};
  MemorySpace memory_space{MemorySpace::Host};
  KernelExecutionMode execution_mode{KernelExecutionMode::Synchronous};
  bool forward{true};
  bool backward{false};
  bool available{false};
  bool deterministic{true};
  int64_t priority{100};
  std::string reason{};
};

struct KernelPlacementInfo {
  std::string op{};
  DType dtype{DType::Float32};
  Device requested{};
  Device selected{};
  bool supported{false};
  bool fallback_to_cpu{false};
  bool requires_transfer{false};
  std::string reason{};
};

struct BackendExecutionPlan {
  std::string op{};
  DType dtype{DType::Float32};
  Device requested{};
  Device selected{};
  std::vector<Device> input_devices{};
  std::vector<DeviceTransferInfo> input_transfers{};
  bool executable{false};
  bool uses_fallback{false};
  bool requires_transfer{false};
  bool stream_required{false};
  bool graph_compatible{false};
  std::string reason{};
};

Device cpu_device();
Device parse_device(const std::string& spec);
std::string device_type_name(DeviceType type);
std::string memory_space_name(MemorySpace space);
MemorySpace memory_space_from_string(const std::string& name);
std::string kernel_execution_mode_name(KernelExecutionMode mode);
KernelExecutionMode kernel_execution_mode_from_string(const std::string& name);
bool is_device_available(const Device& device);
int64_t device_count(DeviceType type);
int64_t device_count(const Device& device);
std::vector<Device> available_devices();
std::vector<BackendInfo> backend_info();
std::vector<BackendAllocatorInfo> backend_allocator_info();
BackendAllocatorInfo backend_allocator_info(const Device& device);
std::vector<BackendRuntimeInfo> backend_runtime_info();
BackendRuntimeInfo backend_runtime_info(const Device& device);
std::vector<BackendDeviceProperties> backend_device_properties();
BackendDeviceProperties backend_device_properties(const Device& device);
std::vector<LogicalDeviceInfo> logical_device_info();
std::vector<LogicalDeviceInfo> logical_device_info(const Device& device);
DeviceTransferInfo device_transfer_info(const Device& source, const Device& target);
bool can_transfer(const Device& source, const Device& target);
std::vector<BackendOpInfo> backend_op_info();
BackendOpInfo backend_op_info(const std::string& op);
std::vector<BackendKernelInfo> backend_kernel_info();
std::vector<BackendKernelInfo> backend_kernel_info(const Device& device);
bool backend_supports_kernel(const Device& device, const std::string& op, DType dtype);
KernelPlacementInfo kernel_placement_info(const Device& requested, const std::string& op, DType dtype);
BackendExecutionPlan backend_execution_plan(
    const Device& requested,
    const std::string& op,
    DType dtype,
    const std::vector<Device>& input_devices = {});
void register_backend(BackendInfo info);
void unregister_backend(const std::string& name);
void register_backend_kernel(BackendKernelInfo info);

}  // namespace tensorstudio
