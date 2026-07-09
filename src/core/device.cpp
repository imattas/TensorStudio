#include "tensorstudio/device.hpp"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "tensorstudio/errors.hpp"

namespace tensorstudio {
namespace {

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

std::mutex& registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<BackendInfo>& plugin_backends() {
  static std::vector<BackendInfo> backends;
  return backends;
}

std::vector<BackendKernelInfo>& plugin_kernels() {
  static std::vector<BackendKernelInfo> kernels;
  return kernels;
}

bool is_builtin_backend_name(const std::string& name) {
  return name == "cpu" || name == "cuda" || name == "gpu" || name == "metal" || name == "mps";
}

void validate_plugin_name(const std::string& name) {
  if (name.empty()) {
    throw DeviceError("backend name cannot be empty");
  }
  if (is_builtin_backend_name(name)) {
    throw DeviceError("backend name '" + name + "' is reserved for a built-in TensorStudio backend");
  }
  for (const char ch : name) {
    const bool valid =
        (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.';
    if (!valid) {
      throw DeviceError(
          "backend name '" + name +
          "' must contain only lowercase letters, digits, underscores, dashes, or dots");
    }
  }
}

void validate_op_name(const std::string& name) {
  if (name.empty()) {
    throw DeviceError("backend op name cannot be empty");
  }
  for (const char ch : name) {
    const bool valid =
        (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.';
    if (!valid) {
      throw DeviceError(
          "backend op name '" + name +
          "' must contain only lowercase letters, digits, underscores, or dots");
    }
  }
}

auto find_plugin_backend_unlocked(const std::string& name) {
  auto& backends = plugin_backends();
  return std::find_if(backends.begin(), backends.end(), [&](const BackendInfo& info) {
    return info.name == name;
  });
}

auto find_plugin_backend_unlocked_const(const std::string& name) {
  const auto& backends = plugin_backends();
  return std::find_if(backends.begin(), backends.end(), [&](const BackendInfo& info) {
    return info.name == name;
  });
}

bool plugin_backend_registered(const std::string& name) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  return find_plugin_backend_unlocked_const(name) != plugin_backends().end();
}

DeviceType parse_device_type(const std::string& name) {
  if (name == "cpu") {
    return DeviceType::CPU;
  }
  if (name == "cuda" || name == "gpu") {
    return DeviceType::CUDA;
  }
  if (name == "metal" || name == "mps") {
    return DeviceType::Metal;
  }
  if (plugin_backend_registered(name)) {
    return DeviceType::Plugin;
  }
  throw DeviceError("unsupported device type '" + name + "'; expected cpu, cuda, metal, or a registered plugin backend");
}

bool cuda_compiled() {
#if defined(TENSORSTUDIO_HAS_CUDA)
  return true;
#else
  return false;
#endif
}

bool metal_compiled() {
#if defined(TENSORSTUDIO_HAS_METAL)
  return true;
#else
  return false;
#endif
}

BackendInfo builtin_backend_info(DeviceType type) {
  BackendInfo info;
  info.type = type;
  info.name = device_type_name(type);
  if (type == DeviceType::CPU) {
    info.compiled = true;
    info.available = true;
    info.device_count = 1;
    info.reason = "portable CPU backend is always available";
    return info;
  }
  if (type == DeviceType::CUDA) {
    info.compiled = cuda_compiled();
    info.available = false;
    info.device_count = 0;
    info.reason = info.compiled
        ? "CUDA backend hooks were compiled, but runtime kernels and storage are not enabled"
        : "CUDA backend was not compiled into this wheel";
    return info;
  }
  if (type == DeviceType::Metal) {
    info.compiled = metal_compiled();
    info.available = false;
    info.device_count = 0;
    info.reason = info.compiled
        ? "Metal backend hooks were compiled, but runtime kernels and storage are not enabled"
        : "Metal backend was not compiled into this wheel";
    return info;
  }
  throw DeviceError("plugin backend info must be looked up by backend name");
}

BackendInfo info_for_device(const Device& device) {
  if (device.type() != DeviceType::Plugin) {
    return builtin_backend_info(device.type());
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  const auto it = find_plugin_backend_unlocked_const(device.backend());
  if (it == plugin_backends().end()) {
    BackendInfo missing;
    missing.type = DeviceType::Plugin;
    missing.name = device.backend();
    missing.reason = "plugin backend '" + device.backend() + "' is not registered";
    return missing;
  }
  return *it;
}

std::vector<DType> all_dtypes() {
  return {DType::Float32, DType::Float64, DType::Int32, DType::Int64, DType::Bool};
}

std::vector<DType> floating_dtypes() {
  return {DType::Float32, DType::Float64};
}

BackendKernelInfo cpu_kernel(std::string op, std::vector<DType> dtypes, bool backward) {
  BackendKernelInfo info;
  info.type = DeviceType::CPU;
  info.backend = "cpu";
  info.op = std::move(op);
  info.dtypes = std::move(dtypes);
  info.memory_space = MemorySpace::Host;
  info.execution_mode = KernelExecutionMode::Synchronous;
  info.forward = true;
  info.backward = backward;
  info.available = true;
  info.deterministic = true;
  info.priority = 100;
  info.reason = "registered native CPU kernel";
  return info;
}

BackendOpInfo op_info(
    std::string op,
    int64_t min_inputs,
    int64_t max_inputs,
    bool differentiable,
    std::string category,
    std::string notes = {},
    bool stateful = false) {
  BackendOpInfo info;
  info.op = std::move(op);
  info.min_inputs = min_inputs;
  info.max_inputs = max_inputs;
  info.differentiable = differentiable;
  info.category = std::move(category);
  info.notes = std::move(notes);
  info.stateful = stateful;
  return info;
}

const std::vector<BackendOpInfo>& op_registry() {
  static const std::vector<BackendOpInfo> registry{
      op_info("add", 2, 2, true, "elementwise"),
      op_info("sub", 2, 2, true, "elementwise"),
      op_info("mul", 2, 2, true, "elementwise"),
      op_info("div", 2, 2, true, "elementwise"),
      op_info("neg", 1, 1, true, "elementwise"),
      op_info("pow", 1, 1, true, "elementwise", "scalar exponent"),
      op_info("equal", 2, 2, false, "comparison"),
      op_info("not_equal", 2, 2, false, "comparison"),
      op_info("less", 2, 2, false, "comparison"),
      op_info("less_equal", 2, 2, false, "comparison"),
      op_info("greater", 2, 2, false, "comparison"),
      op_info("greater_equal", 2, 2, false, "comparison"),
      op_info("maximum", 2, 2, false, "elementwise"),
      op_info("minimum", 2, 2, false, "elementwise"),
      op_info("where", 3, 3, false, "selection"),
      op_info("matmul", 2, 2, true, "linear_algebra", "2D matrix multiplication"),
      op_info("sum", 1, 1, true, "reduction"),
      op_info("mean", 1, 1, true, "reduction"),
      op_info("max", 1, 1, false, "reduction"),
      op_info("min", 1, 1, false, "reduction"),
      op_info("argmax", 1, 1, false, "reduction"),
      op_info("argmin", 1, 1, false, "reduction"),
      op_info("reshape", 1, 1, true, "view"),
      op_info("flatten", 1, 1, true, "view"),
      op_info("transpose", 1, 1, true, "view"),
      op_info("relu", 1, 1, true, "activation"),
      op_info("sigmoid", 1, 1, true, "activation"),
      op_info("tanh", 1, 1, true, "activation"),
      op_info("exp", 1, 1, true, "math"),
      op_info("log", 1, 1, true, "math"),
      op_info("log1p", 1, 1, true, "math"),
      op_info("sqrt", 1, 1, true, "math"),
      op_info("rsqrt", 1, 1, true, "math"),
      op_info("sin", 1, 1, true, "math"),
      op_info("cos", 1, 1, true, "math"),
      op_info("tan", 1, 1, true, "math"),
      op_info("asin", 1, 1, true, "math"),
      op_info("acos", 1, 1, true, "math"),
      op_info("atan", 1, 1, true, "math"),
      op_info("abs", 1, 1, true, "math"),
      op_info("clamp", 1, 1, true, "math"),
      op_info("concat", 1, -1, false, "shape", "variadic input list"),
      op_info("stack", 1, -1, false, "shape", "variadic input list"),
      op_info("astype", 1, 1, false, "dtype"),
      op_info("conv2d", 2, 3, true, "vision", "NCHW convolution"),
      op_info("max_pool2d", 1, 1, true, "vision"),
      op_info("avg_pool2d", 1, 1, true, "vision"),
  };
  return registry;
}

std::vector<BackendKernelInfo> cpu_kernel_registry() {
  const std::vector<DType> all = all_dtypes();
  const std::vector<DType> floating = floating_dtypes();
  std::vector<BackendKernelInfo> kernels;
  auto add = [&](const std::string& op, const std::vector<DType>& dtypes, bool backward) {
    kernels.push_back(cpu_kernel(op, dtypes, backward));
  };

  for (const char* op : {"add", "sub", "mul", "div", "neg", "pow", "sum", "mean", "reshape", "flatten", "transpose"}) {
    add(op, all, true);
  }
  for (const char* op : {"matmul", "relu", "sigmoid", "tanh", "exp", "log", "sqrt", "rsqrt", "conv2d", "max_pool2d", "avg_pool2d"}) {
    add(op, floating, true);
  }
  for (const char* op : {
           "equal",
           "not_equal",
           "less",
           "less_equal",
           "greater",
           "greater_equal",
           "maximum",
           "minimum",
           "max",
           "min",
           "argmax",
           "argmin",
       }) {
    add(op, all, false);
  }
  return kernels;
}

bool dtype_in_list(DType dtype, const std::vector<DType>& dtypes) {
  return std::find(dtypes.begin(), dtypes.end(), dtype) != dtypes.end();
}

}  // namespace

Device::Device() : type_(DeviceType::CPU), index_(0), backend_("cpu") {}

Device::Device(DeviceType type, int64_t index, std::string backend) : type_(type), index_(index) {
  if (index < 0) {
    throw DeviceError("device index must be non-negative");
  }
  if (type == DeviceType::CPU && index != 0) {
    throw DeviceError("cpu device only supports index 0");
  }
  if (type == DeviceType::Plugin) {
    backend_ = lowercase(std::move(backend));
    validate_plugin_name(backend_);
  } else {
    backend_ = device_type_name(type);
  }
}

DeviceType Device::type() const {
  return type_;
}

const std::string& Device::backend() const {
  return backend_;
}

int64_t Device::index() const {
  return index_;
}

std::string Device::str() const {
  if (type_ == DeviceType::CPU) {
    return "cpu";
  }
  return backend_ + ":" + std::to_string(index_);
}

bool Device::is_cpu() const {
  return type_ == DeviceType::CPU;
}

bool Device::is_accelerator() const {
  return type_ == DeviceType::CUDA || type_ == DeviceType::Metal || type_ == DeviceType::Plugin;
}

bool Device::operator==(const Device& other) const {
  return type_ == other.type_ && index_ == other.index_ && backend_ == other.backend_;
}

bool Device::operator!=(const Device& other) const {
  return !(*this == other);
}

Device cpu_device() {
  return Device(DeviceType::CPU, 0);
}

Device parse_device(const std::string& spec) {
  const std::string normalized = lowercase(spec);
  const std::size_t separator = normalized.find(':');
  const std::string type_name = normalized.substr(0, separator);
  const DeviceType type = parse_device_type(type_name);
  int64_t index = 0;
  if (separator != std::string::npos) {
    const std::string index_text = normalized.substr(separator + 1);
    if (index_text.empty()) {
      throw DeviceError("device index is empty in '" + spec + "'");
    }
    try {
      std::size_t parsed = 0;
      index = std::stoll(index_text, &parsed);
      if (parsed != index_text.size()) {
        throw DeviceError("invalid device index in '" + spec + "'");
      }
    } catch (const std::invalid_argument&) {
      throw DeviceError("invalid device index in '" + spec + "'");
    } catch (const std::out_of_range&) {
      throw DeviceError("device index is out of range in '" + spec + "'");
    }
  }
  return Device(type, index, type == DeviceType::Plugin ? type_name : std::string{});
}

std::string device_type_name(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    case DeviceType::Metal:
      return "metal";
    case DeviceType::Plugin:
      return "plugin";
  }
  throw DeviceError("unknown device type");
}

std::string memory_space_name(MemorySpace space) {
  switch (space) {
    case MemorySpace::Host:
      return "host";
    case MemorySpace::Device:
      return "device";
    case MemorySpace::Unified:
      return "unified";
  }
  throw DeviceError("unknown memory space");
}

MemorySpace memory_space_from_string(const std::string& name) {
  const std::string normalized = lowercase(name);
  if (normalized == "host") {
    return MemorySpace::Host;
  }
  if (normalized == "device") {
    return MemorySpace::Device;
  }
  if (normalized == "unified") {
    return MemorySpace::Unified;
  }
  throw DeviceError("unsupported memory space '" + name + "'; expected host, device, or unified");
}

std::string kernel_execution_mode_name(KernelExecutionMode mode) {
  switch (mode) {
    case KernelExecutionMode::Synchronous:
      return "sync";
    case KernelExecutionMode::Asynchronous:
      return "async";
  }
  throw DeviceError("unknown kernel execution mode");
}

KernelExecutionMode kernel_execution_mode_from_string(const std::string& name) {
  const std::string normalized = lowercase(name);
  if (normalized == "sync" || normalized == "synchronous") {
    return KernelExecutionMode::Synchronous;
  }
  if (normalized == "async" || normalized == "asynchronous") {
    return KernelExecutionMode::Asynchronous;
  }
  throw DeviceError("unsupported kernel execution mode '" + name + "'; expected sync or async");
}

bool is_device_available(const Device& device) {
  return device.index() < device_count(device);
}

int64_t device_count(DeviceType type) {
  if (type == DeviceType::Plugin) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    int64_t total = 0;
    for (const BackendInfo& info : plugin_backends()) {
      total += info.device_count;
    }
    return total;
  }
  return builtin_backend_info(type).device_count;
}

int64_t device_count(const Device& device) {
  return info_for_device(device).device_count;
}

std::vector<Device> available_devices() {
  std::vector<Device> result;
  for (const BackendInfo& info : backend_info()) {
    for (int64_t index = 0; index < info.device_count; ++index) {
      result.emplace_back(info.type, index, info.type == DeviceType::Plugin ? info.name : std::string{});
    }
  }
  return result;
}

std::vector<BackendInfo> backend_info() {
  std::vector<BackendInfo> result{
      builtin_backend_info(DeviceType::CPU),
      builtin_backend_info(DeviceType::CUDA),
      builtin_backend_info(DeviceType::Metal),
  };
  std::lock_guard<std::mutex> lock(registry_mutex());
  result.insert(result.end(), plugin_backends().begin(), plugin_backends().end());
  return result;
}

BackendAllocatorInfo backend_allocator_info(const Device& device) {
  BackendAllocatorInfo info;
  info.device = device;
  info.alignment = 64;

  if (device.is_cpu()) {
    info.memory_space = MemorySpace::Host;
    info.can_allocate = true;
    info.supports_pinned_host = false;
    info.supports_unified_memory = false;
    info.supports_streams = false;
    info.reason = "CPU backend allocates owned host memory through TensorStudio Storage";
    return info;
  }

  const BackendInfo backend = info_for_device(device);
  info.memory_space = MemorySpace::Device;
  info.can_allocate = false;
  info.supports_pinned_host = false;
  info.supports_unified_memory = false;
  info.supports_streams = backend.compiled;
  if (device.type() == DeviceType::CUDA) {
    info.reason =
        "CUDA allocator interface is declared, but CUDA tensor storage and streams are not enabled";
    return info;
  }
  if (device.type() == DeviceType::Metal) {
    info.reason =
        "Metal allocator interface is declared, but Metal tensor storage and command queues are not enabled";
    return info;
  }
  info.reason =
      "plugin backend '" + device.backend() +
      "' is descriptor-only until audited native storage ownership is implemented";
  return info;
}

std::vector<BackendAllocatorInfo> backend_allocator_info() {
  std::vector<BackendAllocatorInfo> result;
  result.push_back(backend_allocator_info(Device(DeviceType::CPU, 0)));
  result.push_back(backend_allocator_info(Device(DeviceType::CUDA, 0)));
  result.push_back(backend_allocator_info(Device(DeviceType::Metal, 0)));
  std::vector<std::string> plugin_names;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (const BackendInfo& info : plugin_backends()) {
      plugin_names.push_back(info.name);
    }
  }
  for (const std::string& name : plugin_names) {
    result.push_back(backend_allocator_info(Device(DeviceType::Plugin, 0, name)));
  }
  return result;
}

BackendRuntimeInfo backend_runtime_info(const Device& device) {
  BackendRuntimeInfo info;
  info.device = device;
  info.compiler = "none";

  if (device.is_cpu()) {
    info.executable = true;
    info.supports_eager = true;
    info.supports_graph = false;
    info.supports_compiler = false;
    info.supports_streams = false;
    info.supports_events = false;
    info.supports_peer_access = false;
    info.supports_host_fallback = false;
    info.reason = "CPU eager runtime executes native TensorStudio kernels";
    return info;
  }

  const BackendInfo backend = info_for_device(device);
  info.supports_streams = backend.compiled;
  info.supports_events = backend.compiled;
  info.supports_host_fallback = true;
  if (device.type() == DeviceType::CUDA) {
    info.compiler = backend.compiled ? "future cuda/xla-style lowering boundary" : "none";
    info.reason = backend.compiled
        ? "CUDA runtime hooks are declared, but CUDA storage, streams, kernels, and compiler lowering are not enabled"
        : "CUDA runtime is not compiled into this wheel";
    return info;
  }
  if (device.type() == DeviceType::Metal) {
    info.compiler = backend.compiled ? "future metal/mps lowering boundary" : "none";
    info.reason = backend.compiled
        ? "Metal runtime hooks are declared, but Metal storage, command queues, kernels, and compiler lowering are not enabled"
        : "Metal runtime is not compiled into this wheel";
    return info;
  }

  info.compiler = "external plugin descriptor";
  info.reason =
      "plugin backend '" + device.backend() +
      "' is descriptor-only until audited native storage, streams, kernels, and synchronization are implemented";
  return info;
}

std::vector<BackendRuntimeInfo> backend_runtime_info() {
  std::vector<BackendRuntimeInfo> result;
  result.push_back(backend_runtime_info(Device(DeviceType::CPU, 0)));
  result.push_back(backend_runtime_info(Device(DeviceType::CUDA, 0)));
  result.push_back(backend_runtime_info(Device(DeviceType::Metal, 0)));
  std::vector<std::string> plugin_names;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (const BackendInfo& info : plugin_backends()) {
      plugin_names.push_back(info.name);
    }
  }
  for (const std::string& name : plugin_names) {
    result.push_back(backend_runtime_info(Device(DeviceType::Plugin, 0, name)));
  }
  return result;
}

BackendDeviceProperties backend_device_properties(const Device& device) {
  BackendDeviceProperties properties;
  properties.device = device;
  properties.available = is_device_available(device);

  if (device.is_cpu()) {
    properties.name = "TensorStudio host CPU";
    properties.vendor = "host";
    properties.architecture = "portable-cpu";
    properties.max_threads_per_block = static_cast<int64_t>(std::max(1U, std::thread::hardware_concurrency()));
    properties.max_shared_memory_bytes = 0;
    properties.supports_fp16 = false;
    properties.supports_bf16 = false;
    properties.supports_tf32 = false;
    properties.unified_addressing = true;
    properties.reason = "CPU properties are portable host-runtime metadata; total memory is not reserved by TensorStudio";
    return properties;
  }

  const BackendInfo backend = info_for_device(device);
  properties.name = backend.name + " device";
  properties.vendor = backend.name;
  properties.architecture = device_type_name(device.type());
  properties.reason = backend.reason;
  if (device.type() == DeviceType::CUDA) {
    properties.name = "CUDA device descriptor";
    properties.vendor = "nvidia";
    properties.architecture = "cuda";
    properties.unified_addressing = backend.compiled;
    return properties;
  }
  if (device.type() == DeviceType::Metal) {
    properties.name = "Metal/MPS device descriptor";
    properties.vendor = "apple";
    properties.architecture = "metal";
    properties.unified_addressing = backend.compiled;
    return properties;
  }
  properties.architecture = "external-plugin";
  properties.reason =
      "plugin backend '" + device.backend() +
      "' has descriptor metadata only; native device properties are not available";
  return properties;
}

std::vector<BackendDeviceProperties> backend_device_properties() {
  std::vector<BackendDeviceProperties> result;
  result.push_back(backend_device_properties(Device(DeviceType::CPU, 0)));
  result.push_back(backend_device_properties(Device(DeviceType::CUDA, 0)));
  result.push_back(backend_device_properties(Device(DeviceType::Metal, 0)));
  std::vector<std::string> plugin_names;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (const BackendInfo& info : plugin_backends()) {
      plugin_names.push_back(info.name);
    }
  }
  for (const std::string& name : plugin_names) {
    result.push_back(backend_device_properties(Device(DeviceType::Plugin, 0, name)));
  }
  return result;
}

std::vector<LogicalDeviceInfo> logical_device_info(const Device& device) {
  std::vector<LogicalDeviceInfo> result;
  if (!is_device_available(device)) {
    return result;
  }
  LogicalDeviceInfo info;
  info.physical_device = device;
  info.logical_device = device;
  info.memory_limit_bytes = 0;
  info.priority = 0;
  info.default_device = device.is_cpu();
  info.reason = device.is_cpu()
      ? "CPU exposes one default logical device backed by host storage"
      : "backend exposes one logical device for each available physical device";
  result.push_back(info);
  return result;
}

std::vector<LogicalDeviceInfo> logical_device_info() {
  std::vector<LogicalDeviceInfo> result;
  for (const Device& device : available_devices()) {
    std::vector<LogicalDeviceInfo> devices = logical_device_info(device);
    result.insert(result.end(), devices.begin(), devices.end());
  }
  return result;
}

DeviceTransferInfo device_transfer_info(const Device& source, const Device& target) {
  DeviceTransferInfo info;
  info.source = source;
  info.target = target;
  if (!is_device_available(source)) {
    info.reason = "source device '" + source.str() + "' is not available in this TensorStudio build";
    return info;
  }
  if (!is_device_available(target)) {
    info.reason = "target device '" + target.str() + "' is not available in this TensorStudio build";
    return info;
  }
  if (source.is_cpu() && target.is_cpu()) {
    info.supported = true;
    info.direct = true;
    info.reason = "CPU-to-CPU host copy is supported";
    return info;
  }
  info.reason =
      "transfer from '" + source.str() + "' to '" + target.str() +
      "' requires backend-specific storage and copy kernels that are not enabled in this build";
  return info;
}

bool can_transfer(const Device& source, const Device& target) {
  return device_transfer_info(source, target).supported;
}

std::vector<BackendOpInfo> backend_op_info() {
  return op_registry();
}

BackendOpInfo backend_op_info(const std::string& op) {
  validate_op_name(op);
  const auto& registry = op_registry();
  const auto it = std::find_if(registry.begin(), registry.end(), [&](const BackendOpInfo& info) {
    return info.op == op;
  });
  if (it == registry.end()) {
    throw DeviceError("operation '" + op + "' is not registered in the TensorStudio backend registry");
  }
  return *it;
}

std::vector<BackendKernelInfo> backend_kernel_info() {
  std::vector<BackendKernelInfo> kernels = cpu_kernel_registry();
  std::lock_guard<std::mutex> lock(registry_mutex());
  kernels.insert(kernels.end(), plugin_kernels().begin(), plugin_kernels().end());
  return kernels;
}

std::vector<BackendKernelInfo> backend_kernel_info(const Device& device) {
  std::vector<BackendKernelInfo> result;
  for (const BackendKernelInfo& info : backend_kernel_info()) {
    if (info.type != device.type()) {
      continue;
    }
    if (device.type() == DeviceType::Plugin && info.backend != device.backend()) {
      continue;
    }
    result.push_back(info);
  }
  return result;
}

bool backend_supports_kernel(const Device& device, const std::string& op, DType dtype) {
  if (!is_device_available(device)) {
    return false;
  }
  for (const BackendKernelInfo& info : backend_kernel_info(device)) {
    if (info.available && info.op == op && dtype_in_list(dtype, info.dtypes)) {
      return true;
    }
  }
  return false;
}

KernelPlacementInfo kernel_placement_info(const Device& requested, const std::string& op, DType dtype) {
  KernelPlacementInfo info;
  validate_op_name(op);
  info.op = op;
  info.dtype = dtype;
  info.requested = requested;
  info.selected = requested;
  const Device cpu = cpu_device();
  const bool cpu_supported = backend_supports_kernel(cpu, op, dtype);
  if (!is_device_available(requested)) {
    if (cpu_supported) {
      info.selected = cpu;
      info.fallback_to_cpu = !requested.is_cpu();
      info.requires_transfer = false;
      info.reason =
          "requested device '" + requested.str() +
          "' is not available; a native CPU kernel is available as an explicit fallback";
      return info;
    }
    info.reason =
        "requested device '" + requested.str() +
        "' is not available in this TensorStudio build and no CPU fallback kernel is registered";
    return info;
  }
  if (backend_supports_kernel(requested, op, dtype)) {
    info.supported = true;
    info.reason =
        "kernel '" + op + "' for dtype '" + dtype_name(dtype) + "' is registered on backend '" +
        requested.backend() + "'";
    return info;
  }
  if (!requested.is_cpu() && cpu_supported) {
    info.selected = cpu;
    info.fallback_to_cpu = true;
    info.requires_transfer = true;
    info.reason =
        "kernel '" + op + "' for dtype '" + dtype_name(dtype) + "' is not registered on backend '" +
        requested.backend() + "'; a CPU fallback kernel is registered";
    return info;
  }
  info.reason =
      "kernel '" + op + "' for dtype '" + dtype_name(dtype) + "' is not registered on backend '" +
      requested.backend() + "'";
  return info;
}

BackendExecutionPlan backend_execution_plan(
    const Device& requested,
    const std::string& op,
    DType dtype,
    const std::vector<Device>& input_devices) {
  BackendExecutionPlan plan;
  plan.op = op;
  plan.dtype = dtype;
  plan.requested = requested;

  const KernelPlacementInfo placement = kernel_placement_info(requested, op, dtype);
  plan.selected = placement.selected;
  plan.uses_fallback = placement.fallback_to_cpu;

  if (input_devices.empty()) {
    plan.input_devices.push_back(plan.selected);
  } else {
    plan.input_devices = input_devices;
  }

  bool transfers_supported = true;
  for (const Device& input_device : plan.input_devices) {
    DeviceTransferInfo transfer = device_transfer_info(input_device, plan.selected);
    if (input_device != plan.selected) {
      plan.requires_transfer = true;
    }
    if (!transfer.supported) {
      transfers_supported = false;
    }
    if (!transfer.direct) {
      plan.stream_required = true;
    }
    plan.input_transfers.push_back(std::move(transfer));
  }

  const BackendRuntimeInfo runtime = backend_runtime_info(plan.selected);
  const bool selected_kernel = backend_supports_kernel(plan.selected, op, dtype);
  plan.executable = runtime.executable && selected_kernel && transfers_supported;

  const BackendOpInfo op_metadata = backend_op_info(op);
  plan.graph_compatible = runtime.supports_graph && op_metadata.shape_inference && !op_metadata.stateful;
  if (plan.executable) {
    plan.reason = plan.uses_fallback
        ? "requested placement falls back to an executable CPU kernel with supported input transfers"
        : "requested placement is executable on the selected backend";
    return plan;
  }
  if (!runtime.executable) {
    plan.reason = "selected backend '" + plan.selected.str() + "' does not have an executable runtime";
    return plan;
  }
  if (!selected_kernel) {
    plan.reason =
        "selected backend '" + plan.selected.str() + "' does not provide kernel '" + op +
        "' for dtype '" + dtype_name(dtype) + "'";
    return plan;
  }
  if (!transfers_supported) {
    plan.reason = "one or more input device transfers into selected backend '" + plan.selected.str() + "' are unsupported";
    return plan;
  }
  plan.reason = placement.reason;
  return plan;
}

void register_backend(BackendInfo info) {
  info.name = lowercase(info.name);
  validate_plugin_name(info.name);
  if (info.available || info.device_count != 0) {
    throw DeviceError(
        "external backend descriptors cannot be marked available until TensorStudio has backend storage hooks");
  }
  info.type = DeviceType::Plugin;
  if (info.reason.empty()) {
    info.reason = "registered external backend descriptor; executable storage is not implemented in this build";
  }

  std::lock_guard<std::mutex> lock(registry_mutex());
  auto it = find_plugin_backend_unlocked(info.name);
  if (it == plugin_backends().end()) {
    plugin_backends().push_back(std::move(info));
  } else {
    *it = std::move(info);
  }
}

void unregister_backend(const std::string& name) {
  const std::string normalized = lowercase(name);
  validate_plugin_name(normalized);
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto& backends = plugin_backends();
  backends.erase(
      std::remove_if(backends.begin(), backends.end(), [&](const BackendInfo& info) {
        return info.name == normalized;
      }),
      backends.end());
  auto& kernels = plugin_kernels();
  kernels.erase(
      std::remove_if(kernels.begin(), kernels.end(), [&](const BackendKernelInfo& info) {
        return info.backend == normalized;
      }),
      kernels.end());
}

void register_backend_kernel(BackendKernelInfo info) {
  info.backend = lowercase(info.backend);
  validate_plugin_name(info.backend);
  validate_op_name(info.op);
  if (info.dtypes.empty()) {
    throw DeviceError("backend kernel registration requires at least one dtype");
  }
  if (info.priority < 0) {
    throw DeviceError("backend kernel priority must be non-negative");
  }
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    if (find_plugin_backend_unlocked_const(info.backend) == plugin_backends().end()) {
      throw DeviceError("plugin backend '" + info.backend + "' must be registered before registering kernels");
    }
    info.type = DeviceType::Plugin;
    info.memory_space = info.memory_space == MemorySpace::Unified ? MemorySpace::Unified : MemorySpace::Device;
    if (info.reason.empty()) {
      info.reason = "registered external backend kernel metadata";
    }
    plugin_kernels().push_back(std::move(info));
  }
}

}  // namespace tensorstudio
