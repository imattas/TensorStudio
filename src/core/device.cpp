#include "tensorstudio/device.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "tensorstudio/errors.hpp"

namespace tensorstudio {
namespace {

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
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
  throw DeviceError("unsupported device type '" + name + "'; expected cpu, cuda, or metal");
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

BackendInfo make_backend_info(DeviceType type) {
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
        ? "CUDA backend hooks were compiled, but runtime kernels are not enabled in this build"
        : "CUDA backend was not compiled into this wheel";
    return info;
  }
  info.compiled = metal_compiled();
  info.available = false;
  info.device_count = 0;
  info.reason = info.compiled
      ? "Metal backend hooks were compiled, but runtime kernels are not enabled in this build"
      : "Metal backend was not compiled into this wheel";
  return info;
}

}  // namespace

Device::Device() : type_(DeviceType::CPU), index_(0) {}

Device::Device(DeviceType type, int64_t index) : type_(type), index_(index) {
  if (index < 0) {
    throw DeviceError("device index must be non-negative");
  }
  if (type == DeviceType::CPU && index != 0) {
    throw DeviceError("cpu device only supports index 0");
  }
}

DeviceType Device::type() const {
  return type_;
}

int64_t Device::index() const {
  return index_;
}

std::string Device::str() const {
  if (type_ == DeviceType::CPU) {
    return "cpu";
  }
  return device_type_name(type_) + ":" + std::to_string(index_);
}

bool Device::is_cpu() const {
  return type_ == DeviceType::CPU;
}

bool Device::is_accelerator() const {
  return type_ == DeviceType::CUDA || type_ == DeviceType::Metal;
}

bool Device::operator==(const Device& other) const {
  return type_ == other.type_ && index_ == other.index_;
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
  return Device(type, index);
}

std::string device_type_name(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    case DeviceType::Metal:
      return "metal";
  }
  throw DeviceError("unknown device type");
}

bool is_device_available(const Device& device) {
  return device.index() < device_count(device.type());
}

int64_t device_count(DeviceType type) {
  return make_backend_info(type).device_count;
}

std::vector<Device> available_devices() {
  std::vector<Device> result;
  for (const BackendInfo& info : backend_info()) {
    for (int64_t index = 0; index < info.device_count; ++index) {
      result.emplace_back(info.type, index);
    }
  }
  return result;
}

std::vector<BackendInfo> backend_info() {
  return {
      make_backend_info(DeviceType::CPU),
      make_backend_info(DeviceType::CUDA),
      make_backend_info(DeviceType::Metal),
  };
}

}  // namespace tensorstudio
