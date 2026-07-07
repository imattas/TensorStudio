#include "tensorstudio/device.hpp"

namespace tensorstudio {

Device::Device() : type_(DeviceType::CPU) {}

Device::Device(DeviceType type) : type_(type) {}

DeviceType Device::type() const {
  return type_;
}

std::string Device::str() const {
  return "cpu";
}

Device cpu_device() {
  return Device(DeviceType::CPU);
}

}  // namespace tensorstudio
