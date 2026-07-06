#pragma once

#include <string>

namespace tensorstudio {

enum class DeviceType {
  CPU,
};

class Device {
 public:
  Device();
  explicit Device(DeviceType type);

  DeviceType type() const;
  std::string str() const;

 private:
  DeviceType type_;
};

Device cpu_device();

}  // namespace tensorstudio
