#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tensorstudio {

enum class DeviceType {
  CPU,
  CUDA,
  Metal,
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
  explicit Device(DeviceType type, int64_t index = 0);

  DeviceType type() const;
  int64_t index() const;
  std::string str() const;
  bool is_cpu() const;
  bool is_accelerator() const;

  bool operator==(const Device& other) const;
  bool operator!=(const Device& other) const;

 private:
  DeviceType type_;
  int64_t index_;
};

Device cpu_device();
Device parse_device(const std::string& spec);
std::string device_type_name(DeviceType type);
bool is_device_available(const Device& device);
int64_t device_count(DeviceType type);
std::vector<Device> available_devices();
std::vector<BackendInfo> backend_info();

}  // namespace tensorstudio
