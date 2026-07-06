#pragma once

#include <stdexcept>
#include <string>

namespace tensorstudio {

class TensorStudioError : public std::runtime_error {
 public:
  explicit TensorStudioError(const std::string& message);
};

class ShapeError : public TensorStudioError {
 public:
  explicit ShapeError(const std::string& message);
};

class DTypeError : public TensorStudioError {
 public:
  explicit DTypeError(const std::string& message);
};

class DeviceError : public TensorStudioError {
 public:
  explicit DeviceError(const std::string& message);
};

class AutogradError : public TensorStudioError {
 public:
  explicit AutogradError(const std::string& message);
};

}  // namespace tensorstudio
