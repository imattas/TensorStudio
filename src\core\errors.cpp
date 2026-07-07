#include "tensorstudio/errors.hpp"

namespace tensorstudio {

TensorStudioError::TensorStudioError(const std::string& message) : std::runtime_error(message) {}

ShapeError::ShapeError(const std::string& message) : TensorStudioError(message) {}

DTypeError::DTypeError(const std::string& message) : TensorStudioError(message) {}

DeviceError::DeviceError(const std::string& message) : TensorStudioError(message) {}

AutogradError::AutogradError(const std::string& message) : TensorStudioError(message) {}

}  // namespace tensorstudio
