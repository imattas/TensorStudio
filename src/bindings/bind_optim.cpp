#include "bindings.hpp"

#include "tensorstudio/optim.hpp"

namespace tensorstudio::bindings {

void bind_optim(py::module_& module) {
  module.def("_optimizer_system_note", &optimizer_system_note);
}

}  // namespace tensorstudio::bindings
