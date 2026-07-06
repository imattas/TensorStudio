#include "bindings.hpp"

#include "tensorstudio/module.hpp"

namespace tensorstudio::bindings {

void bind_nn(py::module_& module) {
  module.def("_module_system_note", &module_system_note);
}

}  // namespace tensorstudio::bindings
