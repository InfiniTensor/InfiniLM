#include "execution/submit.hpp"
#pragma once

#include <pybind11/pybind11.h>

#include "runtime/context.hpp"

namespace py = pybind11;

namespace infinilm::runtime::ops {

inline void bind_add(py::module &m) {
    m.def("add", [](const Tensor &a, const Tensor &b) {
              if (!a){ throw std::invalid_argument("Add input is null");
}
              auto owner = a->allocation_owner();
              if (!owner || owner->device() != a->device()){ owner = context::resolveRuntime(a->device());
}
              auto &ops_runtime = *owner;
              if (!a || a->device() != ops_runtime.device()){
                  throw std::invalid_argument("Ops output allocation requires a nonnull input on the Runtime device");
}
              auto ops_output = infinilm::runtime::Tensor::empty(ops_runtime, a->shape(), a->dtype());
              infinilm::execution::submit<infini::ops::Add>(
                  ops_runtime, infini::ops::Add::DefaultConfig(ops_runtime.device().type()), a, b, ops_output);
              INFINILM_TRACE_OP(Add, ops_output);
              return ops_output; }, py::arg("a"), py::arg("b"), R"doc(Addition of two tensors.)doc");
}

} // namespace infinilm::runtime::ops
