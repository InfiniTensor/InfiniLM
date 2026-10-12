#include "execution/submit.hpp"
#pragma once

#include <pybind11/pybind11.h>

#include "runtime/context.hpp"

namespace py = pybind11;

namespace infinilm::runtime::ops {

inline void bind_add(py::module &m) {
    m.def("add",
          [](const Tensor &a, const Tensor &b) {
              if (!a) throw std::invalid_argument("Add input is null");
              auto owner = a->allocation_owner();
              if (!owner || owner->device() != a->device()) owner = context::resolveRuntime(a->device());
              auto &ops_runtime = *owner;
              if (!a || a->device() != ops_runtime.device())
                  throw std::invalid_argument("Ops output allocation requires a nonnull input on the Runtime device");
              auto ops_output = infinilm::runtime::Tensor::empty(ops_runtime, a->shape(), a->dtype());
              infinilm::execution::submit<infini::ops::Add>(
                  ops_runtime, infini::ops::Add::DefaultConfig(ops_runtime.device().type()), a, b, ops_output);
              INFINILM_TRACE_OP(Add, ops_output);
              return ops_output;
          },
          py::arg("a"),
          py::arg("b"),
          R"doc(Addition of two tensors.)doc");

    m.def("add_",
          [](Tensor c, const Tensor &a, const Tensor &b) {
              {
                  if (!c) throw std::invalid_argument("Add output is null");
                  auto owner = c->allocation_owner();
                  if (!owner || owner->device() != c->device()) owner = context::resolveRuntime(c->device());
                  auto &ops_runtime = *owner;
                  infinilm::execution::submit<infini::ops::Add>(
                      ops_runtime, infini::ops::Add::DefaultConfig(ops_runtime.device().type()), a, b, c);
                  INFINILM_TRACE_OP(Add, c);
              }
          },
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place tensor addition.)doc");
}

} // namespace infinilm::runtime::ops
