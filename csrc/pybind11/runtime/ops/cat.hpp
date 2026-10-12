#pragma once

#include "runtime/context.hpp"
#include "execution/tensor_checks.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinilm::runtime::ops {

inline void bind_cat(py::module &m) {

    m.def("cat",
          [](std::vector<Tensor> tensors, int dim) {
              if (tensors.empty()) throw std::invalid_argument("Cat requires input tensors");
              const Tensor *reference = nullptr;
              for (const auto &input : tensors) {
                  if (!input) throw std::invalid_argument("Cat input is null");
                  reference = &input;
                  if (!infinilm::execution::tensor_checks::cat::empty_vector(input)) break;
              }
              auto owner = (*reference)->allocation_owner();
              if (!owner || owner->device() != (*reference)->device()) owner = context::resolveRuntime((*reference)->device());
              auto &runtime = *owner;
              auto layout = infinilm::execution::tensor_checks::cat::analyze(runtime, tensors, dim);
              auto shape = (*layout.reference)->shape();
              shape[layout.axis] = layout.length;
              auto output = Tensor::empty(runtime, shape, (*layout.reference)->dtype());
              infinilm::execution::tensor_checks::cat::validate(runtime, output, tensors, layout);
              tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                  infinilm::execution::tensor_checks::cat::empty_vector), tensors.end());
              if (output->numel()) {
                  infinilm::execution::submit<infini::ops::Cat>(runtime,
                      infini::ops::Cat::DefaultConfig(runtime.device().type()),
                      tensors, static_cast<int64_t>(layout.axis), output);
              }
              return output;
          },
          py::arg("tensors"),
          py::arg("dim") = 0,
          R"doc(opertor: torch.cat, out-of-place mode)doc");

    m.def("cat_",
          [](Tensor out, std::vector<Tensor> tensors, int dim) {
              if (!out) throw std::invalid_argument("Cat output is null");
              auto owner = out->allocation_owner();
              if (!owner || owner->device() != out->device()) owner = context::resolveRuntime(out->device());
              auto &runtime = *owner;
              const auto rank = out->ndim();
              if (!rank || dim < -static_cast<int64_t>(rank) || dim >= static_cast<int64_t>(rank))
                  throw std::invalid_argument("Cat dimension is outside the output rank");
              const auto axis = static_cast<size_t>(dim < 0 ? dim + static_cast<int64_t>(rank) : dim);
              infinilm::execution::tensor_checks::cat::validate(runtime, out, tensors, {nullptr, axis, 0});
              tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                  infinilm::execution::tensor_checks::cat::empty_vector), tensors.end());
              if (out->numel()) {
                  infinilm::execution::submit<infini::ops::Cat>(runtime,
                      infini::ops::Cat::DefaultConfig(runtime.device().type()),
                      tensors, static_cast<int64_t>(axis), out);
              }
          },
          py::arg("out"),
          py::arg("tensors"),
          py::arg("dim") = 0,
          R"doc(opertor: torch.cat, in-place mode)doc");
}

} // namespace infinilm::runtime::ops
