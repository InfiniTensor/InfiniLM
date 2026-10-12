#include "execution/submit.hpp"
#include <base/gemm.h>
#pragma once

#include <pybind11/pybind11.h>

#include "runtime/context.hpp"

namespace py = pybind11;

namespace infinilm::runtime::ops {

inline void bind_matmul(py::module &m) {
    m.def("matmul",
          [](Tensor a, Tensor b, float alpha) {
              if (!a) throw std::invalid_argument("Gemm input is null");
              auto owner = a->allocation_owner();
              if (!owner || owner->device() != a->device()) owner = context::resolveRuntime(a->device());
              auto &runtime = *owner;
              auto matmul_output_shape = a->shape();
              matmul_output_shape[a->ndim() - 1] = b->size(a->ndim() - 1);
              auto matmul_output = infinilm::runtime::Tensor::empty(runtime, matmul_output_shape, a->dtype());
              infinilm::execution::submit<infini::ops::Gemm>(
                  runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
                  infini::ops::Gemm::DefaultConfig(runtime.device().type()), a, b,
                  std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{alpha},
                  std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, matmul_output);
              INFINILM_TRACE_OP(Gemm, matmul_output);
              return matmul_output;
          },
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          R"doc(Matrix multiplication of two tensors.)doc");

    m.def("matmul_",
          [](Tensor output, Tensor a, Tensor b, float alpha) {
              if (!output) throw std::invalid_argument("Gemm output is null");
              auto owner = output->allocation_owner();
              if (!owner || owner->device() != output->device()) owner = context::resolveRuntime(output->device());
              auto &runtime = *owner;
              infinilm::execution::submit<infini::ops::Gemm>(
                  runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
                  infini::ops::Gemm::DefaultConfig(runtime.device().type()),
                  a, b, std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{alpha},
                  std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, output);
              INFINILM_TRACE_OP(Gemm, output);
          },
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          R"doc(In-place matrix multiplication.)doc");
}

} // namespace infinilm::runtime::ops
