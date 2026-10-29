#include <infini/ops.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "cache/cache.hpp"
#include "engine/engine.hpp"

namespace py = pybind11;

namespace infinilm::runtime {
void bind_runtime(py::module_ &m);
}

PYBIND11_MODULE(_infinilm, m) {
    m.doc() = "InfiniLM Python bindings";

    auto core = m.def_submodule("_core");
    infinilm::runtime::bind_runtime(core);

    infinilm::cache::bind_cache(m);
    infinilm::engine::bind_hook_registry(m);
    infinilm::engine::distributed::bind_dist_config(m);
    infinilm::engine::bind_infer_engine(m);

    // Loading-only CPU conversion. Storage stays with the caller; numerical
    // conversion and validation belong to Ops, not a second LM implementation.
    m.def("_cast_checkpoint_tensor_", [](infinilm::runtime::Tensor input, infinilm::runtime::Tensor output) {
        const auto checkpoint_float = [](infinilm::runtime::DataType dtype) {
            return dtype == infinilm::runtime::DataType::kFloat32
                || dtype == infinilm::runtime::DataType::kFloat16
                || dtype == infinilm::runtime::DataType::kBFloat16;
        };
        if (!input || !output
            || input->device().type() != infinilm::runtime::Device::Type::kCpu
            || output->device() != input->device()
            || input->shape() != output->shape()
            || !checkpoint_float(input->dtype()) || !checkpoint_float(output->dtype())) {
            throw std::invalid_argument("checkpoint Cast requires matching CPU shapes and FP32/FP16/BF16 tensors");
}
        if (input->numel() != 0 && input->data() == output->data()
            && input->dtype() != output->dtype()) {
            throw std::invalid_argument("checkpoint dtype conversion requires distinct storage");
        }
        const infini::ops::Tensor input_view{input->view()};
        const infini::ops::Tensor output_view{output->view()};
        infini::ops::Cast::Call(input_view, output_view); }, py::arg("input"), py::arg("output"), py::call_guard<py::gil_scoped_release>());

    m.def("_scale_checkpoint_tensor_", [](infinilm::runtime::Tensor input, double alpha, infinilm::runtime::Tensor output) {
        if (!input || !output
            || input->device().type() != infinilm::runtime::Device::Type::kCpu
            || output->device() != input->device()
            || input->shape() != output->shape() || input->dtype() != output->dtype()
            || (input->dtype() != infinilm::runtime::DataType::kFloat32
                && input->dtype() != infinilm::runtime::DataType::kFloat16
                && input->dtype() != infinilm::runtime::DataType::kBFloat16)) {
            throw std::invalid_argument("checkpoint scalar Mul requires matching CPU shapes and FP32/FP16/BF16 dtype");
        }
        const infini::ops::Tensor input_view{input->view()};
        const infini::ops::Tensor output_view{output->view()};
        infini::ops::Mul::Call(input_view, alpha, output_view); }, py::arg("input"), py::arg("alpha"), py::arg("output"), py::call_guard<py::gil_scoped_release>());

    m.def("_normalize_checkpoint_tensor_", [](infinilm::runtime::Tensor input, double eps, infinilm::runtime::Tensor output) {
        if (!input || !output
            || input->device().type() != infinilm::runtime::Device::Type::kCpu
            || output->device() != input->device()
            || input->shape() != output->shape() || input->ndim() != 2
            || input->dtype() != output->dtype()
            || (input->dtype() != infinilm::runtime::DataType::kFloat32
                && input->dtype() != infinilm::runtime::DataType::kFloat16
                && input->dtype() != infinilm::runtime::DataType::kBFloat16)) {
            throw std::invalid_argument("checkpoint RowL2Normalize requires matching rank2 CPU shapes and FP32/FP16/BF16 dtype");
        }
        const infini::ops::Tensor input_view{input->view()};
        const infini::ops::Tensor output_view{output->view()};
        infini::ops::RowL2Normalize::Call(input_view, eps, output_view); }, py::arg("input"), py::arg("eps"), py::arg("output"), py::call_guard<py::gil_scoped_release>());
}
