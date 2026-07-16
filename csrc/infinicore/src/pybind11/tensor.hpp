#pragma once

#include "infinicore.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::tensor {

Tensor from_list_py(py::handle data, DataType dtype);

inline void bind(py::module &m) {
    py::class_<Tensor>(m, "Tensor")
        .def_property_readonly("shape", [](const Tensor &tensor) { return tensor->shape(); })
        .def_property_readonly("strides", [](const Tensor &tensor) { return tensor->strides(); })
        .def_property_readonly("ndim", [](const Tensor &tensor) { return tensor->ndim(); })
        .def_property_readonly("dtype", [](const Tensor &tensor) { return tensor->dtype(); })
        .def_property_readonly("device", [](const Tensor &tensor) { return tensor->device(); })
        .def("data_ptr", [](const Tensor &tensor) { return reinterpret_cast<std::uintptr_t>(tensor->data()); })
        .def("size", [](const Tensor &tensor, std::size_t dim) { return tensor->size(dim); })
        .def("stride", [](const Tensor &tensor, std::size_t dim) { return tensor->stride(dim); })
        .def("numel", [](const Tensor &tensor) { return tensor->numel(); })
        .def("is_contiguous", [](const Tensor &tensor) { return tensor->is_contiguous(); })
        .def("is_pinned", [](const Tensor &tensor) { return tensor->is_pinned(); })
        .def("info", [](const Tensor &tensor) { return tensor->info(); })

        .def("debug", [](const Tensor &tensor) { return tensor->debug(); })
        .def("debug", [](const Tensor &tensor, const std::string &filename) { return tensor->debug(filename); })

        .def("copy_", [](Tensor &tensor, const Tensor &other) { tensor->copy_from(other); })
        .def("to", [](const Tensor &tensor, const Device &device) { return tensor->to(device); })
        .def("contiguous", [](const Tensor &tensor) { return tensor->contiguous(); })

        .def("as_strided", [](const Tensor &tensor, const Shape &shape, const Strides &strides) { return tensor->as_strided(shape, strides); })
        .def("narrow", [](const Tensor &tensor, std::size_t dim, std::size_t start, std::size_t length) { return tensor->narrow({{dim, start, length}}); })
        .def("permute", [](const Tensor &tensor, const Shape &dims) { return tensor->permute(dims); })
        .def("view", [](const Tensor &tensor, const Shape &shape) { return tensor->view(shape); })
        .def("unsqueeze", [](const Tensor &tensor, std::size_t dim) { return tensor->unsqueeze(dim); })
        .def("squeeze", [](const Tensor &tensor, std::size_t dim) { return tensor->squeeze(dim); })
        .def("reset", static_cast<void (Tensor::*)() noexcept>(&Tensor::reset))
        .def("use_count", &Tensor::use_count)
        .def("__str__", [](const Tensor &tensor) {
            std::ostringstream oss;
            oss << tensor;
            return oss.str();
        })
        .def("__repr__", [](const Tensor &tensor) {
            std::ostringstream oss;
            oss << tensor;
            return oss.str();
        })
        .def("__bool__", [](const Tensor &tensor) {
            return bool(tensor);
        });

    using EmptyFuncType = Tensor (*)(const Shape &, const DataType &, const Device &, bool);
    using StridedEmptyFuncType = Tensor (*)(const Shape &, const Strides &, const DataType &, const Device &, bool);

    m.def("empty", static_cast<EmptyFuncType>(&Tensor::empty),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("pin_memory") = false);

    m.def("strided_empty", static_cast<StridedEmptyFuncType>(&Tensor::strided_empty),
          py::arg("shape"),
          py::arg("strides"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("pin_memory") = false);

    m.def("zeros", static_cast<EmptyFuncType>(&Tensor::zeros),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("pin_memory") = false);

    m.def(
        "_from_numpy_copy", [](py::buffer buffer, const DataType &dtype) {
            const py::buffer_info info = buffer.request();
            if (info.itemsize < 0 || static_cast<size_t>(info.itemsize) != dsize(dtype)) {
                throw py::value_error("NumPy item size does not match the target dtype");
            }

            Shape shape;
            shape.reserve(info.shape.size());
            for (const auto dim : info.shape) {
                if (dim < 0) {
                    throw py::value_error("NumPy shape must not contain negative dimensions");
                }
                shape.push_back(static_cast<Size>(dim));
            }

            py::ssize_t expected_stride = info.itemsize;
            for (py::ssize_t dim = info.ndim; dim-- > 0;) {
                if (info.shape[dim] > 1 && info.strides[dim] != expected_stride) {
                    throw py::value_error("NumPy array must be C-contiguous");
                }
                expected_stride *= info.shape[dim];
            }

            auto result = Tensor::empty(shape, dtype, Device(Device::Type::kCpu, 0));
            if (result->nbytes() != static_cast<size_t>(info.size) * dsize(dtype)) {
                throw py::value_error("NumPy buffer size does not match its shape");
            }
            if (result->nbytes() != 0) {
                std::memcpy(result->data(), info.ptr, result->nbytes());
            }
            return result;
        },
        py::arg("array"), py::arg("dtype"));

    m.def(
        "from_blob", [](uintptr_t raw_ptr, Shape &shape, const DataType &dtype, const Device &device) {
            return Tensor{infinicore::Tensor::from_blob(reinterpret_cast<void *>(raw_ptr), shape, dtype, device)};
        },
        pybind11::arg("raw_ptr"), pybind11::arg("shape"), pybind11::arg("dtype"), pybind11::arg("device"));

    m.def(
        "strided_from_blob", [](uintptr_t raw_ptr, Shape &shape, Strides &strides, const DataType &dtype, const Device &device) {
            return Tensor{infinicore::Tensor::strided_from_blob(reinterpret_cast<void *>(raw_ptr), shape, strides, dtype, device)};
        },
        pybind11::arg("raw_ptr"), pybind11::arg("shape"), pybind11::arg("strides"), pybind11::arg("dtype"), pybind11::arg("device"));

    m.def("from_list", &from_list_py, py::arg("data"), py::arg("dtype"));
}

} // namespace infinicore::tensor
