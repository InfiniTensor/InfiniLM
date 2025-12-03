#pragma once

#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinilm::models::utils {

// Helper function to convert Python object (InfiniCore tensor, numpy array, or torch tensor) to C++ Tensor
inline infinicore::Tensor convert_to_tensor(py::object obj, const infinicore::Device &device) {
    // First check if it's already an InfiniCore tensor (has _underlying attribute)
    if (py::hasattr(obj, "_underlying")) {
        try {
            // Extract the underlying C++ tensor from Python InfiniCore tensor
            auto underlying = obj.attr("_underlying");
            auto infini_tensor = underlying.cast<infinicore::Tensor>();
            return infini_tensor;
        } catch (const py::cast_error &) {
            // Fall through to other conversion methods
        }
    }

    // Try direct cast (in case it's already a C++ tensor exposed to Python)
    try {
        auto infini_tensor = obj.cast<infinicore::Tensor>();
        return infini_tensor;
    } catch (const py::cast_error &) {
        // Not an InfiniCore tensor, continue with other conversions
    }

    // Try to get data pointer and shape from numpy array or torch tensor
    void *data_ptr = nullptr;
    std::vector<size_t> shape;
    infinicore::DataType dtype = infinicore::DataType::F32;

    // Check if it's a numpy array
    if (py::hasattr(obj, "__array_interface__")) {
        auto array_info = obj.attr("__array_interface__");
        auto data = array_info["data"];
        if (py::isinstance<py::tuple>(data)) {
            auto data_tuple = data.cast<py::tuple>();
            data_ptr = reinterpret_cast<void *>(data_tuple[0].cast<uintptr_t>());
        } else {
            data_ptr = reinterpret_cast<void *>(data.cast<uintptr_t>());
        }

        auto shape_obj = array_info["shape"];
        if (py::isinstance<py::tuple>(shape_obj)) {
            auto shape_tuple = shape_obj.cast<py::tuple>();
            for (auto dim : shape_tuple) {
                shape.push_back(dim.cast<size_t>());
            }
        } else {
            shape.push_back(shape_obj.cast<size_t>());
        }

        // Get dtype
        std::string typestr = array_info["typestr"].cast<std::string>();
        if (typestr == "<f4" || typestr == "float32") {
            dtype = infinicore::DataType::F32;
        } else if (typestr == "<f2" || typestr == "float16") {
            dtype = infinicore::DataType::F16;
        } else if (typestr == "<i4" || typestr == "int32") {
            dtype = infinicore::DataType::I32;
        } else if (typestr == "<i8" || typestr == "int64") {
            dtype = infinicore::DataType::I64;
        }
    } else if (py::hasattr(obj, "data_ptr")) {
        // Try torch tensor
        data_ptr = reinterpret_cast<void *>(obj.attr("data_ptr")().cast<uintptr_t>());
        auto shape_obj = obj.attr("shape");
        if (py::isinstance<py::tuple>(shape_obj) || py::isinstance<py::list>(shape_obj)) {
            for (auto dim : shape_obj) {
                shape.push_back(dim.cast<size_t>());
            }
        } else {
            shape.push_back(shape_obj.cast<size_t>());
        }

        // Get dtype from torch tensor
        std::string dtype_str = py::str(obj.attr("dtype"));
        if (dtype_str.find("float32") != std::string::npos) {
            dtype = infinicore::DataType::F32;
        } else if (dtype_str.find("float16") != std::string::npos) {
            dtype = infinicore::DataType::F16;
        } else if (dtype_str.find("int32") != std::string::npos) {
            dtype = infinicore::DataType::I32;
        } else if (dtype_str.find("int64") != std::string::npos) {
            dtype = infinicore::DataType::I64;
        }
    } else {
        throw std::runtime_error("Unsupported tensor type. Expected InfiniCore tensor, numpy array, or torch tensor.");
    }

    return infinicore::Tensor::from_blob(data_ptr, shape, dtype, device);
}

} // namespace infinilm::models::utils
