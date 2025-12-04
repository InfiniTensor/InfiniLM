#pragma once

#include "../../wrapper/wrapper.hpp"
#include "../utils.hpp"
#include "infinicore/device.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinilm::models::llama {

/**
 * @brief Python bindings for LlamaForCausalLMWrapper
 *
 * This provides the same interface as llama.hpp bindings for LlamaForCausalLM,
 * allowing the wrapper to be a drop-in replacement in Python.
 */
inline void bind_wrapper(py::module &m) {
    // Bind LlamaForCausalLMWrapper with the same interface as LlamaForCausalLM
    py::class_<LlamaForCausalLMWrapper, std::shared_ptr<LlamaForCausalLMWrapper>>(
        m, "LlamaForCausalLMWrapper")
        .def(py::init([](const LlamaConfig &config, const infinicore::Device &device,
                         py::object dtype_obj) {
                 infinicore::DataType dtype = infinicore::DataType::F32;
                 if (!dtype_obj.is_none()) {
                     // Extract dtype from Python object
                     if (py::hasattr(dtype_obj, "_underlying")) {
                         dtype = dtype_obj.attr("_underlying").cast<infinicore::DataType>();
                     } else {
                         dtype = dtype_obj.cast<infinicore::DataType>();
                     }
                 }
                 return std::make_shared<LlamaForCausalLMWrapper>(config, device, dtype);
             }),
             py::arg("config"), py::arg("device"), py::arg("dtype") = py::none())
        .def("state_dict", [](const LlamaForCausalLMWrapper &model) {
            // Convert state_dict to Python dict with shape information
            auto state_dict = model.state_dict();
            py::dict result;
            for (const auto &[name, param] : state_dict) {
                // Parameter is a Parameter object, get shape from its tensor
                py::dict param_info;
                param_info["shape"] = py::cast(param->shape());
                param_info["dtype"] = py::cast(static_cast<int>(param->dtype()));
                result[py::cast(name)] = param_info;
            }
            return result;
        })
        .def(
            "get_parameter", [](const LlamaForCausalLMWrapper &model, const std::string &name) {
                // Get parameter object and return its tensor
                auto param = model.get_parameter(name);
                // Return the tensor from the Parameter object
                return static_cast<infinicore::Tensor>(param);
            },
            py::arg("name"))
        .def(
            "load_state_dict", [](LlamaForCausalLMWrapper &model, py::dict state_dict, const infinicore::Device &device) {
                // Convert Python dict to C++ state_dict
                std::unordered_map<std::string, infinicore::Tensor> cpp_state_dict;
                for (auto item : state_dict) {
                    std::string key = item.first.cast<std::string>();
                    py::object value = item.second.cast<py::object>();
                    cpp_state_dict.emplace(key, utils::convert_to_tensor(value, device));
                }
                model.load_state_dict(cpp_state_dict);
            },
            py::arg("state_dict"), py::arg("device"))
        .def("config", &LlamaForCausalLMWrapper::config, py::return_value_policy::reference_internal)
        .def(
            "forward", [](const LlamaForCausalLMWrapper &model, py::object input_ids, py::object position_ids, py::object kv_caches = py::none()) {
                // Helper to extract C++ tensor from Python object
                auto get_tensor = [](py::object obj) -> infinicore::Tensor {
                    // If it's already a Python InfiniCore tensor wrapper, extract underlying
                    if (py::hasattr(obj, "_underlying")) {
                        return obj.attr("_underlying").cast<infinicore::Tensor>();
                    }
                    // Try direct cast (in case it's already a C++ tensor)
                    try {
                        return obj.cast<infinicore::Tensor>();
                    } catch (const py::cast_error &) {
                        // Extract device from first tensor for conversion
                        infinicore::Device device = infinicore::Device(
                            infinicore::Device::Type::CPU, 0);
                        if (py::hasattr(obj, "device")) {
                            try {
                                auto py_device = obj.attr("device");
                                if (py::hasattr(py_device, "_underlying")) {
                                    device = py_device.attr("_underlying")
                                                 .cast<infinicore::Device>();
                                } else {
                                    device = py_device.cast<infinicore::Device>();
                                }
                            } catch (...) {
                                // Keep default CPU device
                            }
                        }
                        return utils::convert_to_tensor(obj, device);
                    }
                };

                // Convert Python tensors to C++ tensors
                auto infini_input_ids = get_tensor(input_ids);
                auto infini_position_ids = get_tensor(position_ids);

                // Handle kv_caches if provided
                std::vector<void *> *kv_caches_ptr = nullptr;

                return model.forward(infini_input_ids, infini_position_ids, kv_caches_ptr);
            },
            py::arg("input_ids"), py::arg("position_ids"), py::arg("kv_caches") = py::none());
}

} // namespace infinilm::models::llama
