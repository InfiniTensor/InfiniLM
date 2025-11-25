#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "models/llama/llama.hpp"
#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"
#include "infinicore/nn/module.hpp"

namespace py = pybind11;
using infinicore::Device;

namespace infinilm::models::llama {

inline void bind_llama(py::module &m) {
    // Bind HookRegistry
    py::class_<HookRegistry, std::shared_ptr<HookRegistry>>(m, "HookRegistry")
        .def(py::init<>())
        .def("register_hook", [](HookRegistry &self, const std::string &name, py::object callback) {
            // Convert Python callable to C++ function
            self.register_hook(name, [callback](const std::string &hook_name, const infinicore::Tensor &tensor, int layer_idx) {
                try {
                    // Call Python callback with hook name, tensor, and layer index
                    callback(hook_name, tensor, layer_idx);
                } catch (const py::error_already_set &e) {
                    // Re-raise Python exception
                    throw;
                }
            });
        }, py::arg("name"), py::arg("callback"))
        .def("clear", &HookRegistry::clear)
        .def("has_hooks", &HookRegistry::has_hooks);

    // Bind LlamaConfig
    py::class_<LlamaConfig> config(m, "LlamaConfig");
    config
        .def(py::init<>())
        .def_readwrite("vocab_size", &LlamaConfig::vocab_size)
        .def_readwrite("hidden_size", &LlamaConfig::hidden_size)
        .def_readwrite("intermediate_size", &LlamaConfig::intermediate_size)
        .def_readwrite("num_hidden_layers", &LlamaConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &LlamaConfig::num_attention_heads)
        .def_readwrite("num_key_value_heads", &LlamaConfig::num_key_value_heads)
        .def_readwrite("head_dim", &LlamaConfig::head_dim)
        .def_readwrite("max_position_embeddings", &LlamaConfig::max_position_embeddings)
        .def_readwrite("rms_norm_eps", &LlamaConfig::rms_norm_eps)
        .def_readwrite("rope_theta", &LlamaConfig::rope_theta)
        .def_readwrite("attention_bias", &LlamaConfig::attention_bias)
        .def_readwrite("mlp_bias", &LlamaConfig::mlp_bias)
        .def_readwrite("tie_word_embeddings", &LlamaConfig::tie_word_embeddings)
        .def("validate", &LlamaConfig::validate)
        .def("kv_dim", &LlamaConfig::kv_dim);

    // Note: Device is already bound in InfiniCore bindings, so we don't need to bind it here

    // Helper function to convert Python object (InfiniCore tensor, numpy array, or torch tensor) to C++ Tensor
    auto convert_to_tensor = [](py::object obj, const Device &device) -> infinicore::Tensor {
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
    };

    // Bind LlamaModel
    py::class_<LlamaModel, std::shared_ptr<LlamaModel>>(m, "LlamaModel")
        .def(py::init<const LlamaConfig &, const Device &>(),
             py::arg("config"), py::arg("device"))
        .def("state_dict", [](const LlamaModel &model) {
            // Convert state_dict to Python dict with shape information
            auto state_dict = model.state_dict();
            py::dict result;
            for (const auto &[name, param] : state_dict) {
                // Parameter is a shared_ptr<Tensor>, get shape from it
                py::dict param_info;
                param_info["shape"] = py::cast(param->shape());
                param_info["dtype"] = py::cast(static_cast<int>(param->dtype()));
                result[py::cast(name)] = param_info;
            }
            return result;
        })
        .def("get_parameter", [](const LlamaModel &model, const std::string &name) {
            // Get actual tensor parameter by name
            auto state_dict = model.state_dict();
            auto it = state_dict.find(name);
            if (it != state_dict.end()) {
                // Parameter inherits from Tensor, cast to Tensor for pybind11
                const infinicore::Tensor &tensor = it->second;
                return tensor;
            }
            throw std::runtime_error("Parameter '" + name + "' not found in model");
        }, py::arg("name"))
        .def("load_state_dict", [convert_to_tensor](LlamaModel &model, py::dict state_dict, const Device &device) {
            // Convert Python dict to C++ state_dict
            std::unordered_map<std::string, infinicore::Tensor> cpp_state_dict;
            for (auto item : state_dict) {
                std::string key = item.first.cast<std::string>();
                py::object value = item.second.cast<py::object>();
                cpp_state_dict.emplace(key, convert_to_tensor(value, device));
            }
            model.load_state_dict(cpp_state_dict);
        }, py::arg("state_dict"), py::arg("device"))
        .def("config", &LlamaModel::config, py::return_value_policy::reference_internal)
        .def("num_layers", &LlamaModel::num_layers)
        .def("forward", [convert_to_tensor](const LlamaModel &model, py::object input_ids, py::object position_ids, py::object kv_caches = py::none()) {
            // Helper to extract C++ tensor from Python object
            auto get_tensor = [convert_to_tensor](py::object obj) -> infinicore::Tensor {
                // If it's already a Python InfiniCore tensor wrapper, extract underlying
                if (py::hasattr(obj, "_underlying")) {
                    return obj.attr("_underlying").cast<infinicore::Tensor>();
                }
                // Try direct cast (in case it's already a C++ tensor)
                try {
                    return obj.cast<infinicore::Tensor>();
                } catch (const py::cast_error &) {
                    // Extract device from first tensor for conversion
                    Device device = Device(Device::Type::CPU, 0);
                    if (py::hasattr(obj, "device")) {
                        try {
                            auto py_device = obj.attr("device");
                            if (py::hasattr(py_device, "_underlying")) {
                                device = py_device.attr("_underlying").cast<Device>();
                            } else {
                                device = py_device.cast<Device>();
                            }
                        } catch (...) {
                            // Keep default CPU device
                        }
                    }
                    return convert_to_tensor(obj, device);
                }
            };

            // Convert Python tensors to C++ tensors
            auto infini_input_ids = get_tensor(input_ids);
            auto infini_position_ids = get_tensor(position_ids);

            // Handle kv_caches if provided
            std::vector<void *> *kv_caches_ptr = nullptr;
            if (!kv_caches.is_none()) {
                // TODO: Implement kv_caches conversion if needed
                // For now, we'll pass nullptr
                kv_caches_ptr = nullptr;
            }

            return model.forward(infini_input_ids, infini_position_ids, kv_caches_ptr);
        }, py::arg("input_ids"), py::arg("position_ids"), py::arg("kv_caches") = py::none());

    // Bind LlamaForCausalLM
    py::class_<LlamaForCausalLM, std::shared_ptr<LlamaForCausalLM>>(m, "LlamaForCausalLM")
        .def(py::init<const LlamaConfig &, const Device &>(),
             py::arg("config"), py::arg("device"))
        .def("state_dict", [](const LlamaForCausalLM &model) {
            // Convert state_dict to Python dict with shape information
            auto state_dict = model.state_dict();
            py::dict result;
            for (const auto &[name, param] : state_dict) {
                // Parameter is a shared_ptr<Tensor>, get shape from it
                py::dict param_info;
                param_info["shape"] = py::cast(param->shape());
                param_info["dtype"] = py::cast(static_cast<int>(param->dtype()));
                result[py::cast(name)] = param_info;
            }
            return result;
        })
        .def("get_parameter", [](const LlamaForCausalLM &model, const std::string &name) {
            // Get actual tensor parameter by name
            auto state_dict = model.state_dict();
            auto it = state_dict.find(name);
            if (it != state_dict.end()) {
                // Parameter inherits from Tensor, cast to Tensor for pybind11
                const infinicore::Tensor &tensor = it->second;
                return tensor;
            }
            throw std::runtime_error("Parameter '" + name + "' not found in model");
        }, py::arg("name"))
        .def("load_state_dict", [convert_to_tensor](LlamaForCausalLM &model, py::dict state_dict, const Device &device) {
            // Convert Python dict to C++ state_dict
            std::unordered_map<std::string, infinicore::Tensor> cpp_state_dict;
            for (auto item : state_dict) {
                std::string key = item.first.cast<std::string>();
                py::object value = item.second.cast<py::object>();
                cpp_state_dict.emplace(key, convert_to_tensor(value, device));
            }
            model.load_state_dict(cpp_state_dict);
        }, py::arg("state_dict"), py::arg("device"))
        .def("config", &LlamaForCausalLM::config, py::return_value_policy::reference_internal)
        .def("forward", [convert_to_tensor](const LlamaForCausalLM &model, py::object input_ids, py::object position_ids, py::object kv_caches = py::none(), py::object hook_registry = py::none()) {
            // Helper to extract C++ tensor from Python object
            auto get_tensor = [convert_to_tensor](py::object obj) -> infinicore::Tensor {
                // If it's already a Python InfiniCore tensor wrapper, extract underlying
                if (py::hasattr(obj, "_underlying")) {
                    return obj.attr("_underlying").cast<infinicore::Tensor>();
                }
                // Try direct cast (in case it's already a C++ tensor)
                try {
                    return obj.cast<infinicore::Tensor>();
                } catch (const py::cast_error &) {
                    // Extract device from first tensor for conversion
                    Device device = Device(Device::Type::CPU, 0);
                    if (py::hasattr(obj, "device")) {
                        try {
                            auto py_device = obj.attr("device");
                            if (py::hasattr(py_device, "_underlying")) {
                                device = py_device.attr("_underlying").cast<Device>();
                            } else {
                                device = py_device.cast<Device>();
                            }
                        } catch (...) {
                            // Keep default CPU device
                        }
                    }
                    return convert_to_tensor(obj, device);
                }
            };

            // Convert Python tensors to C++ tensors
            auto infini_input_ids = get_tensor(input_ids);
            auto infini_position_ids = get_tensor(position_ids);

            // Handle kv_caches if provided
            std::vector<void *> *kv_caches_ptr = nullptr;
            if (!kv_caches.is_none()) {
                // TODO: Implement kv_caches conversion if needed
                // For now, we'll pass nullptr
                kv_caches_ptr = nullptr;
            }

            // Handle hook_registry if provided
            const HookRegistry *hook_registry_ptr = nullptr;
            if (!hook_registry.is_none()) {
                try {
                    // Try to cast as shared_ptr first
                    auto hook_registry_shared = hook_registry.cast<std::shared_ptr<HookRegistry>>();
                    hook_registry_ptr = hook_registry_shared.get();
                } catch (const py::cast_error &) {
                    try {
                        // Try direct cast
                        hook_registry_ptr = hook_registry.cast<const HookRegistry *>();
                    } catch (const py::cast_error &) {
                        // Try to get underlying if it's a Python wrapper
                        if (py::hasattr(hook_registry, "_underlying")) {
                            auto underlying = hook_registry.attr("_underlying");
                            try {
                                auto hook_registry_shared = underlying.cast<std::shared_ptr<HookRegistry>>();
                                hook_registry_ptr = hook_registry_shared.get();
                            } catch (const py::cast_error &) {
                                hook_registry_ptr = underlying.cast<const HookRegistry *>();
                            }
                        }
                    }
                }
            }

            return model.forward(infini_input_ids, infini_position_ids, kv_caches_ptr, hook_registry_ptr);
        }, py::arg("input_ids"), py::arg("position_ids"), py::arg("kv_caches") = py::none(), py::arg("hook_registry") = py::none());
}

} // namespace infinilm::models::llama
