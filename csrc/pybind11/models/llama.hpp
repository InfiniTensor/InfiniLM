#pragma once

#include "../../cache/kv_cache.hpp"
#include "../../models/debug_utils/hooks.hpp"
#include "../../models/llama/llama.hpp"
#include "../../models/llama/llama_attention.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using infinicore::Device;
using infinilm::models::debug_utils::HookRegistry;

namespace infinilm::models::llama {

inline void bind_llama(py::module &m) {
    // TODO: HookRegistry should be moved out from Llama-specific bindings to InfiniCore as common utils in future work
    // Bind HookRegistry
    py::class_<HookRegistry, std::shared_ptr<HookRegistry>>(m, "HookRegistry")
        .def(py::init<>())
        .def(
            "register_hook", [](HookRegistry &self, const std::string &name, py::object callback) {
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
            },
            py::arg("name"), py::arg("callback"))
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
        .def_readwrite("hidden_act", &LlamaConfig::hidden_act)
        .def_readwrite("model_type", &LlamaConfig::model_type)
        .def_readwrite("rope_theta", &LlamaConfig::rope_theta)
        .def_readwrite("attention_bias", &LlamaConfig::attention_bias)
        .def_readwrite("attention_output_bias", &LlamaConfig::attention_output_bias)
        .def_readwrite("mlp_bias", &LlamaConfig::mlp_bias)
        .def_readwrite("tie_word_embeddings", &LlamaConfig::tie_word_embeddings)
        .def_readwrite("use_cache", &LlamaConfig::use_cache)
        .def_readwrite("attention_dropout", &LlamaConfig::attention_dropout)
        .def_readwrite("initializer_range", &LlamaConfig::initializer_range)
        .def_readwrite("pretraining_tp", &LlamaConfig::pretraining_tp)
        .def_readwrite("name_or_path", &LlamaConfig::name_or_path)
        .def_readwrite("pad_token_id", &LlamaConfig::pad_token_id)
        .def_property("bos_token_id",
            [](const LlamaConfig &self) {
                // Always return as list to match Python config format
                return py::cast(self.bos_token_id);
            },
            [](LlamaConfig &self, py::object value) {
                // Accept both single int and list
                if (py::isinstance<py::int_>(value)) {
                    self.bos_token_id = {value.cast<int64_t>()};
                } else if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
                    self.bos_token_id = value.cast<std::vector<int64_t>>();
                } else {
                    throw py::type_error("bos_token_id must be int or list of ints");
                }
            })
        .def_property("eos_token_id",
            [](const LlamaConfig &self) {
                // Always return as list to match Python config format
                return py::cast(self.eos_token_id);
            },
            [](LlamaConfig &self, py::object value) {
                // Accept both single int and list
                if (py::isinstance<py::int_>(value)) {
                    self.eos_token_id = {value.cast<int64_t>()};
                } else if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
                    self.eos_token_id = value.cast<std::vector<int64_t>>();
                } else {
                    throw py::type_error("eos_token_id must be int or list of ints");
                }
            })
        .def("validate", &LlamaConfig::validate)
        .def("kv_dim", &LlamaConfig::kv_dim)
        // Add __dir__ to make attributes discoverable via dir() in Python
        .def("__dir__", [](const LlamaConfig &self) {
            py::list dir_list;
            dir_list.append("vocab_size");
            dir_list.append("hidden_size");
            dir_list.append("intermediate_size");
            dir_list.append("num_hidden_layers");
            dir_list.append("num_attention_heads");
            dir_list.append("num_key_value_heads");
            dir_list.append("head_dim");
            dir_list.append("max_position_embeddings");
            dir_list.append("rms_norm_eps");
            dir_list.append("hidden_act");
            dir_list.append("model_type");
            dir_list.append("rope_theta");
            dir_list.append("attention_bias");
            dir_list.append("attention_output_bias");
            dir_list.append("mlp_bias");
            dir_list.append("tie_word_embeddings");
            dir_list.append("use_cache");
            dir_list.append("attention_dropout");
            dir_list.append("initializer_range");
            dir_list.append("pretraining_tp");
            dir_list.append("name_or_path");
            dir_list.append("pad_token_id");
            dir_list.append("bos_token_id");
            dir_list.append("eos_token_id");
            dir_list.append("validate");
            dir_list.append("kv_dim");
            return dir_list;
        });

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
            } else if (typestr == "bfloat16") {
                dtype = infinicore::DataType::BF16;
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
            } else if (dtype_str.find("bfloat16") != std::string::npos) {
                dtype = infinicore::DataType::BF16;
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

    // Bind LlamaForCausalLM
    py::class_<LlamaForCausalLM, std::shared_ptr<LlamaForCausalLM>>(m, "LlamaForCausalLM")
        .def(py::init([](const LlamaConfig &config, const Device &device, py::object dtype_obj) {
                 infinicore::DataType dtype = infinicore::DataType::F32;
                 if (!dtype_obj.is_none()) {
                     // Extract dtype from Python object
                     if (py::hasattr(dtype_obj, "_underlying")) {
                         dtype = dtype_obj.attr("_underlying").cast<infinicore::DataType>();
                     } else {
                         dtype = dtype_obj.cast<infinicore::DataType>();
                     }
                 }
                 return std::make_shared<LlamaForCausalLM>(config, device, dtype);
             }),
             py::arg("config"), py::arg("device"), py::arg("dtype") = py::none())
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
        .def(
            "get_parameter", [](const LlamaForCausalLM &model, const std::string &name) {
                // Get actual tensor parameter by name
                auto state_dict = model.state_dict();
                auto it = state_dict.find(name);
                if (it != state_dict.end()) {
                    // Parameter inherits from Tensor, cast to Tensor for pybind11
                    const infinicore::Tensor &tensor = it->second;
                    return tensor;
                }
                throw std::runtime_error("Parameter '" + name + "' not found in model");
            },
            py::arg("name"))
        .def(
            "load_state_dict", [convert_to_tensor](LlamaForCausalLM &model, py::dict state_dict, const Device &device) {
                // Convert Python dict to C++ state_dict
                std::unordered_map<std::string, infinicore::Tensor> cpp_state_dict;
                for (auto item : state_dict) {
                    std::string key = item.first.cast<std::string>();
                    py::object value = item.second.cast<py::object>();
                    cpp_state_dict.emplace(key, convert_to_tensor(value, device));
                }
                model.load_state_dict(cpp_state_dict);
            },
            py::arg("state_dict"), py::arg("device"))
        .def("config", &LlamaForCausalLM::config, py::return_value_policy::reference_internal)
        .def(
            "cache", [](const LlamaForCausalLM &model) {
            // Get the cache from the underlying model as an opaque pointer
            return model.model().cache();
        }, "Get the internal cache as an opaque pointer")
        .def(
            "reset_cache", [](const LlamaForCausalLM &model, bool full_reset = true) {
            // Reset the internal cache to prevent state from persisting between generations
            model.model().reset_cache(full_reset);
        }, py::arg("full_reset") = true, "Reset the internal cache (clears state between generations)")
        .def("forward", [convert_to_tensor](const LlamaForCausalLM &model, py::object input_ids, py::object position_ids, py::object kv_cache = py::none()) {
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

            // Handle kv_cache if provided (model-level DynamicCache)
            void *kv_cache_ptr = nullptr;
            if (!kv_cache.is_none()) {
                // Try to extract DynamicCache from Python object
                if (py::hasattr(kv_cache, "_underlying")) {
                    kv_cache_ptr = kv_cache.attr("_underlying").cast<void *>();
                } else {
                    // Try direct cast
                    try {
                        kv_cache_ptr = kv_cache.cast<void *>();
                    } catch (...) {
                        // If conversion fails, pass nullptr (cache will be ignored)
                        kv_cache_ptr = nullptr;
                    }
                }
            }

            return model.forward(infini_input_ids, infini_position_ids, kv_cache_ptr);
        }, py::arg("input_ids"), py::arg("position_ids"), py::arg("kv_caches") = py::none());
}

} // namespace infinilm::models::llama
