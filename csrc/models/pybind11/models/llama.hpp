#pragma once

#include "../../cache/kv_cache.hpp"
#include "../../debug_utils/hooks.hpp"
#include "../../llama/llama.hpp"
#include "../../llama/llama_attention.hpp"
#include "../utils.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using infinicore::Device;
using infinilm::models::debug_utils::HookRegistry;
using infinilm::models::utils::convert_to_tensor;

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
        .def_readwrite("mlp_bias", &LlamaConfig::mlp_bias)
        .def_readwrite("tie_word_embeddings", &LlamaConfig::tie_word_embeddings)
        .def_readwrite("use_cache", &LlamaConfig::use_cache)
        .def_readwrite("pad_token_id", &LlamaConfig::pad_token_id)
        .def_readwrite("bos_token_id", &LlamaConfig::bos_token_id)
        .def_readwrite("eos_token_id", &LlamaConfig::eos_token_id)
        .def("validate", &LlamaConfig::validate)
        .def("kv_dim", &LlamaConfig::kv_dim);

    // Note: Device is already bound in InfiniCore bindings, so we don't need to bind it here

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
            "load_state_dict", [](LlamaForCausalLM &model, py::dict state_dict, const Device &device) {
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
            "forward", [](const LlamaForCausalLM &model, py::object input_ids, py::object position_ids, py::object kv_caches = py::none()) {
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

                return model.forward(infini_input_ids, infini_position_ids, kv_caches_ptr);
            },
            py::arg("input_ids"), py::arg("position_ids"), py::arg("kv_caches") = py::none());
}

} // namespace infinilm::models::llama
