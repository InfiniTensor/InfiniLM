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

    py::class_<InfinilmModel::Config> config(m, "Config");

    // Bind LlamaConfig
    py::class_<LlamaConfig, InfinilmModel::Config> llama_config(m, "LlamaConfig");
    llama_config
        .def(py::init<>())
        // TODO: Change this to `dtype` after updating InfiniCore pybind11 exposing mechanism.
        .def_readwrite("_dtype", &LlamaConfig::dtype)
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
        .def_property("bos_token_id", [](const LlamaConfig &self) {
                // Always return as list to match Python config format
                return py::cast(self.bos_token_id); }, [](LlamaConfig &self, py::object value) {
                // Accept both single int and list
                if (py::isinstance<py::int_>(value)) {
                    self.bos_token_id = {value.cast<int64_t>()};
                } else if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
                    self.bos_token_id = value.cast<std::vector<int64_t>>();
                } else {
                    throw py::type_error("bos_token_id must be int or list of ints");
                } })
        .def_property("eos_token_id", [](const LlamaConfig &self) {
                // Always return as list to match Python config format
                return py::cast(self.eos_token_id); }, [](LlamaConfig &self, py::object value) {
                // Accept both single int and list
                if (py::isinstance<py::int_>(value)) {
                    self.eos_token_id = {value.cast<int64_t>()};
                } else if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
                    self.eos_token_id = value.cast<std::vector<int64_t>>();
                } else {
                    throw py::type_error("eos_token_id must be int or list of ints");
                } })
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
            return dir_list; });

    // Note: Device is already bound in InfiniCore bindings, so we don't need to bind it here
}

} // namespace infinilm::models::llama
