#pragma once

#include "../../config/model_config.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

namespace infinilm::config {

inline void bind_model_config(py::module &m) {
    py::class_<ModelConfig, std::shared_ptr<ModelConfig>>(m, "ModelConfig")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const std::string &>(), py::arg("path"),
             "Constructor from config file path")
        // Template get methods - bind common types
        .def("get", [](const ModelConfig &self, const std::string &key) -> size_t { return self.get<size_t>(key); }, py::arg("key"), "Get integer value by key")
        .def("get", [](const ModelConfig &self, const std::string &key) -> double { return self.get<double>(key); }, py::arg("key"), "Get float value by key")
        .def("get", [](const ModelConfig &self, const std::string &key) -> std::string { return self.get<std::string>(key); }, py::arg("key"), "Get string value by key")
        .def("get", [](const ModelConfig &self, const std::string &key) -> bool { return self.get<bool>(key); }, py::arg("key"), "Get boolean value by key")
        // Template get_or methods - bind common types
        .def("get_or", [](const ModelConfig &self, const std::string &key, size_t default_value) -> size_t { return self.get_or<size_t>(key, default_value); }, py::arg("key"), py::arg("default_value"), "Get integer value by key or return default")
        .def("get_or", [](const ModelConfig &self, const std::string &key, double default_value) -> double { return self.get_or<double>(key, default_value); }, py::arg("key"), py::arg("default_value"), "Get float value by key or return default")
        .def("get_or", [](const ModelConfig &self, const std::string &key, const std::string &default_value) -> std::string { return self.get_or<std::string>(key, default_value); }, py::arg("key"), py::arg("default_value"), "Get string value by key or return default")
        .def("get_or", [](const ModelConfig &self, const std::string &key, bool default_value) -> bool { return self.get_or<bool>(key, default_value); }, py::arg("key"), py::arg("default_value"), "Get boolean value by key or return default")
        // Specialized getter methods
        .def("get_eos_token_ids", &ModelConfig::get_eos_token_ids, "Get eos_token_id as list (handles both int and array in config)")
        .def("get_kv_dim", &ModelConfig::get_kv_dim, "Get KV dimension (hidden_size * num_key_value_heads / num_attention_heads)")
        .def("get_head_dim", &ModelConfig::get_head_dim, "Get head dimension")
        .def("get_dtype", &ModelConfig::get_dtype, "Get data type from config")
        .def("get_quant_scheme", &ModelConfig::get_quant_scheme, "Get quantization scheme")
        .def("get_quant_config", &ModelConfig::get_quant_config, "Get quantization config")
        .def("get_quantization_method", &ModelConfig::get_quantization_method, "Get quantization method")
        .def("get_rope_scaling", &ModelConfig::get_rope_scaling, "Get RoPE scaling configuration")
        .def("__repr__", [](const ModelConfig &self) {
            std::ostringstream oss;
            oss << self;
            return "<ModelConfig>\n" + oss.str(); })
        .def("__str__", [](const ModelConfig &self) {
            std::ostringstream oss;
            oss << self;
            return oss.str(); });
}

} // namespace infinilm::config
