#pragma once

#include "../../models/minicpmv/minicpmv_config.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinilm::models::minicpmv {

inline void bind_minicpmv(py::module &m) {
    py::class_<SiglipVisionConfig>(m, "SiglipVisionConfig")
        .def(py::init<>())
        .def_readwrite("hidden_size", &SiglipVisionConfig::hidden_size)
        .def_readwrite("intermediate_size", &SiglipVisionConfig::intermediate_size)
        .def_readwrite("num_hidden_layers", &SiglipVisionConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &SiglipVisionConfig::num_attention_heads)
        .def_readwrite("image_size", &SiglipVisionConfig::image_size)
        .def_readwrite("patch_size", &SiglipVisionConfig::patch_size)
        .def_readwrite("layer_norm_eps", &SiglipVisionConfig::layer_norm_eps)
        .def_readwrite("hidden_act", &SiglipVisionConfig::hidden_act)
        .def_readwrite("model_type", &SiglipVisionConfig::model_type);

    py::class_<MiniCPMVConfig, InfinilmModel::Config>(m, "MiniCPMVConfig")
        .def(py::init<>())
        .def_readwrite("_dtype", &MiniCPMVConfig::dtype)
        .def_readwrite("model_type", &MiniCPMVConfig::model_type)
        .def_readwrite("llm_config", &MiniCPMVConfig::llm_config)
        .def_readwrite("vision_config", &MiniCPMVConfig::vision_config)
        .def_readwrite("query_num", &MiniCPMVConfig::query_num)
        .def_readwrite("drop_vision_last_layer", &MiniCPMVConfig::drop_vision_last_layer)
        .def_readwrite("batch_vision_input", &MiniCPMVConfig::batch_vision_input);
}

} // namespace infinilm::models::minicpmv
