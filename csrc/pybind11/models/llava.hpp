#pragma once

#include "../../models/llava/llava_config.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinilm::models::llava {

inline void bind_llava(py::module &m) {
    py::class_<ClipVisionConfig>(m, "ClipVisionConfig")
        .def(py::init<>())
        .def_readwrite("hidden_size", &ClipVisionConfig::hidden_size)
        .def_readwrite("intermediate_size", &ClipVisionConfig::intermediate_size)
        .def_readwrite("num_hidden_layers", &ClipVisionConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &ClipVisionConfig::num_attention_heads)
        .def_readwrite("image_size", &ClipVisionConfig::image_size)
        .def_readwrite("patch_size", &ClipVisionConfig::patch_size)
        .def_readwrite("layer_norm_eps", &ClipVisionConfig::layer_norm_eps)
        .def_readwrite("model_type", &ClipVisionConfig::model_type);

    py::class_<LlavaConfig, InfinilmModel::Config>(m, "LlavaConfig")
        .def(py::init<>())
        .def_readwrite("_dtype", &LlavaConfig::dtype)
        .def_readwrite("model_type", &LlavaConfig::model_type)
        .def_readwrite("image_token_index", &LlavaConfig::image_token_index)
        .def_readwrite("pad_token_id", &LlavaConfig::pad_token_id)
        .def_readwrite("vocab_size", &LlavaConfig::vocab_size)
        .def_readwrite("ignore_index", &LlavaConfig::ignore_index)
        .def_readwrite("projector_hidden_act", &LlavaConfig::projector_hidden_act)
        .def_readwrite("vision_feature_layer", &LlavaConfig::vision_feature_layer)
        .def_readwrite("vision_feature_select_strategy", &LlavaConfig::vision_feature_select_strategy)
        .def_readwrite("tie_word_embeddings", &LlavaConfig::tie_word_embeddings)
        .def_readwrite("vision_config", &LlavaConfig::vision_config)
        .def_readwrite("text_config", &LlavaConfig::text_config);
}

} // namespace infinilm::models::llava
