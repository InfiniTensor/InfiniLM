#pragma once

#include "../../config/model_config.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace infinilm::models::gemma4 {

struct Gemma4LayerSpec {
    size_t layer_idx{0};
    std::string layer_type;
    bool is_sliding{false};
    size_t head_dim{256};
    size_t num_kv_heads{1};
    bool is_kv_shared{false};
    size_t kv_source_layer_idx{0};
    bool use_k_eq_v{false};
    size_t mlp_intermediate_size{0};
    double rope_theta{10000.0};
    double partial_rotary_factor{1.0};
    bool is_proportional_rope{false};
};

struct Gemma4TextSpec {
    size_t vocab_size{262144};
    size_t hidden_size{1536};
    size_t num_hidden_layers{35};
    size_t num_attention_heads{8};
    size_t num_key_value_heads{1};
    size_t num_global_key_value_heads{1};
    size_t head_dim{256};
    size_t global_head_dim{512};
    size_t intermediate_size{6144};
    size_t max_position_embeddings{131072};
    size_t sliding_window{512};
    size_t num_kv_shared_layers{0};
    size_t first_kv_shared_layer_idx{35};
    bool use_double_wide_mlp{false};
    bool attention_k_eq_v{false};
    bool attention_bias{false};
    bool use_bidirectional_vision_attention{false};
    double rms_norm_eps{1e-6};
    double final_logit_softcapping{30.0};
    size_t hidden_size_per_layer_input{0};
    size_t vocab_size_per_layer_input{262144};
    int64_t pad_token_id{0};
    bool enable_moe_block{false};
    size_t num_experts{0};
    size_t top_k_experts{0};
    size_t moe_intermediate_size{0};
    double sliding_rope_theta{10000.0};
    double sliding_partial_rotary_factor{1.0};
    bool sliding_is_proportional{false};
    double full_rope_theta{1000000.0};
    double full_partial_rotary_factor{0.25};
    bool full_is_proportional{true};
    std::vector<Gemma4LayerSpec> layers;
};

Gemma4TextSpec parse_gemma4_text_spec(const std::shared_ptr<infinilm::config::ModelConfig> &model_config);

std::shared_ptr<infinilm::config::ModelConfig> prepare_gemma4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

std::shared_ptr<infinilm::config::ModelConfig> create_gemma4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

std::shared_ptr<infinilm::config::ModelConfig> create_gemma4_text_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::gemma4
