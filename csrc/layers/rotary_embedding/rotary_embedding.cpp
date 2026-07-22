#include "rotary_embedding.hpp"
#include "../../engine/compiler/piecewise_bucket_policy.hpp"
#include <algorithm> // std::clamp
#include <cmath>     // std::llround
#include <string>
#include <unordered_map>

namespace infinilm::layers::rotary_embedding {
namespace {
thread_local std::unordered_map<std::string, std::shared_ptr<infinicore::nn::RoPE>> _ROPE_DICT;

size_t rope_decode_headroom_from_env(size_t default_headroom = 512) {
    const char *raw = std::getenv("INFINI_ROPE_DECODE_HEADROOM");
    if (raw == nullptr || raw[0] == '\0') {
        return default_headroom;
    }
    return static_cast<size_t>(std::stoul(raw));
}
} // namespace

size_t get_rotary_dim(size_t head_dim, double partial_rotary_factor) {
    if (partial_rotary_factor <= 0.0 || partial_rotary_factor >= 1.0) {
        return head_dim;
    }

    size_t rotary_dim = static_cast<size_t>(std::llround(
        static_cast<double>(head_dim) * partial_rotary_factor));
    rotary_dim = std::clamp(rotary_dim, static_cast<size_t>(2), head_dim);

    // RoPE operates on complex pairs, so the rotary dimension must be even
    if (rotary_dim % 2 != 0) {
        rotary_dim -= 1;
    }
    return std::max(rotary_dim, static_cast<size_t>(2));
}

std::shared_ptr<infinicore::nn::RoPE> get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                               const infinicore::Device &device,
                                               infinicore::nn::RoPE::Algo algo) {
    // 1. Get head dimension
    size_t head_dim = model_config->get_head_dim();

    // 2. Safely get partial_rotary_factor, defaulting to 1.0 (full rotation)
    double partial_rotary_factor = model_config->get_or<double>("partial_rotary_factor", 1.0);

    // 3. Compute the actual rotary dimension
    size_t rotary_dim = get_rotary_dim(head_dim, partial_rotary_factor);

    // 4. Cache key must include rotary_dim and effective max seq to avoid stale reuse
    const std::string scaling_type = "default";

    const auto &dtype = model_config->get_dtype();
    size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    const size_t compile_max_seq =
        infinilm::engine::compile_max_seq_from_env(max_position_embeddings);
    const size_t decode_headroom = rope_decode_headroom_from_env();
    const size_t runtime_max_seq = compile_max_seq + decode_headroom;
    const size_t rope_cache_seq_len = std::max(max_position_embeddings, runtime_max_seq);
    double rope_theta = model_config->get<double>("rope_theta");

    std::string cache_key = scaling_type + "_rope_dim_" + std::to_string(rotary_dim) +
                            "_maxseq_" + std::to_string(rope_cache_seq_len);
    auto it = _ROPE_DICT.find(cache_key);
    if (it != _ROPE_DICT.end()) {
        return it->second;
    }

    auto rope = std::make_shared<infinicore::nn::RoPE>(rotary_dim, rope_cache_seq_len, rope_theta,
                                                       algo, dtype, device,
                                                       model_config->get_rope_scaling());

    _ROPE_DICT.emplace(cache_key, rope);
    return rope;
}

} // namespace infinilm::layers::rotary_embedding
