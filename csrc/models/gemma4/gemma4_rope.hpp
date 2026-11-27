#pragma once

#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cstddef>
#include <memory>

namespace infinilm::models::gemma4 {

// Gemma 4 Rotary Position Embedding supporting both:
// 1. Standard (default) RoPE for sliding_attention layers (head_dim=256, theta=10000).
// 2. Proportional RoPE for full_attention layers (head_dim=512, theta=1000000, partial_rotary_factor=0.25):
//    first rope_angles = floor(partial_rotary_factor * head_dim / 2) pairs rotate with
//    inv_freq[j] = 1 / theta^(2*j / head_dim), and remaining pairs have inv_freq = 0 (cos=1, sin=0).
// Uses GPT_NEOX (rotate_half) pairing matching HuggingFace modeling_gemma4.py.
class Gemma4RoPE {
public:
    Gemma4RoPE(infinilm::runtime::Runtime &runtime,
               size_t head_dim,
               size_t max_seq_len,
               double theta,
               double partial_rotary_factor,
               bool is_proportional,
               infinilm::runtime::DataType dtype);

    // Applies RoPE to x of shape [..., head_dim] (e.g. [batch, seq, heads, head_dim] or [tokens, heads, head_dim])
    // with positions of shape [seq] or [batch, seq].
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x,
                                      const infinilm::runtime::Tensor &positions,
                                      bool in_place = false) const;

    size_t head_dim() const { return head_dim_; }
    size_t max_seq_len() const { return max_seq_len_; }

private:
    infinilm::runtime::Device device_;
    std::shared_ptr<infinilm::runtime::Runtime> runtime_owner_;
    size_t head_dim_{0};
    size_t max_seq_len_{0};
    infinilm::runtime::DataType dtype_{infinilm::runtime::DataType::kBFloat16};
    infinilm::runtime::Tensor cos_sin_cache_; // [max_seq_len, head_dim] on GPU
    infinilm::runtime::Tensor cos_cache_;     // [max_seq_len, head_dim/2]
    infinilm::runtime::Tensor sin_cache_;     // [max_seq_len, head_dim/2]
};

} // namespace infinilm::models::gemma4
