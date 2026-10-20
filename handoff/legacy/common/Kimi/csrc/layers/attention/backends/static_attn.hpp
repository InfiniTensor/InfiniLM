#pragma once

#include "../../../global_state/forward_context.hpp"
#include "../../../layers/quantization/kv_quant.hpp"
#include "../../../weights/weights.hpp"
#include "runtime/tensor.hpp"
#include <tuple>
#include <array>
#include <cstdint>
#include <vector>

namespace infinilm::layers::attention {
class AttentionLayer;
}
namespace infinilm::runtime { class Runtime; }

namespace infinilm::layers::attention::backends {

class StaticAttentionImpl {
public:
    StaticAttentionImpl(size_t num_heads,
                        size_t head_size,
                        float scale,
                        size_t num_kv_heads,
                        size_t layer_idx,
                        infinilm::quantization::KVQuantAlgo kv_quant_scheme);

    infinilm::runtime::Tensor forward(const AttentionLayer &layer,
                               infinilm::runtime::Tensor &q_reshaped, // query
                               infinilm::runtime::Tensor &k_permuted, // key
                               infinilm::runtime::Tensor &v_permuted, // value
                               infinilm::runtime::Tensor &kv_cache,
                               const infinilm::global_state::AttentionMetadata &attn_metadata,
                               infinilm::runtime::Runtime &runtime) const;

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> do_kv_cache_update(const AttentionLayer &layer,
                                                                          const infinilm::runtime::Tensor key,
                                                                          const infinilm::runtime::Tensor value,
                                                                          infinilm::runtime::Tensor &kv_cache,
                                                                          size_t cache_pos,
                                                                          infinilm::runtime::Runtime &runtime) const;

private:
    infinilm::runtime::Tensor forward_decode_(
        const infinilm::runtime::Tensor &query,
        const infinilm::runtime::Tensor &key,
        const infinilm::runtime::Tensor &value,
        infinilm::runtime::Tensor &kv_cache,
        const infinilm::global_state::AttentionMetadata &attn_metadata,
        infinilm::runtime::Runtime &runtime, const std::vector<int64_t> &window_size) const;

    // One bounded, metadata-only algorithm choice per layer. No Tensor/storage
    // or Ops instance is retained here; provider instances remain owned by Ops
    // (eager) or the recorded call (graph). A new layout replaces the choice.
    struct DecodeChoice {
        std::array<infinilm::runtime::TensorMetaData, 7> layouts;
        const infinilm::runtime::Runtime *runtime;
        std::vector<int64_t> window_size;
        bool fused;
    };
    mutable std::optional<DecodeChoice> decode_choice_;
    bool use_cache_decode_(const infinilm::runtime::Tensor &query,
                           const infinilm::runtime::Tensor &key,
                           const infinilm::runtime::Tensor &value,
                           const infinilm::runtime::Tensor &kv_cache,
                           const infinilm::global_state::AttentionMetadata &metadata,
                           infinilm::runtime::Runtime &runtime, const std::vector<int64_t> &window_size) const;

    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t layer_idx_;
    size_t head_dim_; // Note: head_dim equals to head_size

    infinilm::quantization::KVQuantAlgo kv_quant_scheme_;
};
} // namespace infinilm::layers::attention::backends
