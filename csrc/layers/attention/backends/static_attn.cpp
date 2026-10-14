#include "static_attn.hpp"
#include "../../../utils.hpp"
#include "attention_layer.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/per_tensor_dequant_i8.hpp"
#include "infinicore/ops/per_tensor_quant_i8.hpp"

#include <algorithm>

namespace infinilm::layers::attention::backends {
namespace {

size_t sequence_length_from_metadata(
    const std::optional<size_t> &snapshot,
    const std::optional<infinicore::Tensor> &lengths) {
    if (snapshot.has_value()) {
        return snapshot.value();
    }

    ASSERT(lengths.has_value());
    return reinterpret_cast<const int32_t *>(
        lengths.value()
            ->to(infinicore::Device{infinicore::Device::Type::kCpu})
            ->data())[0];
}

} // namespace

StaticAttentionImpl::StaticAttentionImpl(size_t num_heads,
                                         size_t head_size,
                                         float scale,
                                         size_t num_kv_heads,
                                         size_t layer_idx)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size) {
    kv_quant_scheme_ = infinilm::global_state::get_infinilm_config().model_config->get_kv_quant_scheme();
}

infinicore::Tensor StaticAttentionImpl::forward(const AttentionLayer &layer,
                                                infinicore::Tensor &q_rope,
                                                infinicore::Tensor &k_reshaped,
                                                infinicore::Tensor &v_reshaped,
                                                infinicore::Tensor &kv_cache,
                                                const infinilm::global_state::AttentionMetadata &attn_metadata) const {

    auto k_scale = layer.get_k_scale();
    auto v_scale = layer.get_v_scale();
    if (infinilm::quantization::KVQuantAlgo::NONE != this->kv_quant_scheme_) {
        infinilm::KVQuantUtils::quantize(
            k_reshaped, v_reshaped,
            this->kv_quant_scheme_,
            k_scale,
            v_scale);
    }

    auto q_reshaped = q_rope->permute({0, 2, 1, 3});     // [bs, n_q_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]

    //  Prepare Attn
    auto shape = q_reshaped->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[2];
    size_t value_head_dim = v_reshaped->size(3);

    if (infinicore::context::isGraphRecording()) {
        throw std::runtime_error("static attention does not support graph capture");
    }

    const size_t cache_pos = sequence_length_from_metadata(
        attn_metadata.first_past_sequence_length,
        attn_metadata.past_sequence_lengths);
    const size_t total_seq_len = sequence_length_from_metadata(
        attn_metadata.first_total_sequence_length,
        attn_metadata.total_sequence_lengths);

    // update static kv cache
    // k_total:  [bs, n_kv_head, max_seq_len, head_dim]
    // v_total : [bs, n_kv_head, max_seq_len, head_dim]
    auto [k_total, v_total] = do_kv_cache_update(
        layer, k_permuted, v_permuted, kv_cache, cache_pos);

    if (infinilm::quantization::KVQuantAlgo::NONE != this->kv_quant_scheme_) {
        infinilm::KVQuantUtils::dequantize(
            k_total, v_total,
            this->kv_quant_scheme_,
            k_scale,
            v_scale,
            q_reshaped);
    }

    k_total = k_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]
    v_total = v_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]

    // Compute attention.
    size_t ngroup = num_heads_ / num_kv_heads_;
    auto Q = q_reshaped->contiguous()->view({batch_size * num_kv_heads_, ngroup * seq_len, head_dim_});
    auto K = k_total->view({batch_size * num_kv_heads_, total_seq_len, head_dim_});
    auto V = v_total->view({batch_size * num_kv_heads_, total_seq_len, value_head_dim});

    // Denglin static-attn GEMM/softmax launch fails around ~600+ query tokens
    // ("too many resources requested for launch"). Chunk Q along the sequence
    // so each causal matmul stays small; FA/paged remains M2.
    constexpr size_t kQyAttnChunkTokens = 128;
    const bool chunk_qy = q_reshaped->device().type() == infinicore::Device::Type::kQy
                       && (seq_len > kQyAttnChunkTokens || total_seq_len > 512);

    auto run_chunk = [&](size_t q_begin, size_t q_rows) -> infinicore::Tensor {
        // Queries at absolute positions [cache_pos + q_begin, ...); keys needed
        // are only [0, cache_pos + q_begin + q_rows) under causal masking.
        const size_t abs_begin = cache_pos + q_begin;
        const size_t kv_len = abs_begin + q_rows;
        auto Q_chunk = Q->narrow({{1, ngroup * q_begin, ngroup * q_rows}});
        auto K_chunk = K->narrow({{1, 0, kv_len}});
        auto V_chunk = V->narrow({{1, 0, kv_len}});
        auto K_T = K_chunk->permute({0, 2, 1});
        auto attn_weight = infinicore::op::matmul(Q_chunk, K_T, scale_);
        auto attn_weight_softmax = attn_weight->view({batch_size * num_heads_, q_rows, kv_len});
        infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);
        return infinicore::op::matmul(attn_weight, V_chunk);
    };

    infinicore::Tensor out;
    if (!chunk_qy) {
        out = run_chunk(/*q_begin=*/0, seq_len);
    } else {
        out = infinicore::Tensor::empty(
            {batch_size * num_kv_heads_, ngroup * seq_len, value_head_dim},
            Q->dtype(),
            Q->device());
        for (size_t begin = 0; begin < seq_len; begin += kQyAttnChunkTokens) {
            const size_t rows = std::min(kQyAttnChunkTokens, seq_len - begin);
            auto chunk_out = run_chunk(begin, rows);
            out->narrow({{1, ngroup * begin, ngroup * rows}})->copy_from(chunk_out);
        }
    }

    return out->view({batch_size, num_heads_, seq_len, value_head_dim})
        ->permute({0, 2, 1, 3})
        ->contiguous()
        ->view({batch_size, seq_len, num_heads_ * value_head_dim}); // [bs, seq_len, n_q_head * value_head_dim]
}

std::tuple<infinicore::Tensor, infinicore::Tensor> StaticAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                           const infinicore::Tensor key,
                                                                                           const infinicore::Tensor value,
                                                                                           infinicore::Tensor &kv_cache,
                                                                                           size_t cache_pos) const {

    auto batch_size = key->size(0);
    auto update_len = key->size(2);
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    v_cache_layer = v_cache_layer->narrow({{3, 0, value->size(3)}});

    size_t max_batch_size = k_cache_layer->size(0);
    size_t max_seq_len = k_cache_layer->size(2);
    auto device = k_cache_layer->device();

    ASSERT_EQ(batch_size, max_batch_size);

    auto result_len = cache_pos + update_len;
    ASSERT(result_len <= max_seq_len);

    auto k_cache_update = k_cache_layer->narrow({{2, cache_pos, update_len}});
    auto v_cache_update = v_cache_layer->narrow({{2, cache_pos, update_len}});

    k_cache_update->copy_from(key);
    v_cache_update->copy_from(value);

    return {k_cache_layer, v_cache_layer};
}

} // namespace infinilm::layers::attention::backends
