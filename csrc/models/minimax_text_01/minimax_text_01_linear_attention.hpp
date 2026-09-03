#pragma once

#include "../../layers/common_modules.hpp"
#include "../../layers/linear/fused_linear.hpp"

#include <infinicore/nn/rmsnorm.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::minimax_text_01 {

/**
 * @brief Lightning (linear) attention for MiniMax-Text-01.
 *
 * Follows the reference `modeling_minimax_text_01.py` inference path:
 *   qkv_proj -> silu -> split q/k/v -> recurrent linear attention
 *   (state update S = exp(-slope) * S + outer(k, v); output o = q @ S)
 *   -> reshape -> RMSNorm -> sigmoid(output_gate(x)) * output -> out_proj.
 *
 * The recurrent state is one [head_dim, head_dim] matrix per (batch, head),
 * kept in the shared `ssm_state_vec` cache indexed by `layer_idx` (the same
 * mechanism as the Kimi-K3 / Qwen3Next linear attention layers).
 */
class MiniMaxText01LinearAttention : public infinicore::nn::Module {
public:
    MiniMaxText01LinearAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 size_t layer_idx,
                                 const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override {
        qkv_proj_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        qkv_proj_->reset_runtime_state();
    }

    // Exposed for tests: the chunk size in use (>= 2 when the chunked prefill
    // path is active, 1 when falling back to the token-by-token recurrence).
    size_t chunk_size() const { return chunk_size_; }

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_heads_; }
    size_t local_num_heads() const { return local_num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    size_t tp_size() const { return tp_size_; }

protected:
    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    // The standard `RMSNorm` (fused, f32 precision) is used when TP == 1 to
    // keep single-GPU validation identical. When TP > 1 this module is empty
    // and the custom global RMSNorm below is used instead (norm_weight_ plus
    // an inlined var_mean + allreduce + float_power in forward), so that the
    // variance is global across TP ranks.
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, output_gate);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, out_proj);

    // Norm weight used when TP > 1 (split along the projection dimension and
    // registered under the name "norm.weight"). When TP == 1 the same-named
    // weight is held by norm_ (`nn::RMSNorm`) and this member stays empty.
    INFINICORE_NN_PARAMETER(norm_weight);

private:
    size_t layer_idx_;
    size_t num_heads_;       // total number of query heads
    size_t local_num_heads_; // query heads on this TP rank = num_heads_ / tp_size_
    size_t head_dim_;
    size_t hidden_size_;
    size_t num_hidden_layers_;
    size_t local_projection_size_; // local_num_heads_ * head_dim_

    size_t tp_size_;
    size_t tp_rank_;
    infinicclComm_t communicator_ = nullptr;

    // Per-head exponential decay exp(-slope), already scaled by layer depth,
    // shaped [1, heads, 1, 1] so it broadcasts over the [batch, heads, d, d]
    // recurrent state.
    infinicore::Tensor decay_tensor_;

    // ------------------------------------------------------------------ //
    //  Chunked prefill (coexists with the token-by-token recurrence).     //
    //                                                                     //
    //  Lightning prefill is a linear recurrence S_t = a_h*S_{t-1} + k_t^T v_t
    //  with a per-head scalar decay a_h.  Prefill over a long sequence is a
    //  serial chain of tiny per-token kernels, so forward() splits the tokens
    //  into blocks of chunk_size_ and applies the exact HF-style block scan:
    //  within a block the pairwise decayed attention is computed densely
    //  (parallel over the block), while only the block-to-block carry is
    //  serial.  Mathematically it is identical to the recurrence; only the
    //  floating-point accumulation order differs.  decode always keeps the
    //  single-token recurrence path below.
    //
    //  Enabled by config `lightning_chunk_size` (default 64, >=2).  Setting it
    //  to 0/1 (or env INFINILM_LIGHTNING_CHUNK=0) keeps the legacy
    //  token-by-token path, so both implementations coexist and can be
    //  A/B-compared.  All tables below are per-head constants derived from the
    //  same a_h = exp(-slope_h * layer_scale) as decay_tensor_ and are built
    //  once in the ctor (host side, no runtime exp()).
    size_t chunk_size_ = 1; // 1 => chunked path disabled

    // a_h^k for k = 0..chunk_size_, shaped [1, heads, C+1, 1].
    infinicore::Tensor chunk_pow_;
    // Cross-block query scale a_h^{pos+1}, shaped [1, heads, C, 1].
    infinicore::Tensor chunk_qdec_;
    // Intra-block key scale a_h^{C-1-pos}, shaped [1, heads, C, 1]
    // (sliced from the tail so an m-token block gets a_h^{m-1-pos}).
    infinicore::Tensor chunk_kdec_;
    // Intra-block causal decay a_h^{i-j} (i>=j) else 0, shaped [1, heads, C, C].
    infinicore::Tensor chunk_diag_;

    // Token-by-token recurrence shared by decode and the non-chunked prefill.
    // Runs over q/k/v shaped [batch, seq, local_heads, head_dim]; `state` is
    // updated in place to the final recurrent state and the pre-norm output
    // [batch, seq, local_projection] is returned.
    infinicore::Tensor forward_recurrent_(
        const infinicore::Tensor &q_heads,
        const infinicore::Tensor &k_heads,
        const infinicore::Tensor &v_heads,
        const infinicore::Tensor &decay,
        infinicore::Tensor &state) const;

    // Chunked prefill scan over q/k/v shaped [batch, seq, local_heads, head_dim].
    // `state` must be the (zero-initialised) carry-in; it is updated to the
    // exact post-sequence recurrent state, matching forward_recurrent_.
    infinicore::Tensor forward_chunked_(
        const infinicore::Tensor &q_heads,
        const infinicore::Tensor &k_heads,
        const infinicore::Tensor &v_heads,
        infinicore::Tensor &state) const;
};

} // namespace infinilm::models::minimax_text_01
