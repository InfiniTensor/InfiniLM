#include "qwen3_5_fused_qkv_linear.hpp"

#include "runtime/runtime.hpp"
#include <exception>
#include <limits>
#include <stdexcept>

namespace infinilm::models::qwen3_5 {

namespace {
size_t checked_fused_output(infinilm::runtime::Runtime &runtime, size_t hidden, size_t dim,
                            size_t q_heads, size_t kv_heads,
                            const infinilm::runtime::DataType &dtype,
                            const engine::distributed::RankInfo &rank) {
    // Same pre-allocation rank/eager/geometry contract as existing typed QKV.
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Fused Linear construction requires eager Runtime");
    if (rank.device != runtime.device() || rank.tp_size <= 0 || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size)
        throw std::invalid_argument("Invalid explicit fused Linear rank/device");
    const size_t tp = static_cast<size_t>(rank.tp_size);
    if (!hidden || !dim || !q_heads || !kv_heads || q_heads % tp || (kv_heads % tp && tp % kv_heads))
        throw std::invalid_argument("Invalid explicit Qwen35 fused QKV geometry");
    const size_t replicas = kv_heads % tp == 0 ? 1 : tp / kv_heads;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (q_heads > limit / dim || kv_heads > limit / dim || kv_heads * dim > limit / replicas)
        throw std::length_error("Explicit Qwen35 fused QKV size overflow");
    const size_t q = q_heads * dim, kv = kv_heads * dim * replicas;
    if (q > limit / 2 || kv > limit / 2 || 2 * q > limit - 2 * kv)
        throw std::length_error("Explicit Qwen35 fused QKV size overflow");
    const size_t output = 2 * q + 2 * kv;
    if (hidden > limit / output || output * hidden > limit / infinilm::runtime::dsize(dtype))
        throw std::length_error("Explicit Qwen35 fused QKV storage size overflow");
    return output;
}
} // namespace

Qwen35FusedQKVLinear::Qwen35FusedQKVLinear(infinilm::runtime::Runtime &runtime, size_t hidden_size,
                                           size_t head_dim,
                                           size_t num_q_head,
                                           size_t num_kv_head,
                                           const std::string &q_name,
                                           const std::string &k_name,
                                           const std::string &v_name,
                                           infinilm::layers::linear::RegisterParamFn register_fn,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           bool bias,
                                           const infinilm::runtime::DataType &dtype,
                                           engine::distributed::RankInfo rank_info)
    : infinilm::layers::linear::ColumnParallelLinear(
        runtime, hidden_size,
        checked_fused_output(runtime, hidden_size, head_dim, num_q_head, num_kv_head, dtype, rank_info),
        quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
        bias,
        dtype,
        rank_info.tp_rank,
        rank_info.tp_size),
      head_dim_(head_dim),
      local_num_q_heads_(num_q_head / tp_size_),
      q_proj_out_size_(num_q_head * head_dim * 2 / tp_size_),
      q_out_size_(num_q_head * head_dim / tp_size_),
      k_out_size_(calculate_kv_replicas(num_kv_head, rank_info.tp_size) * num_kv_head * head_dim / tp_size_),
      v_out_size_(calculate_kv_replicas(num_kv_head, rank_info.tp_size) * num_kv_head * head_dim / tp_size_),
      num_kv_head_(num_kv_head),
      register_fn_(register_fn) {
    // Hold split view objects outside the try until catch-side completion.
    std::vector<infinilm::quantization::SplitParam> params;
    try {
    split_infos_ = {
        {q_name, 0, q_proj_out_size_, 0},
        {k_name, q_proj_out_size_, k_out_size_, num_kv_head_},
        {v_name, q_proj_out_size_ + k_out_size_, v_out_size_, num_kv_head_},
    };
        params = this->split_params(split_infos_, tp_rank_, tp_size_, num_kv_head_);
        for (const auto &sp : params) {
            register_fn_(sp.full_name, sp.param);
        }
    } catch (...) {
        const auto failure = std::current_exception();
        // Body catch runs before members/base and prepared views are destroyed.
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor, infinilm::runtime::Tensor, infinilm::runtime::Tensor>
Qwen35FusedQKVLinear::forward_split(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_ || !input || input->device() != runtime.device())
        throw std::invalid_argument("Qwen35 fused forward Runtime/input devices differ");
    if (input->shape().size() != 3 || input->size(2) != in_features_)
        throw std::invalid_argument("Qwen35 fused forward expects [batch, sequence, hidden_size]");
    auto output = this->forward(runtime, input);
    auto shape = output->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto q_proj_out = output->narrow({{2, 0, q_proj_out_size_}});
    auto q_proj_heads = q_proj_out->view({batch_size, seq_len, local_num_q_heads_, head_dim_ * 2});
    auto q_out = q_proj_heads->narrow({{3, 0, head_dim_}});
    auto gate_out = q_proj_heads->narrow({{3, head_dim_, head_dim_}});
    auto k_out = output->narrow({{2, q_proj_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_proj_out_size_ + k_out_size_, v_out_size_}});

    return std::make_tuple(q_out, gate_out, k_out, v_out);
}

void Qwen35FusedQKVLinear::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_)
        throw std::invalid_argument("postload Runtime and Linear devices differ");
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("Qwen35 fused postload requires inactive graph state");

    // Match QKV/GateUp: retain old bindings and every prepared split view
    // across conversion, callback publication and catch-side owner drain.
    auto old_parameters = local_weights_;
    auto old_bound = bound_params_;
    auto old_quant = quantization_;
    std::vector<infinilm::quantization::SplitParam> params;
    try {
        BaseLinear::process_weights_after_loading(runtime, pre_transpose);
        refresh_logical_aliases_(params);
        // The worker's outer fence is after return. Prove completion here
        // while old storage and new views are still strongly owned.
        runtime.syncStream();
    } catch (...) {
        const auto failure = std::current_exception();
        // Callbacks can publish some aliases before throwing. The worker is
        // Failed/unusable; this is local lifetime safety, not tree rollback.
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

void Qwen35FusedQKVLinear::refresh_logical_aliases_(std::vector<infinilm::quantization::SplitParam> &params) {
    if (register_fn_ && !split_infos_.empty()) {
        // Quantization owns the transformed split axis; keep original Q/Gate
        // interleave, KV shard/replica metadata and bias slicing unchanged.
        params = this->split_params(split_infos_, tp_rank_, tp_size_, num_kv_head_);
        for (const auto &sp : params) {
            register_fn_(sp.full_name, sp.param);
        }
    }
}

} // namespace infinilm::models::qwen3_5
