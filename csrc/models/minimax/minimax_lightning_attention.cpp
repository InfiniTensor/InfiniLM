#include "minimax_lightning_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/context/context.hpp"
#include "../../utils.hpp"

#include <infinicore/ops/lightning_attention.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/sigmoid.hpp>
#include <infinicore/ops/silu.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace infinilm::models::minimax {

std::vector<float> MiniMaxLightningAttention::build_slopes(size_t num_heads) {
    // Same construction as HF transformers `minimax` (MiniMaxLightningAttention):
    //   base = 1 / 2^(8/H); rate[h] = base^(h+1)
    const float base = 1.0f / std::pow(2.0f, 8.0f / static_cast<float>(num_heads));
    std::vector<float> slopes(num_heads);
    for (size_t h = 0; h < num_heads; ++h) {
        slopes[h] = std::pow(base, static_cast<float>(h + 1));
    }
    return slopes;
}

MiniMaxLightningAttention::MiniMaxLightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                     size_t layer_idx,
                                                     const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    head_dim_ = model_config->get_or<size_t>("head_dim", 0);
    if (head_dim_ == 0) {
        head_dim_ = hidden_size / total_num_heads;
    }
    block_size_ = model_config->get_or<size_t>("block", 256);
    const std::string hidden_act = model_config->get_or<std::string>("hidden_act", "silu");
    if (hidden_act != "silu") {
        throw std::runtime_error("MiniMaxLightningAttention: unsupported hidden_act '" + hidden_act + "'");
    }
    silu_act_ = true;
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get_or<double>("linear_rms_norm_eps", 1e-6);

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    num_heads_ = total_num_heads / rank_info.tp_size;
    inner_dim_ = num_heads_ * head_dim_;

    qkv_proj_ = this->register_module<layers::linear::ColumnParallelLinear>(
        "qkv_proj", hidden_size, inner_dim_ * 3, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    output_gate_ = this->register_module<layers::linear::ColumnParallelLinear>(
        "output_gate", hidden_size, inner_dim_, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    out_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "out_proj", inner_dim_, hidden_size, model_config->get_quantization_method(), false, dtype, device,
        rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
    INFINICORE_NN_MODULE_INIT(norm, inner_dim_, rms_norm_eps, dtype, device);

    // Per-head ALiBi slope, decayed by layer index (same as MiniMax-01 / vLLM).
    const std::vector<float> slopes = build_slopes(total_num_heads);
    float layer_scale = 1.0f;
    if (num_hidden_layers > 1) {
        // transformers MiniMax: factor = 1 - layer_idx/(num_hidden_layers - 1 + 1e-5) + 1e-5
        layer_scale = 1.0f - static_cast<float>(layer_idx_) / (static_cast<float>(num_hidden_layers - 1) + 1e-5f) + 1e-5f;
    }
    auto slope_cpu = infinicore::Tensor::empty({total_num_heads}, infinicore::DataType::F32, infinicore::Device::cpu());
    auto *slope_data = reinterpret_cast<float *>(slope_cpu->data());
    auto ratio_cpu = infinicore::Tensor::empty({total_num_heads}, infinicore::DataType::F32, infinicore::Device::cpu());
    auto *ratio_data = reinterpret_cast<float *>(ratio_cpu->data());
    for (size_t i = 0; i < total_num_heads; ++i) {
        slope_data[i] = slopes[i] * layer_scale;
        ratio_data[i] = std::exp(-slope_data[i]);
    }
    slope_ = slope_cpu->to(device);
    ratio_ = ratio_cpu->to(device);
    if (rank_info.tp_size > 1) {
        slope_ = slope_->narrow({{0, static_cast<size_t>(rank_info.tp_rank) * num_heads_, num_heads_}});
        ratio_ = ratio_->narrow({{0, static_cast<size_t>(rank_info.tp_rank) * num_heads_, num_heads_}});
    }
}

namespace {
// Build a one-element int32 index tensor for a single-request op call.
infinicore::Tensor make_index_tensor(size_t value, const infinicore::Device &device) {
    auto cpu = infinicore::Tensor::empty({1}, infinicore::DataType::I32, infinicore::Device::cpu());
    reinterpret_cast<int32_t *>(cpu->data())[0] = static_cast<int32_t>(value);
    return cpu->to(device);
}
} // namespace

infinicore::Tensor MiniMaxLightningAttention::forward(const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    auto hidden_mutable = hidden_states;
    auto qkv = infinicore::op::silu(qkv_proj_->forward(hidden_mutable)); // [B, T, 3 * inner]
    auto qkv4 = qkv->view({batch_size, seq_len, num_heads_, 3 * head_dim_});
    auto q = qkv4->narrow({{3, 0, head_dim_}});       // [B, T, H, D]
    auto k = qkv4->narrow({{3, head_dim_, head_dim_}});
    auto v = qkv4->narrow({{3, 2 * head_dim_, head_dim_}});

    auto &forward_context = infinilm::global_state::get_forward_context();
    const auto &mamba_metadata = forward_context.mamba_metadata;
    if (!mamba_metadata.input_offsets.has_value() ||
        !mamba_metadata.init_state_indices.has_value() ||
        !mamba_metadata.final_state_indices.has_value()) {
        throw std::runtime_error("MiniMaxLightningAttention: linear attention requires mamba state indices");
    }
    if (forward_context.ssm_state_vec.size() <= layer_idx_ || !forward_context.ssm_state_vec[layer_idx_]) {
        throw std::runtime_error("MiniMaxLightningAttention: lightning state cache is not allocated for layer " + std::to_string(layer_idx_));
    }
    auto state_pool = forward_context.ssm_state_vec[layer_idx_];
    const auto &init_indices = mamba_metadata.init_state_indices.value();
    const auto &final_indices = mamba_metadata.final_state_indices.value();

    const bool is_decode = mamba_metadata.input_offsets.value()->shape()[0] - 1 == seq_len;
    auto attn_out = infinicore::Tensor::empty(
        {batch_size, seq_len, num_heads_, head_dim_}, state_pool->dtype(), state_pool->device());
    if (is_decode) {
        // Batched decode: one token per request, one op call.
        infinicore::op::lightning_attention_(
            attn_out, state_pool, q, k, v, slope_, init_indices, final_indices);
    } else {
        // Prefill: requests may have different lengths, so run one op call per request.
        // The offsets/indices may live on an accelerator; copy them to the host
        // first (the per-request loop itself is device-agnostic).
        auto cpu_offsets = mamba_metadata.input_offsets.value();
        auto cpu_init = init_indices;
        auto cpu_final = final_indices;
        if (cpu_offsets->device().getType() != infinicore::Device::Type::CPU) {
            cpu_offsets = cpu_offsets->to(infinicore::Device::cpu());
        }
        if (cpu_init->device().getType() != infinicore::Device::Type::CPU) {
            cpu_init = cpu_init->to(infinicore::Device::cpu());
        }
        if (cpu_final->device().getType() != infinicore::Device::Type::CPU) {
            cpu_final = cpu_final->to(infinicore::Device::cpu());
        }
        infinicore::context::syncStream();
        auto read_index = [](const infinicore::Tensor &t, size_t i) -> size_t {
            if (t->dtype() == infinicore::DataType::I32) {
                return static_cast<size_t>(reinterpret_cast<const int32_t *>(t->data())[i]);
            }
            return static_cast<size_t>(reinterpret_cast<const int64_t *>(t->data())[i]);
        };
        const auto *offsets_ptr = reinterpret_cast<const int32_t *>(cpu_offsets->data());
        const size_t num_requests = cpu_offsets->shape()[0] - 1;
        for (size_t r = 0; r < num_requests; ++r) {
            const size_t start = static_cast<size_t>(offsets_ptr[r]);
            const size_t end = static_cast<size_t>(offsets_ptr[r + 1]);
            const size_t len = end - start;
            if (len == 0) {
                continue;
            }
            auto q_r = q->narrow({{0, 0, 1}})->narrow({{1, start, len}});
            auto k_r = k->narrow({{0, 0, 1}})->narrow({{1, start, len}});
            auto v_r = v->narrow({{0, 0, 1}})->narrow({{1, start, len}});
            auto out_r = attn_out->narrow({{0, 0, 1}})->narrow({{1, start, len}});
            auto init_r = make_index_tensor(read_index(cpu_init, r), state_pool->device());
            auto final_r = make_index_tensor(read_index(cpu_final, r), state_pool->device());
            infinicore::op::lightning_attention_(out_r, state_pool, q_r, k_r, v_r, slope_, init_r, final_r);
        }
    }

    auto attn_flat = attn_out->view({batch_size, seq_len, inner_dim_});
    auto normed = norm_->forward(attn_flat);
    auto gate = infinicore::op::sigmoid(output_gate_->forward(hidden_mutable));
    auto gated = infinicore::op::mul(normed, gate);
    return out_proj_->forward(gated);
}

} // namespace infinilm::models::minimax



