#include "gemma4_moe.hpp"

#include "../../engine/distributed/communication_group.hpp"
#include "../../execution/submit.hpp"
#include "../../layers/linear/dense.hpp"
#include "gemma4_ops.hpp"
#include <base/embedding.h>
#include <base/fill.h>
#include <base/topk_softmax.h>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace infinilm::models::gemma4 {

Gemma4TextRouter::Gemma4TextRouter(infinilm::runtime::Runtime &runtime,
                                   std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const Gemma4TextSpec &spec)
    : hidden_size_(spec.hidden_size),
      num_experts_(spec.num_experts),
      top_k_experts_(spec.top_k_experts),
      rms_norm_eps_(spec.rms_norm_eps),
      scalar_root_size_(1.0 / std::sqrt(static_cast<double>(spec.hidden_size))) {
    runtime.assertOwnerThread();
    const auto dtype = model_config->get_dtype();
    try {
        proj_ = this->bind_child(
            "proj",
            std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
                runtime, hidden_size_, num_experts_, false /* bias */, dtype));
        scale_ = infinilm::weights::Weight(runtime, {hidden_size_}, dtype);
        this->bind_weight("scale", scale_);
        per_expert_scale_ = infinilm::weights::Weight(runtime, {num_experts_}, dtype);
        this->bind_weight("per_expert_scale", per_expert_scale_);

        ones_hidden_ = ops::make_filled_param(runtime, {hidden_size_}, dtype, 1.0);
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
        }
        throw;
    }
}

void Gemma4TextRouter::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    ModelWeights::process_weights_after_loading(runtime, pre_transpose);
    scale_eff_ = ops::scale_out(runtime, scale_.tensor(), scalar_root_size_);
    auto pes_fp32 = ops::cast(runtime, per_expert_scale_.tensor(), infinilm::runtime::DataType::kFloat32);
    per_expert_scale_fp32_ = pes_fp32->contiguous(runtime)->view({num_experts_, 1});
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> Gemma4TextRouter::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states_2d) const {
    runtime.assertOwnerThread();
    const size_t ntoken = hidden_states_2d->size(0);

    // 1. Unscaled RMSNorm -> * (scale * hidden_size^-0.5)
    auto x_norm = ops::rms_norm(runtime, hidden_states_2d, ones_hidden_, rms_norm_eps_);
    auto eff = scale_eff_ ? scale_eff_ : ops::scale_out(runtime, scale_.tensor(), scalar_root_size_);
    auto x_scaled = ops::mul(runtime, x_norm, eff);

    // 2. Router projection -> [ntoken, num_experts]
    auto logits = proj_->forward(runtime, x_scaled);

    // 3. Softmax(fp32) + top-k + renorm via native TopkSoftmax
    auto scores = infinilm::runtime::Tensor::empty(runtime, {ntoken, top_k_experts_}, infinilm::runtime::DataType::kFloat32);
    auto indices = infinilm::runtime::Tensor::empty(runtime, {ntoken, top_k_experts_}, infinilm::runtime::DataType::kInt32);
    auto scratch = infinilm::runtime::Tensor::empty(runtime, {ntoken, top_k_experts_}, infinilm::runtime::DataType::kInt32);

    infini::ops::TopkSoftmaxConfig topk_config{0.0f, 0.0f};
    topk_config.set_implementation_index(
        infini::ops::TopkSoftmax::DefaultConfig(runtime.device().type()).implementation_index());
    infinilm::execution::submit<infini::ops::TopkSoftmax>(
        runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, topk_config,
        logits, std::optional<infinilm::runtime::Tensor>{},
        std::optional<infinilm::runtime::Tensor>{}, true /* renorm */, scores, indices, scratch);
    INFINILM_TRACE_OP(MoeTopkSoftmax, scores);

    // 4. Multiply by per_expert_scale[top_k_index] on GPU via Embedding + Mul
    auto pes_table = per_expert_scale_fp32_
                       ? per_expert_scale_fp32_
                       : ops::cast(runtime, per_expert_scale_.tensor(), infinilm::runtime::DataType::kFloat32)->view({num_experts_, 1});
    auto gathered_scale = infinilm::runtime::Tensor::empty(
        runtime, {ntoken, top_k_experts_, 1}, infinilm::runtime::DataType::kFloat32);
    infinilm::execution::submit<infini::ops::Embedding>(
        runtime, infini::ops::Embedding::DefaultConfig(runtime.device().type()),
        indices, pes_table, gathered_scale);
    ops::mul_inplace(runtime, scores, gathered_scale->view({ntoken, top_k_experts_}));

    return {scores, indices};
}

Gemma4TextExperts::Gemma4TextExperts(infinilm::runtime::Runtime &runtime,
                                     std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const Gemma4TextSpec &spec,
                                     const engine::distributed::RankInfo &rank_info)
    : num_experts_(spec.num_experts),
      top_k_experts_(spec.top_k_experts),
      hidden_size_(spec.hidden_size),
      moe_intermediate_size_(spec.moe_intermediate_size),
      tp_size_(rank_info.tp_size),
      tp_comm_(rank_info.comm) {
    runtime.assertOwnerThread();
    const size_t tp = static_cast<size_t>(rank_info.tp_size);
    if (moe_intermediate_size_ % tp != 0) {
        throw std::invalid_argument("Gemma4TextExperts: moe_intermediate_size must be divisible by tp_size");
    }
    moe_intermediate_per_rank_ = moe_intermediate_size_ / tp;
    const auto dtype = model_config->get_dtype();
    try {
        // TP-sharded 3D expert weights (zero-copy split from gate_up_proj in _remap_gemma4):
        //   gate_proj: [E, M/tp, H] (sharded on dim 1)
        //   up_proj:   [E, M/tp, H] (sharded on dim 1)
        //   down_proj: [E, H, M/tp] (sharded on dim 2)
        gate_proj_ = infinilm::weights::Weight(
            runtime, {num_experts_, moe_intermediate_size_, hidden_size_}, dtype,
            /*tp_dim=*/1, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("gate_proj", gate_proj_);
        up_proj_ = infinilm::weights::Weight(
            runtime, {num_experts_, moe_intermediate_size_, hidden_size_}, dtype,
            /*tp_dim=*/1, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("up_proj", up_proj_);
        down_proj_ = infinilm::weights::Weight(
            runtime, {num_experts_, hidden_size_, moe_intermediate_size_}, dtype,
            /*tp_dim=*/2, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("down_proj", down_proj_);
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
        }
        throw;
    }
}

infinilm::runtime::Tensor Gemma4TextExperts::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states_2d,
    const infinilm::runtime::Tensor &top_k_index,
    const infinilm::runtime::Tensor &top_k_weights) const {
    runtime.assertOwnerThread();
    const size_t num_tokens = hidden_states_2d->size(0);
    const auto dtype = hidden_states_2d->dtype();

    // Read small routing metadata [num_tokens, top_k] to host to dispatch only active experts.
    const size_t total_assignments = num_tokens * top_k_experts_;
    std::vector<int32_t> host_indices(total_assignments);
    std::vector<float> host_weights(total_assignments);
    if (runtime.device().type() == infinilm::runtime::Device::Type::kCpu) {
        std::memcpy(host_indices.data(), top_k_index->data(), total_assignments * sizeof(int32_t));
        std::memcpy(host_weights.data(), top_k_weights->data(), total_assignments * sizeof(float));
    } else {
        runtime.memcpyD2H(host_indices.data(), top_k_index->data(), total_assignments * sizeof(int32_t));
        runtime.memcpyD2H(host_weights.data(), top_k_weights->data(), total_assignments * sizeof(float));
        runtime.syncStream();
    }

    // Group assignments by active expert id.
    struct ExpertAssignment {
        std::vector<int32_t> token_ids;
        std::vector<float> weights;
    };
    std::unordered_map<int32_t, ExpertAssignment> by_expert;
    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t k = 0; k < top_k_experts_; ++k) {
            const int32_t eid = host_indices[t * top_k_experts_ + k];
            const float w = host_weights[t * top_k_experts_ + k];
            if (eid < 0 || static_cast<size_t>(eid) >= num_experts_) {
                continue;
            }
            auto &bucket = by_expert[eid];
            bucket.token_ids.push_back(static_cast<int32_t>(t));
            bucket.weights.push_back(w);
        }
    }

    auto final_out = ops::make_filled_tensor(runtime, {num_tokens, hidden_size_}, dtype, 0.0);
    const auto cpu_device = infinilm::runtime::Device(infinilm::runtime::Device::Type::kCpu, 0);
    infinilm::runtime::Tensor single_out_buf;

    for (const auto &[eid, bucket] : by_expert) {
        const size_t count = bucket.token_ids.size();
        infinilm::runtime::Tensor x_e;
        if (count == 1) {
            x_e = (num_tokens == 1)
                    ? hidden_states_2d
                    : hidden_states_2d->narrow({{0, static_cast<size_t>(bucket.token_ids[0]), 1}})->contiguous(runtime);
        } else {
            auto idx_host = infinilm::runtime::Tensor::from_blob(
                const_cast<int32_t *>(bucket.token_ids.data()), {count}, infinilm::runtime::DataType::kInt32, cpu_device);
            auto idx_dev = infinilm::runtime::Tensor::empty(runtime, {count}, infinilm::runtime::DataType::kInt32);
            idx_dev->copy_from(runtime, idx_host);
            x_e = infinilm::runtime::Tensor::empty(runtime, {count, hidden_size_}, dtype);
            infinilm::execution::submit<infini::ops::Embedding>(
                runtime, infini::ops::Embedding::DefaultConfig(runtime.device().type()),
                idx_dev, hidden_states_2d, x_e);
        }

        // Expert weights: gate [M/tp, H], up [M/tp, H], down [H, M/tp]
        auto w_gate = gate_proj_.tensor()->narrow({{0, static_cast<size_t>(eid), 1}})->squeeze(0);
        auto w_up = up_proj_.tensor()->narrow({{0, static_cast<size_t>(eid), 1}})->squeeze(0);
        auto w_down = down_proj_.tensor()->narrow({{0, static_cast<size_t>(eid), 1}})->squeeze(0);

        auto gate = infinilm::layers::linear::dense_projection(runtime, x_e, w_gate, std::nullopt);
        auto up = infinilm::layers::linear::dense_projection(runtime, x_e, w_up, std::nullopt);
        auto act = ops::mul(runtime, ops::gelu_tanh(runtime, gate), up);

        if (count == 1) {
            // Fold scalar routing weight directly into Gemm alpha and reuse one [1, H] scratch buffer.
            if (!single_out_buf) {
                single_out_buf = infinilm::runtime::Tensor::empty(runtime, {1, hidden_size_}, dtype);
            }
            infinilm::layers::linear::dense_projection(
                runtime, act, w_down, std::nullopt, bucket.weights[0], std::nullopt, false, single_out_buf);
            auto dst = (num_tokens == 1)
                         ? final_out
                         : final_out->narrow({{0, static_cast<size_t>(bucket.token_ids[0]), 1}});
            ops::add_inplace(runtime, dst, single_out_buf);
        } else {
            auto y_e = infinilm::layers::linear::dense_projection(runtime, act, w_down, std::nullopt);
            // Multiply each row by its routing weight on GPU, then accumulate into final_out rows.
            auto w_host = infinilm::runtime::Tensor::from_blob(
                const_cast<float *>(bucket.weights.data()), {count, 1}, infinilm::runtime::DataType::kFloat32, cpu_device);
            auto w_dev_f32 = infinilm::runtime::Tensor::empty(runtime, {count, 1}, infinilm::runtime::DataType::kFloat32);
            w_dev_f32->copy_from(runtime, w_host);
            auto w_dev = ops::cast(runtime, w_dev_f32, dtype);
            ops::mul_inplace(runtime, y_e, w_dev);
            for (size_t j = 0; j < count; ++j) {
                auto src_row = y_e->narrow({{0, j, 1}});
                auto dst_row = final_out->narrow({{0, static_cast<size_t>(bucket.token_ids[j]), 1}});
                ops::add_inplace(runtime, dst_row, src_row);
            }
        }
    }

    if (tp_size_ > 1) {
        infinilm::engine::distributed::allreduce_(runtime, final_out, final_out, infinicclSum, tp_comm_);
    }
    return final_out;
}

} // namespace infinilm::models::gemma4
