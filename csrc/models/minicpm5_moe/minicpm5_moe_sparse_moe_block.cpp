#include "minicpm5_moe_sparse_moe_block.hpp"

#include "../../global_state/global_state.hpp"
#include "../../global_state/piecewise_inductor_flags.hpp"
#include "minicpm5_moe_router_cpu_detail.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/inductor_segment.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace infinilm::models::minicpm5_moe {

namespace {

/// Host float32 GEMM: logits[T,E] = hs[T,H] @ W[E,H]^T (no device convert_to_f32).
void cpu_router_logits_f32(const infinicore::Tensor &hs_cpu,
                           const infinicore::Tensor &w_cpu,
                           size_t n_tokens,
                           size_t hidden_size,
                           size_t n_experts,
                           std::vector<float> &logits_out) {
    logits_out.assign(n_tokens * n_experts, 0.0f);
    for (size_t t = 0; t < n_tokens; ++t) {
        for (size_t e = 0; e < n_experts; ++e) {
            float acc = 0.0f;
            for (size_t i = 0; i < hidden_size; ++i) {
                acc += router_cpu_detail::scalar_to_f32(hs_cpu, t * hidden_size + i) *
                       router_cpu_detail::scalar_to_f32(w_cpu, e * hidden_size + i);
            }
            logits_out[t * n_experts + e] = acc;
        }
    }
}

size_t pick_moe_bucket(size_t seq_len, size_t layer_idx) {
    static const size_t kBuckets[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    if (infinicore::op::inductor_segment_impl::has_package(
            infinicore::op::PiecewiseInductorSegmentId::Moe, layer_idx, seq_len)) {
        return seq_len;
    }
    for (size_t bucket : kBuckets) {
        if (bucket >= seq_len
            && infinicore::op::inductor_segment_impl::has_package(
                   infinicore::op::PiecewiseInductorSegmentId::Moe, layer_idx, bucket)) {
            return bucket;
        }
    }
    return 0;
}

} // namespace

MiniCPM5MoeSparseMoeBlock::MiniCPM5MoeSparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t num_experts = model_config->get_or<size_t>("n_routed_experts", 0);
    if (num_experts == 0) {
        num_experts = model_config->get_or<size_t>("num_experts", 1);
    }

    INFINICORE_NN_MODULE_INIT(gate, hidden_size, num_experts, false, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(e_score_correction_bias, ({num_experts}, dtype, device, 0, 0, 1));

    experts_.reserve(num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        experts_.push_back(
            this->register_module<MiniCPM5MoeMLP>("experts." + std::to_string(i), model_config, device));
    }

    // Checkpoint under test has n_shared_experts=1 → shared intermediate == moe_intermediate_size.
    INFINICORE_NN_MODULE_INIT(shared_experts, model_config, device);
}

infinicore::op::inductor_segment_impl::MoeExternalWeightTensors
MiniCPM5MoeSparseMoeBlock::moe_external_weights() const {
    if (packed_weights_cache_.has_value()) {
        return *packed_weights_cache_;
    }

    const size_t n_experts = experts_.size();
    if (n_experts == 0) {
        throw std::runtime_error("MiniCPM5MoeSparseMoeBlock: no experts to pack");
    }
    auto gate_w = gate_->weight();
    const auto device = gate_w->device();
    const auto dtype = gate_w->dtype();
    const size_t hidden_size = gate_w->shape()[1];
    const size_t intermediate = experts_[0]->moe_intermediate_size();

    // Pack with InfiniCore view/copy (no ATen): InfiniLM is not built with ENABLE_ATEN.
    auto w_gate_up = infinicore::Tensor::empty(
        {n_experts, 2 * intermediate, hidden_size}, dtype, device);
    auto w_down = infinicore::Tensor::empty(
        {n_experts, hidden_size, intermediate}, dtype, device);

    for (size_t e = 0; e < n_experts; ++e) {
        w_gate_up->narrow({{0, e, 1}, {1, 0, intermediate}})
            ->squeeze(0)
            ->copy_from(experts_[e]->gate_weight());
        w_gate_up->narrow({{0, e, 1}, {1, intermediate, intermediate}})
            ->squeeze(0)
            ->copy_from(experts_[e]->up_weight());
        w_down->narrow({{0, e, 1}})->squeeze(0)->copy_from(experts_[e]->down_weight());
    }

    auto shared_gu = infinicore::Tensor::empty(
        {2 * intermediate, hidden_size}, dtype, device);
    shared_gu->narrow({{0, 0, intermediate}})->copy_from(shared_experts_->gate_weight());
    shared_gu->narrow({{0, intermediate, intermediate}})->copy_from(shared_experts_->up_weight());

    infinicore::op::inductor_segment_impl::MoeExternalWeightTensors packed{
        gate_w,
        e_score_correction_bias_,
        w_gate_up,
        w_down,
        shared_gu,
        shared_experts_->down_weight(),
    };
    packed_weights_cache_ = packed;

    // Drop per-expert gate/up/down storage — pack holds the live [E,…] copies.
    // Shared gate/up packed into shared_gu; down + router gate_/bias stay live.
    for (size_t e = 0; e < n_experts; ++e) {
        experts_[e]->release_weight_storage();
    }
    shared_experts_->release_gate_up_weight_storage();

    // Return idle IC pool pages to the driver so PyTorch AOTI can allocate scratch.
    infinicore::context::syncDevice();
    infinicore::context::trimDeviceMemory();

    return packed;
}

infinicore::Tensor MiniCPM5MoeSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    if (!infinilm::global_state::piecewise_inductor_segment_enabled()) {
        throw std::runtime_error(
            "MiniCPM5MoeSparseMoeBlock: INFINI_PIECEWISE_INDUCTOR_SEGMENT must be set "
            "(Track B requires AOTI MoE; CPU fallback disabled)");
    }

    const auto &shape = hidden_states->shape();
    if (shape.size() != 3) {
        throw std::runtime_error("MiniCPM5MoeSparseMoeBlock: expected hidden [B, S, H]");
    }
    const size_t seq_len = shape[1];
    const size_t bucket = pick_moe_bucket(seq_len, layer_idx_);
    if (bucket == 0
        || !infinicore::op::inductor_segment_impl::has_package(
               infinicore::op::PiecewiseInductorSegmentId::Moe, layer_idx_, bucket)) {
        throw std::runtime_error(
            "MiniCPM5MoeSparseMoeBlock: missing MoE AOTI package for layer="
            + std::to_string(layer_idx_) + " seq_len=" + std::to_string(seq_len)
            + " (compile with scripts/aot_compile_minicpm5_moe_segment.sh; "
              "CPU MoE fallback disabled)");
    }

    auto out = infinicore::Tensor::empty(
        hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    infinicore::op::inductor_moe_(hidden_states, out, layer_idx_, bucket);
    return out;
}

infinicore::Tensor MiniCPM5MoeSparseMoeBlock::forward_cpu_sparse_(
    const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hidden_size = shape[2];
    const size_t n_tokens = batch_size * seq_len;

    const auto &cfg = infinilm::global_state::get_infinilm_config().model_config;
    const size_t top_k = cfg->get<size_t>("num_experts_per_tok");
    const bool norm_topk_prob = cfg->get_or<bool>("norm_topk_prob", true);
    const float routed_scaling_factor =
        static_cast<float>(cfg->get_or<double>("routed_scaling_factor", 1.0));
    const size_t n_group = cfg->get_or<size_t>("n_group", 1);
    const size_t topk_group = cfg->get_or<size_t>("topk_group", 1);

    const size_t n_routed_experts = experts_.size();
    if (n_group == 0 || topk_group == 0 || n_routed_experts == 0 || (n_routed_experts % n_group) != 0) {
        throw std::runtime_error("MiniCPM5MoeSparseMoeBlock: invalid n_group/topk_group/n_routed_experts");
    }

    auto hs2d = hidden_states->view({n_tokens, hidden_size});

    auto hs_cpu = hs2d->to(infinicore::Device::cpu())->contiguous();
    auto w_cpu = gate_->weight()->to(infinicore::Device::cpu())->contiguous();
    auto bias_cpu = e_score_correction_bias_->to(infinicore::Device::cpu())->contiguous();

    std::vector<float> logits_host;
    cpu_router_logits_f32(hs_cpu, w_cpu, n_tokens, hidden_size, n_routed_experts, logits_host);

    auto logits_cpu =
        infinicore::Tensor::empty({n_tokens, n_routed_experts}, infinicore::DataType::F32, infinicore::Device::cpu());
    std::memcpy(logits_cpu->data(), logits_host.data(), logits_host.size() * sizeof(float));

    router_cpu_detail::RouterTopkCpuResult router_out;
    router_cpu_detail::run_router_topk_cpu(
        logits_cpu,
        bias_cpu,
        n_tokens,
        n_routed_experts,
        top_k,
        norm_topk_prob,
        routed_scaling_factor,
        n_group,
        topk_group,
        router_out);

    auto out2d_cpu =
        infinicore::Tensor::zeros({n_tokens, hidden_size}, infinicore::DataType::F32, infinicore::Device::cpu());
    float *out_flat = reinterpret_cast<float *>(out2d_cpu->data());

    for (size_t t = 0; t < n_tokens; ++t) {
        float *row_acc = out_flat + t * hidden_size;
        std::memset(row_acc, 0, hidden_size * sizeof(float));
        for (size_t j = 0; j < top_k; ++j) {
            int32_t expert_id = router_out.topk_indices[t][j];
            if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) {
                continue;
            }
            float w = router_out.topk_weights[t][j];
            if (w == 0.0f) {
                continue;
            }
            auto token_in = hs2d->narrow({{0, t, 1}});
            auto token_out = experts_.at(static_cast<size_t>(expert_id))->forward(token_in);
            auto tok_on_cpu = token_out->to(infinicore::Device::cpu())->contiguous();
            for (size_t i = 0; i < hidden_size; ++i) {
                row_acc[i] += router_cpu_detail::scalar_to_f32(tok_on_cpu, i) * w;
            }
        }
    }

    infinicore::Tensor routed;
    if (hidden_states->dtype() == infinicore::DataType::F32) {
        routed = out2d_cpu->to(hidden_states->device())->view({batch_size, seq_len, hidden_size});
    } else {
        auto routed_cpu =
            infinicore::Tensor::empty({n_tokens, hidden_size}, hidden_states->dtype(), infinicore::Device::cpu());
        const size_t numel = n_tokens * hidden_size;
        for (size_t i = 0; i < numel; ++i) {
            router_cpu_detail::write_f32_as_element(
                routed_cpu, i, router_cpu_detail::scalar_to_f32(out2d_cpu, i));
        }
        routed = routed_cpu->to(hidden_states->device())->view({batch_size, seq_len, hidden_size});
    }

    auto shared = shared_experts_->forward(hidden_states);
    return infinicore::op::add(routed, shared);
}

} // namespace infinilm::models::minicpm5_moe
