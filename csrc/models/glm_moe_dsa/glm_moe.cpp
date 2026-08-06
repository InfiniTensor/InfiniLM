#include "glm_moe.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/dynamic_scaled_int8_quant.hpp"
#include "infinicore/ops/grouped_topk_vendor.hpp"
#include "infinicore/ops/moe_argsort_bincount.hpp"
#include "infinicore/ops/moe_expand_input.hpp"
#include "infinicore/ops/moe_silu_and_mul_quant.hpp"
#include "infinicore/ops/moe_sum_vendor.hpp"
#include "infinicore/ops/moe_w4a8_marlin.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/w4a8_group_gemm.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace infinilm::models::glm_moe_dsa {
namespace {
struct MoeEnvironment {
    bool debug_dump;
    bool debug_dump_all_ranks;
    bool debug_dump_has_layer;
    size_t debug_dump_layer;
    bool use_hygon_marlin;
};

const MoeEnvironment kMoeEnvironment = [] {
    const char *dump_layer = std::getenv("INFINILM_GLM_DEBUG_DUMP_LAYER");
    return MoeEnvironment{
        std::getenv("INFINILM_GLM_DEBUG_DUMP") != nullptr,
        std::getenv("INFINILM_GLM_DEBUG_DUMP_ALL_RANKS") != nullptr,
        dump_layer != nullptr,
        dump_layer == nullptr
            ? 0
            : static_cast<size_t>(std::strtoull(dump_layer, nullptr, 10)),
        std::getenv("INFINILM_HYGON_MARLIN_MOE") != nullptr,
    };
}();

void debug_dump_moe(const infinicore::Tensor &tensor,
                    const std::string &name,
                    size_t layer_idx) {
    if (!kMoeEnvironment.debug_dump) {
        return;
    }
    if (kMoeEnvironment.debug_dump_has_layer
        && layer_idx != kMoeEnvironment.debug_dump_layer) {
        return;
    }
    const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (!kMoeEnvironment.debug_dump_all_ranks && rank.global_rank != 0) {
        return;
    }
    const std::string rank_prefix = kMoeEnvironment.debug_dump_all_ranks
                                      ? "rank_" + std::to_string(rank.global_rank) + "_"
                                      : "";
    tensor->debug(
        "/tmp/glmmoe_" + rank_prefix + "layer_"
        + std::to_string(layer_idx) + "_" + name + ".bin");
}
} // namespace

GlmTopKRouter::GlmTopKRouter(std::shared_ptr<infinilm::config::ModelConfig> c, const infinicore::Device &d) {
    auto h = c->get<size_t>("hidden_size");
    num_experts_ = c->get<size_t>("num_experts");
    top_k_ = c->get<size_t>("num_experts_per_tok");
    num_expert_group_ = c->get_or_alias<size_t>("num_expert_group", "n_group", 1);
    topk_group_ = c->get_or<size_t>("topk_group", 1);
    renormalize_ = c->get_or<bool>("norm_topk_prob", true);
    routed_scaling_factor_ = c->get_or<float>("routed_scaling_factor", 1);
    if (!num_experts_ || !top_k_ || top_k_ > num_experts_) {
        throw std::runtime_error("GlmTopKRouter: invalid config");
    }
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, h}, c->get_dtype(), d));
    INFINICORE_NN_PARAMETER_INIT(e_score_correction_bias, ({num_experts_}, infinicore::DataType::F32, d));
}
void GlmTopKRouter::process_weights_after_loading() {
    const auto runtime_bias_dtype = weight_->device().getType() == infinicore::Device::Type::HYGON
                                      ? infinicore::DataType::F32
                                      : weight_->dtype();
    runtime_bias_ = infinicore::Tensor::empty({num_experts_}, runtime_bias_dtype, weight_->device());
    infinicore::op::cast_(runtime_bias_, e_score_correction_bias_);
}
std::tuple<infinicore::Tensor, infinicore::Tensor> GlmTopKRouter::forward(const infinicore::Tensor &x) const {
    auto logits = infinicore::op::linear(x, weight_, std::nullopt, 1);
    auto w = infinicore::Tensor::empty({x->size(0), top_k_}, infinicore::DataType::F32, x->device());
    auto ids = infinicore::Tensor::empty({x->size(0), top_k_}, infinicore::DataType::I32, x->device());
    infinicore::op::grouped_topk_vendor_(w, ids, logits, num_expert_group_, topk_group_, renormalize_, routed_scaling_factor_, runtime_bias_, "sigmoid");

    return {w, ids};
}
GlmW4A8Experts::GlmW4A8Experts(std::shared_ptr<infinilm::config::ModelConfig> c, const infinicore::Device &d) {
    hidden_ = c->get<size_t>("hidden_size");
    nexpert_ = c->get<size_t>("num_experts");
    topk_ = c->get<size_t>("num_experts_per_tok");
    auto &r = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_ = r.tp_size;
    comm_ = r.comm;
    auto full_i = c->get<size_t>("moe_intermediate_size");
    inter_ = full_i / tp_;
    size_t pi = inter_ / 2;
    w1_ = infinicore::Tensor::empty({nexpert_, inter_ * 2, hidden_ / 2}, infinicore::DataType::I8, d);
    s1_ = infinicore::Tensor::empty({nexpert_, inter_ * 2, 1}, infinicore::DataType::F32, d);
    w2_ = infinicore::Tensor::empty({nexpert_, hidden_, pi}, infinicore::DataType::I8, d);
    s2_ = infinicore::Tensor::empty({nexpert_, hidden_, 1}, infinicore::DataType::F32, d);
    for (size_t i = 0; i < nexpert_; ++i) {
        auto p = std::to_string(i);
        auto gw = w1_->narrow({{0, i, 1}, {1, 0, inter_}, {2, 0, hidden_ / 2}})->view({inter_, hidden_ / 2});
        auto uw = w1_->narrow({{0, i, 1}, {1, inter_, inter_}, {2, 0, hidden_ / 2}})->view({inter_, hidden_ / 2});
        auto dw = w2_->narrow({{0, i, 1}, {1, 0, hidden_}, {2, 0, pi}})->view({hidden_, pi});
        auto gs = s1_->narrow({{0, i, 1}, {1, 0, inter_}, {2, 0, 1}})->view({1, inter_});
        auto us = s1_->narrow({{0, i, 1}, {1, inter_, inter_}, {2, 0, 1}})->view({1, inter_});
        auto ds = s2_->narrow({{0, i, 1}, {1, 0, hidden_}, {2, 0, 1}})->view({1, hidden_});
        register_parameter(p + ".gate_proj.weight", infinicore::nn::Parameter(gw, 0, r.tp_rank, r.tp_size));
        register_parameter(p + ".up_proj.weight", infinicore::nn::Parameter(uw, 0, r.tp_rank, r.tp_size));
        register_parameter(p + ".down_proj.weight", infinicore::nn::Parameter(dw, 1, r.tp_rank, r.tp_size));
        register_parameter(p + ".gate_proj.weight_scale", infinicore::nn::Parameter(gs, 1, r.tp_rank, r.tp_size));
        register_parameter(p + ".up_proj.weight_scale", infinicore::nn::Parameter(us, 1, r.tp_rank, r.tp_size));
        register_parameter(p + ".down_proj.weight_scale", infinicore::nn::Parameter(ds));
    }
}

void GlmW4A8Experts::process_weights_after_loading() {
    if (!w1_ || w1_->device().getType() != infinicore::Device::Type::HYGON
        || !kMoeEnvironment.use_hygon_marlin) {
        return;
    }
    auto repacked = infinicore::Tensor::empty(
        w1_->shape(), w1_->dtype(), w1_->device());
    infinicore::op::prepare_w4a8_marlin_weight_(repacked, w1_);
    w1_->copy_from(repacked);
    infinicore::context::syncStream();

    repacked = infinicore::Tensor::empty(
        w2_->shape(), w2_->dtype(), w2_->device());
    infinicore::op::prepare_w4a8_marlin_weight_(repacked, w2_);
    w2_->copy_from(repacked);

    // LightOp Marlin applies an internal int4 scale factor of 16, while GLM
    // checkpoint scales use 18. Keep separate scales so the fallback remains
    // bit-for-bit unchanged when the experimental path is disabled.
    marlin_s1_ = infinicore::Tensor::empty(s1_->shape(), s1_->dtype(), s1_->device());
    marlin_s2_ = infinicore::Tensor::empty(s2_->shape(), s2_->dtype(), s2_->device());
    infinicore::op::mul_scalar_(marlin_s1_, s1_, 18.0 / 16.0);
    infinicore::op::mul_scalar_(marlin_s2_, s2_, 18.0 / 16.0);
    infinicore::context::syncStream();
}

infinicore::Tensor GlmW4A8Experts::forward(const infinicore::Tensor &x,
                                           const infinicore::Tensor &ids,
                                           const infinicore::Tensor &tw,
                                           std::optional<infinicore::Tensor> shared_output,
                                           size_t layer_idx) const {
    if (!w1_) {
        throw std::runtime_error("GlmW4A8Experts: weights not ready");
    }
    size_t m = x->size(0), total = m * topk_;
    auto &metadata = infinilm::global_state::get_forward_context().attn_metadata;
    if (!metadata.total_sequence_lengths.has_value()) {
        throw std::runtime_error("GlmW4A8Experts: missing sequence metadata");
    }
    const bool dec = m == metadata.total_sequence_lengths.value()->numel();
    const bool use_packed_decode = dec && x->device().getType() != infinicore::Device::Type::HYGON;
    int64_t fmt = use_packed_decode ? 2 : 1;
    auto cnt = infinicore::Tensor::empty({nexpert_}, infinicore::DataType::I32, x->device()), sorted = infinicore::Tensor::empty({total}, infinicore::DataType::I32, x->device()), inv = infinicore::Tensor::empty({total}, infinicore::DataType::I32, x->device());
    infinicore::op::moe_argsort_bincount_with_inv_pos_(cnt, sorted, inv, ids, nexpert_);
    debug_dump_moe(cnt, "counts", layer_idx);
    debug_dump_moe(sorted, "sorted", layer_idx);
    debug_dump_moe(inv, "inverse", layer_idx);

    if (x->device().getType() == infinicore::Device::Type::HYGON
        && kMoeEnvironment.use_hygon_marlin) {
        auto padded_sorted = infinicore::Tensor::empty(
            {total * 16}, infinicore::DataType::I32, x->device());
        auto expert_ids = infinicore::Tensor::empty(
            {total}, infinicore::DataType::I32, x->device());
        auto num_tokens_post_pad = infinicore::Tensor::empty(
            {1}, infinicore::DataType::I32, x->device());
        infinicore::op::moe_align_block_size_from_counts_(
            padded_sorted, expert_ids, num_tokens_post_pad,
            sorted, cnt, 16, topk_);

        auto a1 = infinicore::Tensor::empty(
            {m, hidden_}, infinicore::DataType::I8, x->device());
        auto a1s = infinicore::Tensor::empty(
            {m, 1}, infinicore::DataType::F32, x->device());
        infinicore::op::dynamic_scaled_int8_quant_(a1, x, a1s);
        debug_dump_moe(a1, "a1_marlin", layer_idx);
        debug_dump_moe(a1s, "a1_scale_marlin", layer_idx);

        auto a2 = infinicore::Tensor::empty(
            {total, inter_ * 2}, x->dtype(), x->device());
        infinicore::op::moe_w4a8_marlin_(
            a2, a1, w1_, a1s, marlin_s1_, std::nullopt,
            padded_sorted, expert_ids, num_tokens_post_pad,
            topk_, topk_);
        debug_dump_moe(a2, "a2_marlin", layer_idx);

        auto a2q = infinicore::Tensor::empty(
            {total, inter_}, infinicore::DataType::I8, x->device());
        auto a2s = infinicore::Tensor::empty(
            {total, 1}, infinicore::DataType::F32, x->device());
        infinicore::op::moe_silu_and_mul_quant_(a2q, a2s, a2, 1);
        debug_dump_moe(a2q, "a2_quant_marlin", layer_idx);
        debug_dump_moe(a2s, "a2_scale_marlin", layer_idx);

        auto a3 = infinicore::Tensor::empty(
            {total, hidden_}, x->dtype(), x->device());
        infinicore::op::moe_w4a8_marlin_(
            a3, a2q, w2_, a2s, marlin_s2_, tw,
            padded_sorted, expert_ids, num_tokens_post_pad,
            1, topk_);
        debug_dump_moe(a3, "a3_marlin", layer_idx);

        auto out = infinicore::Tensor::empty(
            {m, hidden_}, x->dtype(), x->device());
        infinicore::op::moe_sum_vendor_(
            out, a3->view({m, topk_, hidden_}),
            std::nullopt, shared_output);
        debug_dump_moe(out, "combined_marlin", layer_idx);
        if (tp_ > 1 && comm_) {
            infinicore::op::distributed::allreduce_(
                out, out, INFINICCL_SUM, comm_);
        }
        return out;
    }

    const bool group_counts_on_device = dec || x->device().getType() == infinicore::Device::Type::HYGON;
    auto gc = group_counts_on_device ? cnt : cnt->to(infinicore::Device::Type::CPU);
    auto a1 = infinicore::Tensor::empty({total, hidden_}, infinicore::DataType::I8, x->device()), a1s = infinicore::Tensor::empty({total, 1}, infinicore::DataType::F32, x->device());
    infinicore::op::moe_expand_input_with_inv_pos_(a1, a1s, x, inv, topk_, 128, fmt);
    debug_dump_moe(a1, "a1", layer_idx);
    debug_dump_moe(a1s, "a1_scale", layer_idx);
    auto a2 = infinicore::Tensor::empty({total, inter_ * 2}, x->dtype(), x->device());
    infinicore::op::w4a8_group_gemm_(a2, a1, w1_, a1s, s1_, gc, std::nullopt, std::nullopt, true, dec);
    debug_dump_moe(a2, "a2", layer_idx);
    auto a2q = infinicore::Tensor::empty({total, inter_}, infinicore::DataType::I8, x->device()), a2s = infinicore::Tensor::empty({total, 1}, infinicore::DataType::F32, x->device());
    infinicore::op::moe_silu_and_mul_quant_(a2q, a2s, a2, fmt);
    debug_dump_moe(a2q, "a2_quant", layer_idx);
    debug_dump_moe(a2s, "a2_scale", layer_idx);
    auto a3 = infinicore::Tensor::empty({total, hidden_}, x->dtype(), x->device());
    infinicore::op::w4a8_group_gemm_(a3, a2q, w2_, a2s, s2_, gc, sorted, std::nullopt, true, dec);
    debug_dump_moe(a3, "a3", layer_idx);
    auto out = infinicore::Tensor::empty({m, hidden_}, x->dtype(), x->device());
    infinicore::op::moe_sum_vendor_(
        out, a3->view({m, topk_, hidden_}), tw, shared_output);
    debug_dump_moe(out, "combined", layer_idx);
    if (tp_ > 1 && comm_) {
        infinicore::op::distributed::allreduce_(out, out, INFINICCL_SUM, comm_);
    }
    return out;
}
GlmMoE::GlmMoE(std::shared_ptr<infinilm::config::ModelConfig> c,
               size_t layer_idx,
               const infinicore::Device &d)
    : layer_idx_(layer_idx) {
    INFINICORE_NN_MODULE_INIT(gate, c, d);
    INFINICORE_NN_MODULE_INIT(experts, c, d);
    auto n = c->get_or<size_t>("n_shared_experts", 0);
    shared_ = n > 0;
    if (shared_) {
        auto j = c->get_config_json();
        j["intermediate_size"] = c->get<size_t>("moe_intermediate_size") * n;
        j["reduce_results"] = false;
        auto sc = std::make_shared<infinilm::config::ModelConfig>(j);
        INFINICORE_NN_MODULE_INIT(shared_experts, sc, d);
    }
}
infinicore::Tensor GlmMoE::forward(const infinicore::Tensor &x) const {
    auto s = x->shape();
    auto f = x->view({s[0] * s[1], s[2]});
    auto [w, i] = gate_->forward(f);
    debug_dump_moe(w, "router_weights", layer_idx_);
    debug_dump_moe(i, "router_ids", layer_idx_);
    std::optional<infinicore::Tensor> shared_output;
    if (shared_) {
        shared_output = shared_experts_->forward(x)->view({s[0] * s[1], s[2]});
        debug_dump_moe(shared_output.value(), "shared", layer_idx_);
    }
    return experts_->forward(f, i, w, shared_output, layer_idx_)->view(s);
}
} // namespace infinilm::models::glm_moe_dsa
