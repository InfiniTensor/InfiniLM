#include "glm_moe.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/fused_moe_w8a8.hpp"
#include "infinicore/ops/grouped_topk_vendor.hpp"
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
namespace {

infinicore::Tensor shuffle_moe_c_w8a8_weight(const infinicore::Tensor &weight) {
    const auto original_shape = weight->shape();
    if (original_shape.size() != 3) {
        throw std::runtime_error("shuffle_moe_c_w8a8_weight: expected a 3D expert weight");
    }

    const size_t num_experts = original_shape[0];
    const size_t output_size = original_shape[1];
    const size_t input_size = original_shape[2];
    constexpr size_t k_tile = 64;
    constexpr size_t n_tile = 16;
    if (input_size % k_tile != 0 || output_size % n_tile != 0) {
        throw std::runtime_error(
            "shuffle_moe_c_w8a8_weight: weight dimensions do not fit the AITER MoE-C tiles");
    }

    // AITER's MoE-C W8A8 kernel consumes a tiled K-major layout rather than
    // the checkpoint's ordinary [expert, output, input] matrix layout.
    auto k_major = weight->permute({0, 2, 1})->contiguous();
    auto tiled = k_major
                     ->view({num_experts,
                             input_size / k_tile,
                             k_tile,
                             output_size / n_tile,
                             n_tile})
                     ->permute({0, 1, 3, 4, 2})
                     ->contiguous();
    return tiled
        ->view({num_experts,
                input_size / k_tile,
                output_size / n_tile,
                1,
                n_tile,
                k_tile / n_tile,
                n_tile})
        ->permute({0, 1, 2, 3, 5, 4, 6})
        ->contiguous()
        ->view(original_shape);
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
    runtime_bias_ = infinicore::Tensor::empty({num_experts_}, weight_->dtype(), weight_->device());
    infinicore::op::cast_(runtime_bias_, e_score_correction_bias_);
}
std::tuple<infinicore::Tensor, infinicore::Tensor> GlmTopKRouter::forward(const infinicore::Tensor &x) const {
    auto logits = infinicore::op::linear(x, weight_, std::nullopt, 1);
    auto w = infinicore::Tensor::empty({x->size(0), top_k_}, infinicore::DataType::F32, x->device());
    auto ids = infinicore::Tensor::empty({x->size(0), top_k_}, infinicore::DataType::I32, x->device());
    infinicore::op::grouped_topk_vendor_(w, ids, logits, num_expert_group_, topk_group_, renormalize_, routed_scaling_factor_, runtime_bias_, "sigmoid");

    return {w, ids};
}
GlmW8A8Experts::GlmW8A8Experts(std::shared_ptr<infinilm::config::ModelConfig> c, const infinicore::Device &d) {
    hidden_ = c->get<size_t>("hidden_size");
    nexpert_ = c->get<size_t>("num_experts");
    topk_ = c->get<size_t>("num_experts_per_tok");
    auto &r = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_ = r.tp_size;
    comm_ = r.comm;
    auto full_i = c->get<size_t>("moe_intermediate_size");
    inter_ = full_i / tp_;
    w1_ = infinicore::Tensor::empty({nexpert_, inter_ * 2, hidden_}, infinicore::DataType::I8, d);
    s1_ = infinicore::Tensor::empty({nexpert_, inter_ * 2, 1}, infinicore::DataType::F32, d);
    w2_ = infinicore::Tensor::empty({nexpert_, hidden_, inter_}, infinicore::DataType::I8, d);
    s2_ = infinicore::Tensor::empty({nexpert_, hidden_, 1}, infinicore::DataType::F32, d);
    for (size_t i = 0; i < nexpert_; ++i) {
        auto p = std::to_string(i);
        auto gw = w1_->narrow({{0, i, 1}, {1, 0, inter_}})->view({inter_, hidden_});
        auto uw = w1_->narrow({{0, i, 1}, {1, inter_, inter_}})->view({inter_, hidden_});
        auto dw = w2_->narrow({{0, i, 1}})->view({hidden_, inter_});
        auto gs = s1_->narrow({{0, i, 1}, {1, 0, inter_}})->view({inter_, 1});
        auto us = s1_->narrow({{0, i, 1}, {1, inter_, inter_}})->view({inter_, 1});
        auto ds = s2_->narrow({{0, i, 1}})->view({hidden_, 1});
        register_parameter(p + ".gate_proj.weight", infinicore::nn::Parameter(gw, 0, r.tp_rank, r.tp_size));
        register_parameter(p + ".up_proj.weight", infinicore::nn::Parameter(uw, 0, r.tp_rank, r.tp_size));
        register_parameter(p + ".down_proj.weight", infinicore::nn::Parameter(dw, 1, r.tp_rank, r.tp_size));
        register_parameter(p + ".gate_proj.weight_scale", infinicore::nn::Parameter(gs, 0, r.tp_rank, r.tp_size));
        register_parameter(p + ".up_proj.weight_scale", infinicore::nn::Parameter(us, 0, r.tp_rank, r.tp_size));
        register_parameter(p + ".down_proj.weight_scale", infinicore::nn::Parameter(ds));
    }
}
void GlmW8A8Experts::process_weights_after_loading() {
    auto shuffled = shuffle_moe_c_w8a8_weight(w1_);
    w1_->copy_from(shuffled);
    shuffled = shuffle_moe_c_w8a8_weight(w2_);
    w2_->copy_from(shuffled);
}
infinicore::Tensor GlmW8A8Experts::forward(const infinicore::Tensor &x,
                                           const infinicore::Tensor &ids,
                                           const infinicore::Tensor &tw,
                                           std::optional<infinicore::Tensor> shared_output) const {
    if (!w1_) {
        throw std::runtime_error("GlmW8A8Experts: weights not ready");
    }
    auto out = infinicore::op::fused_moe_w8a8(x, ids, tw, w1_, s1_, w2_, s2_);
    if (shared_output.has_value()) {
        infinicore::op::add_(out, out, shared_output.value());
    }
    if (tp_ > 1 && comm_) {
        infinicore::op::distributed::allreduce_(out, out, INFINICCL_SUM, comm_);
    }
    return out;
}
GlmMoE::GlmMoE(std::shared_ptr<infinilm::config::ModelConfig> c, const infinicore::Device &d) {
    INFINICORE_NN_MODULE_INIT(gate, c, d);
    INFINICORE_NN_MODULE_INIT(experts, c, d);
    auto n = c->get_or<size_t>("n_shared_experts", 0);
    shared_ = n > 0;
    if (shared_) {
        auto j = c->get_config_json();
        j["intermediate_size"] = c->get<size_t>("moe_intermediate_size") * n;
        j["reduce_results"] = false;
        j["quantization_config"] = nullptr;
        auto sc = std::make_shared<infinilm::config::ModelConfig>(j);
        INFINICORE_NN_MODULE_INIT(shared_experts, sc, d);
    }
}
infinicore::Tensor GlmMoE::forward(const infinicore::Tensor &x) const {
    auto s = x->shape();
    auto f = x->view({s[0] * s[1], s[2]});
    auto [w, i] = gate_->forward(f);
    std::optional<infinicore::Tensor> shared_output;
    if (shared_) {
        shared_output = shared_experts_->forward(x)->view({s[0] * s[1], s[2]});
    }
    return experts_->forward(f, i, w, shared_output)->view(s);
}
} // namespace infinilm::models::glm_moe_dsa
