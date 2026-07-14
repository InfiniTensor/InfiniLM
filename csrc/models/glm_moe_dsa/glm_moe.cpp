#include "glm_moe.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/grouped_topk_vllm.hpp"
#include "infinicore/ops/moe_argsort_bincount.hpp"
#include "infinicore/ops/moe_expand_input.hpp"
#include "infinicore/ops/moe_silu_and_mul_quant.hpp"
#include "infinicore/ops/moe_sum_vllm.hpp"
#include "infinicore/ops/w4a8_group_gemm.hpp"
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
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
    infinicore::op::grouped_topk_vllm_(w, ids, logits, num_expert_group_, topk_group_, renormalize_, routed_scaling_factor_, runtime_bias_, "sigmoid");

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
infinicore::Tensor GlmW4A8Experts::forward(const infinicore::Tensor &x, const infinicore::Tensor &ids, const infinicore::Tensor &tw) const {
    if (!w1_) {
        throw std::runtime_error("GlmW4A8Experts: weights not ready");
    }
    size_t m = x->size(0), total = m * topk_;
    bool dec = m == 1;
    int64_t fmt = dec ? 2 : 1;
    auto cnt = infinicore::Tensor::empty({nexpert_}, infinicore::DataType::I32, x->device()), sorted = infinicore::Tensor::empty({total}, infinicore::DataType::I32, x->device()), inv = infinicore::Tensor::empty({total}, infinicore::DataType::I32, x->device());
    infinicore::op::moe_argsort_bincount_with_inv_pos_(cnt, sorted, inv, ids, nexpert_);
    auto gc = dec ? cnt : cnt->to(infinicore::Device::Type::CPU);
    auto a1 = infinicore::Tensor::empty({total, hidden_}, infinicore::DataType::I8, x->device()), a1s = infinicore::Tensor::empty({total, 1}, infinicore::DataType::F32, x->device());
    infinicore::op::moe_expand_input_with_inv_pos_(a1, a1s, x, inv, topk_, 128, fmt);
    auto a2 = infinicore::Tensor::empty({total, inter_ * 2}, x->dtype(), x->device());
    infinicore::op::w4a8_group_gemm_(a2, a1, w1_, a1s, s1_, gc, std::nullopt, std::nullopt, true, dec);
    auto a2q = infinicore::Tensor::empty({total, inter_}, infinicore::DataType::I8, x->device()), a2s = infinicore::Tensor::empty({total, 1}, infinicore::DataType::F32, x->device());
    infinicore::op::moe_silu_and_mul_quant_(a2q, a2s, a2, fmt);
    auto a3 = infinicore::Tensor::empty({total, hidden_}, x->dtype(), x->device());
    infinicore::op::w4a8_group_gemm_(a3, a2q, w2_, a2s, s2_, gc, sorted, std::nullopt, true, dec);
    auto out = infinicore::Tensor::empty({m, hidden_}, x->dtype(), x->device());
    infinicore::op::moe_sum_vllm_(out, a3->view({m, topk_, hidden_}), tw);
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
        auto sc = std::make_shared<infinilm::config::ModelConfig>(j);
        INFINICORE_NN_MODULE_INIT(shared_experts, sc, d);
    }
}
infinicore::Tensor GlmMoE::forward(const infinicore::Tensor &x) const {
    auto s = x->shape();
    auto f = x->view({s[0] * s[1], s[2]});
    auto [w, i] = gate_->forward(f);
    auto r = experts_->forward(f, i, w)->view(s);
    if (!shared_) {
        return r;
    }
    return infinicore::op::add(r, shared_experts_->forward(x));
}
} // namespace infinilm::models::glm_moe_dsa
