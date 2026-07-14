#include "glm_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/cat.hpp"
#include <cmath>
#include <cstdlib>
#include <string>
namespace infinilm::models::glm_moe_dsa {
GlmAttention::GlmAttention(std::shared_ptr<infinilm::config::ModelConfig> c, size_t layer, const infinicore::Device &d) {
    qn_ = c->get<size_t>("qk_nope_head_dim");
    qr_ = c->get<size_t>("qk_rope_head_dim");
    qh_ = qn_ + qr_;
    vh_ = c->get<size_t>("v_head_dim");
    ql_ = c->get<size_t>("q_lora_rank");
    kvl_ = c->get<size_t>("kv_lora_rank");
    auto total = c->get<size_t>("num_attention_heads");
    auto &r = infinilm::global_state::get_tensor_model_parallel_rank_info();
    heads_ = total / r.tp_size;
    scale_ = 1 / std::sqrt(float(qh_));
    auto q = c->get_quantization_method();
    auto dt = c->get_dtype();
    auto h = c->get<size_t>("hidden_size");
    auto eps = c->get<double>("rms_norm_eps");
    INFINICORE_NN_MODULE_INIT(q_a_proj, h, ql_, q, false, dt, d);
    INFINICORE_NN_MODULE_INIT(q_a_layernorm, ql_, eps, dt, d);
    INFINICORE_NN_MODULE_INIT(q_b_proj, ql_, total * qh_, q, false, dt, d, r.tp_rank, r.tp_size);
    INFINICORE_NN_MODULE_INIT(kv_a_proj_with_mqa, h, kvl_ + qr_, q, false, dt, d);
    INFINICORE_NN_MODULE_INIT(kv_a_layernorm, kvl_, eps, dt, d);
    INFINICORE_NN_MODULE_INIT(kv_b_proj, kvl_, total * (qn_ + vh_), q, false, dt, d, r.tp_rank, r.tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj, total * vh_, h, q, false, dt, d, r.tp_rank, r.tp_size, r.comm);
    rope_ = infinilm::layers::rotary_embedding::get_rope(c, d);
    auto backend = infinilm::global_state::get_infinilm_config().attention_backend;
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(heads_, qh_, scale_, heads_, layer, ks_, vs_, backend);
    infinilm::layers::attention::init_kv_cache_quant_params([this](const std::string &n, infinicore::nn::Parameter p) { register_parameter(n, std::move(p)); }, d, ks_, vs_);
}
infinicore::Tensor GlmAttention::forward(const infinicore::Tensor &pos, const infinicore::Tensor &x) const {
    auto sh = x->shape();
    size_t b = sh[0], s = sh[1], m = b * s;
    auto xm = x;
    auto qa = q_a_proj_->forward(xm);
    auto qan = q_a_layernorm_->forward(qa);
    auto q = q_b_proj_->forward(qan)->view({m, heads_, qh_});
    auto qn = q->narrow({{2, 0, qn_}})->contiguous(), qp = q->narrow({{2, qn_, qr_}})->contiguous();
    auto kva = kv_a_proj_with_mqa_->forward(xm)->view({m, kvl_ + qr_});
    auto kvc = kva->narrow({{1, 0, kvl_}})->contiguous();
    auto kp = kva->narrow({{1, kvl_, qr_}})->contiguous();
    auto kvn = kv_a_layernorm_->forward(kvc);
    auto kv = kv_b_proj_->forward(kvn)->view({m, heads_, qn_ + vh_});
    auto kn = kv->narrow({{2, 0, qn_}})->contiguous();
    auto v = kv->narrow({{2, qn_, vh_}})->contiguous();
    auto p = pos->contiguous()->view({m});
    qp = rope_->forward(qp, p, true);
    auto kpr = rope_->forward(kp->view({m, 1, qr_}), p, true);
    kp = infinicore::op::broadcast_to(kpr, {static_cast<int64_t>(m), static_cast<int64_t>(heads_), static_cast<int64_t>(qr_)})->contiguous();
    auto qs = infinicore::op::cat({qn, qp}, 2);
    auto k = infinicore::op::cat({kn, kp}, 2);
    auto vv = v;
    auto out = attn_->forward(qs, k, vv)->view({b, s, heads_ * vh_});
    return o_proj_->forward(out);
}
} // namespace infinilm::models::glm_moe_dsa
