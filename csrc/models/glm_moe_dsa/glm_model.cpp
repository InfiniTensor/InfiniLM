#include "glm_model.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "infinicore/ops.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace infinilm::models::glm_moe_dsa {
GlmDecoder::GlmDecoder(std::shared_ptr<infinilm::config::ModelConfig> c, size_t i, const infinicore::Device &d) {
    auto h = c->get<size_t>("hidden_size");
    auto e = c->get<double>("rms_norm_eps");
    auto dt = c->get_dtype();
    INFINICORE_NN_MODULE_INIT(input_layernorm, h, e, dt, d);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, h, e, dt, d);
    INFINICORE_NN_MODULE_INIT(self_attn, c, i, d);
    moe_ = i >= c->get_or<size_t>("first_k_dense_replace", 0);
    if (moe_) {
        moe_mlp_ = register_module<GlmMoE>("mlp", c, d);
    } else {
        dense_mlp_ = register_module<GlmDenseMLP>("mlp", c, d);
    }
}
void GlmDecoder::forward(const infinicore::Tensor &p, infinicore::Tensor &x, infinicore::Tensor &r) const {
    input_layernorm_->forward_inplace(x, r);
    x = self_attn_->forward(p, x);
    post_attention_layernorm_->forward_inplace(x, r);
    x = moe_ ? moe_mlp_->forward(x) : dense_mlp_->forward(x);
}
GlmModel::GlmModel(std::shared_ptr<infinilm::config::ModelConfig> c, const infinicore::Device &d) {
    auto dt = c->get_dtype();
    auto h = c->get<size_t>("hidden_size");
    INFINICORE_NN_MODULE_INIT(embed_tokens, c->get<size_t>("vocab_size"), h, dt, d);
    for (size_t i = 0; i < c->get<size_t>("num_hidden_layers"); ++i) {
        layers_.push_back(register_module<GlmDecoder>("layers." + std::to_string(i), c, i, d));
    }
    INFINICORE_NN_MODULE_INIT(norm, h, c->get<double>("rms_norm_eps"), dt, d);
}
infinicore::Tensor GlmModel::forward(const infinilm::InfinilmModel::Input &i) const {
    auto x = embed_tokens_->forward(i.input_ids.value());
    infinicore::Tensor r;
    for (auto &l : layers_) {
        l->forward(i.position_ids.value(), x, r);
    }
    norm_->forward_inplace(x, r);
    return x;
}
GlmForCausalLM::GlmForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> c, const infinicore::Device &d) {
    model_config_ = c;
    INFINICORE_NN_MODULE_INIT(model, c, d);
    INFINICORE_NN_MODULE_INIT(lm_head, c->get<size_t>("hidden_size"), c->get<size_t>("vocab_size"), false, c->get_dtype(), d);
}
infinilm::InfinilmModel::Output GlmForCausalLM::forward(const Input &i) const {
    auto x = model_->forward(i);
    return {lm_head_->forward(x)};
}
std::shared_ptr<infinilm::config::ModelConfig> create_glm_config(std::shared_ptr<infinilm::config::ModelConfig> c) {
    auto j = c->get_config_json();
    auto qh = j.at("qk_nope_head_dim").get<size_t>() + j.at("qk_rope_head_dim").get<size_t>();
    j["head_dim"] = qh;
    if (!j.contains("rope_theta") && j.contains("rope_parameters")) {
        j["rope_theta"] = j["rope_parameters"].value("rope_theta", 10000.0);
    }
    j["partial_rotary_factor"] = double(j.at("qk_rope_head_dim").get<size_t>()) / double(qh);
    j["num_experts"] = j.at("n_routed_experts");
    j["mlp_bias"] = false;
    j["quantization_config"] = {{"quant_method", "glm_w8a8"}};
    auto n = std::make_shared<infinilm::config::ModelConfig>(j);
    n->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);
    return n;
}
} // namespace infinilm::models::glm_moe_dsa
namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(glm_moe_dsa, infinilm::models::glm_moe_dsa::GlmForCausalLM, infinilm::models::glm_moe_dsa::create_glm_config);
}
