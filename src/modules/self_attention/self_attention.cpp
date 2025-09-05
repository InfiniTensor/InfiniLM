#include "self_attention.hpp"

#include "../../context/inference_context.hpp"

namespace infinicore::nn::module {
std::shared_ptr<SelfAttention> SelfAttention::init(
    infiniDtype_t dtype,
    size_t hidden_size,
    size_t n_q_heads,
    size_t n_kv_heads,
    size_t qk_head_dim,
    size_t v_head_dim,
    int nranks,
    bool has_qkv_bias,
    bool has_qk_norm) {
    auto result = std::shared_ptr<SelfAttention>(new SelfAttention());

    return result;
}

void SelfAttention::register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix, int rank) {
}
void SelfAttention::forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual, std::shared_ptr<Tensor> attn_score_buf, std::shared_ptr<Tensor> attn_val_buf) {
}

} // namespace infinicore::nn::module