#pragma once

#include "../../dataloader/weights_loader.hpp"
#include "../../tensor.hpp"

#include "../linear/linear.hpp"
#include "../norm/rms_norm.hpp"

namespace infinicore::nn::module {
class MultiHeadAttention {
private:
    MultiHeadAttention() = default;

public:
    std::shared_ptr<Linear> q_proj, k_proj, v_proj, o_proj;
    std::shared_ptr<RMSNorm> q_norm, k_norm;
    size_t n_q_heads;
    size_t n_kv_heads;
    size_t head_dim;
    static std::shared_ptr<MultiHeadAttention> init(infiniDtype_t dtype, size_t hidden_size, size_t n_q_heads, size_t n_kv_heads, size_t qk_head_dim, size_t v_head_dim, int nranks = 1, bool has_qkv_bias = false, bool has_qk_norm = false, float epsilon = 1e-6);
    void register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix = "", int rank = 0);
    void forward(std::shared_ptr<Tensor> output,
                 std::shared_ptr<Tensor> input,
                 std::shared_ptr<Tensor> residual,
                 std::shared_ptr<Tensor> attn_score_buf,
                 std::shared_ptr<Tensor> attn_val_buf,
                 std::shared_ptr<Tensor> pos_ids,
                 std::shared_ptr<Tensor> sin_table,
                 std::shared_ptr<Tensor> cos_table,
                 std::vector<std::shared_ptr<Tensor>> k_caches,
                 std::vector<std::shared_ptr<Tensor>> v_caches,
                 const std::vector<uint32_t> &req_lens,
                 const std::vector<uint32_t> &req_pos,
                 uint32_t nreq);
};
} // namespace infinicore::nn::module