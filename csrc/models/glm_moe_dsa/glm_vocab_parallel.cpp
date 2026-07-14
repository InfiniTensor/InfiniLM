#include "glm_vocab_parallel.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allgather.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/vocab_parallel_embedding.hpp"
namespace infinilm::models::glm_moe_dsa {
GlmVocabEmbedding::GlmVocabEmbedding(size_t vocab, size_t hidden, const infinicore::DataType &dt, const infinicore::Device &d) {
    auto &r = infinilm::global_state::get_tensor_model_parallel_rank_info();
    hidden_ = hidden;
    start_ = vocab * r.tp_rank / r.tp_size;
    end_ = vocab * (r.tp_rank + 1) / r.tp_size;
    comm_ = r.comm;
    INFINICORE_NN_PARAMETER_INIT(weight, ({vocab, hidden}, dt, d, 0, r.tp_rank, r.tp_size));
}
infinicore::Tensor GlmVocabEmbedding::forward(const infinicore::Tensor &ids) const {
    auto s = ids->shape();
    s.push_back(hidden_);
    auto out = infinicore::Tensor::empty(s, weight_->dtype(), weight_->device());
    infinicore::op::vocab_parallel_embedding_(out, ids, weight_, start_, end_);
    if (comm_) {
        infinicore::op::distributed::allreduce_(out, out, INFINICCL_SUM, comm_);
    }
    return out;
}
GlmVocabLMHead::GlmVocabLMHead(size_t hidden, size_t vocab, bool, const infinicore::DataType &dt, const infinicore::Device &d) {
    auto &r = infinilm::global_state::get_tensor_model_parallel_rank_info();
    vocab_ = vocab;
    world_ = r.tp_size;
    comm_ = r.comm;
    INFINICORE_NN_PARAMETER_INIT(weight, ({vocab, hidden}, dt, d, 0, r.tp_rank, r.tp_size));
}
infinicore::Tensor GlmVocabLMHead::forward(const infinicore::Tensor &x) const {
    auto local = infinicore::op::linear(x, weight_, std::nullopt, 1);
    if (world_ == 1) {
        return local;
    }
    auto t = local->permute({2, 0, 1})->contiguous();
    auto g = infinicore::op::distributed::allgather(t, world_, comm_);
    return g->permute({1, 2, 0})->contiguous();
}
} // namespace infinilm::models::glm_moe_dsa
