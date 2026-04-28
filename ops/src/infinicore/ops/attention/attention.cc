#include "infinicore/ops/attention.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Attention::schema> &Attention::dispatcher() {
    static common::OpDispatcher<Attention::schema> dispatcher_;
    return dispatcher_;
};

void Attention::execute(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v, k_cache, v_cache);
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(out, q, k, v, k_cache, v_cache, pos);
}

Tensor attention(Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    size_t n_q_head = q->shape()[0];
    size_t seq_len = q->shape()[1];
    size_t head_dim = q->shape()[2];
    Shape shape = {seq_len, n_q_head, head_dim};
    auto out = Tensor::empty(shape, q->dtype(), q->device());
    attention_(out, q, k, v, k_cache, v_cache, pos);
    return out;
}

void attention_(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    Attention::execute(out, q, k, v, k_cache, v_cache, pos);
}

} // namespace infinicore::op
