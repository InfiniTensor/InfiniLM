#include "infinicore/ops/mrope.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MRoPE);

MRoPE::MRoPE(Tensor q_out,
             Tensor k_out,
             const Tensor &q,
             const Tensor &k,
             const Tensor &cos,
             const Tensor &sin,
             const Tensor &positions,
             int head_size,
             int rotary_dim,
             int section_t,
             int section_h,
             int section_w,
             bool interleaved) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_out, k_out, q, k, cos, sin, positions);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().type(),
                                 q_out,
                                 k_out,
                                 q,
                                 k,
                                 cos,
                                 sin,
                                 positions,
                                 head_size,
                                 rotary_dim,
                                 section_t,
                                 section_h,
                                 section_w,
                                 interleaved);
}

void MRoPE::execute(Tensor q_out,
                    Tensor k_out,
                    const Tensor &q,
                    const Tensor &k,
                    const Tensor &cos,
                    const Tensor &sin,
                    const Tensor &positions,
                    int head_size,
                    int rotary_dim,
                    int section_t,
                    int section_h,
                    int section_w,
                    bool interleaved) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(MRoPE,
                                      q_out,
                                      k_out,
                                      q,
                                      k,
                                      cos,
                                      sin,
                                      positions,
                                      head_size,
                                      rotary_dim,
                                      section_t,
                                      section_h,
                                      section_w,
                                      interleaved);
}

void mrope_(Tensor q_out,
            Tensor k_out,
            const Tensor &q,
            const Tensor &k,
            const Tensor &cos,
            const Tensor &sin,
            const Tensor &positions,
            int head_size,
            int rotary_dim,
            int section_t,
            int section_h,
            int section_w,
            bool interleaved) {
    MRoPE::execute(q_out, k_out, q, k, cos, sin, positions, head_size, rotary_dim, section_t, section_h, section_w, interleaved);
}

std::pair<Tensor, Tensor> mrope(const Tensor &q,
                                const Tensor &k,
                                const Tensor &cos,
                                const Tensor &sin,
                                const Tensor &positions,
                                int head_size,
                                int rotary_dim,
                                int section_t,
                                int section_h,
                                int section_w,
                                bool interleaved) {
    auto q_out = Tensor::empty(q->shape(), q->dtype(), q->device());
    auto k_out = Tensor::empty(k->shape(), k->dtype(), k->device());
    mrope_(q_out, k_out, q, k, cos, sin, positions, head_size, rotary_dim, section_t, section_h, section_w, interleaved);
    return {q_out, k_out};
}

} // namespace infinicore::op
