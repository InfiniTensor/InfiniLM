#include "infinicore/ops/swiglu.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SwiGLU);

SwiGLU::SwiGLU(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().type(), c, a, b);
}

void SwiGLU::execute(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SwiGLU, c, a, b);
}

Tensor swiglu(const Tensor &a, const Tensor &b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    swiglu_(c, a, b);
    return c;
}

void swiglu_(Tensor c, const Tensor &a, const Tensor &b) {
    constexpr Size MAX_ELEMENTS_PER_LAUNCH = Size{1} << 30;
    INFINICORE_ASSERT(c->shape() == a->shape());
    INFINICORE_ASSERT(a->shape() == b->shape());
    INFINICORE_ASSERT(c->dtype() == a->dtype());
    INFINICORE_ASSERT(a->dtype() == b->dtype());

    if (c->numel() <= MAX_ELEMENTS_PER_LAUNCH) {
        SwiGLU::execute(c, a, b);
        return;
    }

    const Size row_width = c->size(c->ndim() - 1);
    INFINICORE_ASSERT(row_width > 0 && row_width <= MAX_ELEMENTS_PER_LAUNCH);
    const Size num_rows = c->numel() / row_width;
    const Size max_rows = MAX_ELEMENTS_PER_LAUNCH / row_width;
    auto c_rows = c->view({num_rows, row_width});
    auto a_rows = a->view({num_rows, row_width});
    auto b_rows = b->view({num_rows, row_width});

    for (Size start = 0; start < num_rows; start += max_rows) {
        const Size remaining = num_rows - start;
        const Size rows = remaining < max_rows ? remaining : max_rows;
        SwiGLU::execute(c_rows->narrow({{0, start, rows}}),
                        a_rows->narrow({{0, start, rows}}),
                        b_rows->narrow({{0, start, rows}}));
    }
}

} // namespace infinicore::op
