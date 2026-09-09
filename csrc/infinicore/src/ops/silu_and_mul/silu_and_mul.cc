#include "infinicore/ops/silu_and_mul.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SiluAndMul);

SiluAndMul::SiluAndMul(Tensor out, const Tensor &x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(), out, x);
}

void SiluAndMul::execute(Tensor out, const Tensor &x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SiluAndMul, out, x);
}

Tensor silu_and_mul(const Tensor &x) {
    Shape shape = x->shape();
    size_t ndim = x->ndim();

    if (shape[ndim - 1] % 2 != 0) {
        throw std::runtime_error("SiluAndMul input last dim must be even.");
    }
    shape[ndim - 1] /= 2;

    auto out = Tensor::empty(shape, x->dtype(), x->device());
    silu_and_mul_(out, x);
    return out;
}

void silu_and_mul_(Tensor out, const Tensor &x) {
    constexpr Size MAX_ELEMENTS_PER_LAUNCH = Size{1} << 30;
    if (out->numel() <= MAX_ELEMENTS_PER_LAUNCH
        || !out->is_contiguous()
        || !x->is_contiguous()) {
        SiluAndMul::execute(out, x);
        return;
    }

    const Size output_row_width = out->size(out->ndim() - 1);
    const Size input_row_width = x->size(x->ndim() - 1);
    INFINICORE_ASSERT(output_row_width > 0
                      && output_row_width <= MAX_ELEMENTS_PER_LAUNCH);
    INFINICORE_ASSERT(input_row_width == output_row_width * 2);
    INFINICORE_ASSERT(x->numel() == out->numel() * 2);

    const Size num_rows = out->numel() / output_row_width;
    const Size max_rows = MAX_ELEMENTS_PER_LAUNCH / output_row_width;
    auto output_rows = out->view({num_rows, output_row_width});
    auto input_rows = x->view({num_rows, input_row_width});

    for (Size start = 0; start < num_rows; start += max_rows) {
        const Size remaining = num_rows - start;
        const Size rows = remaining < max_rows ? remaining : max_rows;
        SiluAndMul::execute(output_rows->narrow({{0, start, rows}}),
                            input_rows->narrow({{0, start, rows}}));
    }
}

} // namespace infinicore::op
