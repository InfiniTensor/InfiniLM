#include "infinicore/ops/pad.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Pad);

Pad::Pad(Tensor y, const Tensor &x, const std::vector<int> &pad, const std::string &mode, double value) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, pad, mode, value);
}

void Pad::execute(Tensor y, const Tensor &x, const std::vector<int> &pad, const std::string &mode, double value) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Pad, y, x, pad, mode, value);
}

static Shape infer_padded_shape(const Shape &in_shape, const std::vector<int> &pad) {
    if (pad.empty() || (pad.size() % 2) != 0) {
        throw std::runtime_error("pad: pad must have even length.");
    }

    Shape out_shape = in_shape;
    const size_t ndim = out_shape.size();
    const size_t dims_padded = pad.size() / 2;
    if (dims_padded > ndim) {
        throw std::runtime_error("pad: pad has more dimensions than input.");
    }

    for (size_t j = 0; j < dims_padded; ++j) {
        const int left = pad[2 * j];
        const int right = pad[2 * j + 1];
        if (left < 0 || right < 0) {
            throw std::runtime_error("pad: negative pad is not supported.");
        }
        const size_t dim = ndim - 1 - j;
        out_shape[dim] += static_cast<size_t>(left + right);
    }

    return out_shape;
}

Tensor pad(const Tensor &x, const std::vector<int> &pad, const std::string &mode, double value) {
    auto y_shape = infer_padded_shape(x->shape(), pad);
    auto y = Tensor::empty(y_shape, x->dtype(), x->device());
    pad_(y, x, pad, mode, value);
    return y;
}

void pad_(Tensor y, const Tensor &x, const std::vector<int> &pad, const std::string &mode, double value) {
    Pad::execute(y, x, pad, mode, value);
}

} // namespace infinicore::op
