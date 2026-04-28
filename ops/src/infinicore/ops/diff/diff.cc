#include "infinicore/ops/diff.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Diff);

Diff::Diff(Tensor y, const Tensor &x, int dim, int n) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, dim, n);
}

void Diff::execute(Tensor y, const Tensor &x, int dim, int n) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Diff, y, x, dim, n);
}

static int normalize_dim(int dim, size_t ndim) {
    if (ndim == 0) {
        throw std::runtime_error("diff: input tensor must have at least one dimension.");
    }
    if (dim < 0) {
        dim += static_cast<int>(ndim);
    }
    if (dim < 0 || static_cast<size_t>(dim) >= ndim) {
        throw std::runtime_error("diff: dim out of range.");
    }
    return dim;
}

Tensor diff(const Tensor &x, int n, int dim) {
    if (n < 0) {
        throw std::runtime_error("diff: n must be non-negative.");
    }
    Shape y_shape = x->shape();
    const int d = normalize_dim(dim, y_shape.size());
    const auto dim_size = y_shape[static_cast<size_t>(d)];
    y_shape[static_cast<size_t>(d)] = (dim_size >= static_cast<size_t>(n)) ? (dim_size - static_cast<size_t>(n)) : 0;

    auto y = Tensor::empty(y_shape, x->dtype(), x->device());
    if (n == 0) {
        y->copy_from(x);
        return y;
    }
    if (dim_size <= static_cast<size_t>(n)) {
        // Empty output by definition; nothing to compute.
        return y;
    }

    diff_(y, x, n, dim);
    return y;
}

void diff_(Tensor y, const Tensor &x, int n, int dim) {
    if (n < 0) {
        throw std::runtime_error("diff_: n must be non-negative.");
    }
    const int d = normalize_dim(dim, x->shape().size());
    Shape expected = x->shape();
    const auto dim_size = expected[static_cast<size_t>(d)];
    expected[static_cast<size_t>(d)] = (dim_size >= static_cast<size_t>(n)) ? (dim_size - static_cast<size_t>(n)) : 0;
    if (y->shape() != expected) {
        throw std::runtime_error("diff_: output tensor has incorrect shape.");
    }
    if (n == 0) {
        y->copy_from(x);
        return;
    }
    if (x->shape()[static_cast<size_t>(d)] <= static_cast<size_t>(n)) {
        // Empty output by definition; nothing to compute.
        return;
    }
    Diff::execute(y, x, d, n);
}

} // namespace infinicore::op
