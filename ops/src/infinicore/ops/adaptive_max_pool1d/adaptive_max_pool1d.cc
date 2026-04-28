#include "infinicore/ops/adaptive_max_pool1d.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<AdaptiveMaxPool1d::schema> &AdaptiveMaxPool1d::dispatcher() {
    static common::OpDispatcher<AdaptiveMaxPool1d::schema> dispatcher_;
    return dispatcher_;
}

void AdaptiveMaxPool1d::execute(Tensor y, Tensor x, size_t output_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x, output_size);
}

Tensor adaptive_max_pool1d(Tensor x, size_t output_size) {
    infinicore::Shape y_shape = x->shape();
    y_shape.back() = output_size;
    auto y = Tensor::empty(y_shape, x->dtype(), x->device());
    adaptive_max_pool1d_(y, x, output_size);
    return y;
}

void adaptive_max_pool1d_(Tensor y, Tensor x, size_t output_size) {
    AdaptiveMaxPool1d::execute(y, x, output_size);
}

} // namespace infinicore::op
