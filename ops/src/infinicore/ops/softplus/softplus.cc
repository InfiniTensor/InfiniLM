#include "infinicore/ops/softplus.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Softplus::schema> &Softplus::dispatcher() {
    static common::OpDispatcher<Softplus::schema> dispatcher_;
    return dispatcher_;
};

// 修改：增加 beta 和 threshold 参数
void Softplus::execute(Tensor y, Tensor x, float beta, float threshold) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());

    // 修改：将 beta 和 threshold 传递给底层的实现 (infiniop wrapper)
    dispatcher().lookup(y->device().getType())(y, x, beta, threshold);
}

// 修改：增加 beta 和 threshold 参数
Tensor softplus(Tensor x, float beta, float threshold) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    // 传递参数
    softplus_(y, x, beta, threshold);
    return y;
}

// 修改：增加 beta 和 threshold 参数
void softplus_(Tensor y, Tensor x, float beta, float threshold) {
    // 传递参数
    Softplus::execute(y, x, beta, threshold);
}

} // namespace infinicore::op
