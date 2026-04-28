#include "infinicore/ops/addcmul.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Addcmul::schema> &Addcmul::dispatcher() {
    static common::OpDispatcher<Addcmul::schema> dispatcher_;
    return dispatcher_;
};

// 执行核心逻辑：设备校验与后端分发
void Addcmul::execute(Tensor out, Tensor input, Tensor t1, Tensor t2, float value) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, t1, t2);
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(out, input, t1, t2, value);
}

// Out-of-place 接口：自动创建输出 Tensor
Tensor addcmul(Tensor input, Tensor t1, Tensor t2, float value) {
    auto out = Tensor::empty(input->shape(), input->dtype(), input->device());
    addcmul_(out, input, t1, t2, value);
    return out;
}

void addcmul_(Tensor out, Tensor input, Tensor t1, Tensor t2, float value) {
    Addcmul::execute(out, input, t1, t2, value);
}

} // namespace infinicore::op
