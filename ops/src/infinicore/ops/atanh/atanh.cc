#include "infinicore/ops/atanh.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

// 获取单例分发器
common::OpDispatcher<Atanh::schema> &Atanh::dispatcher() {
    static common::OpDispatcher<Atanh::schema> dispatcher_;
    return dispatcher_;
};

// 执行入口：负责设备切换和后端查找
void Atanh::execute(Tensor y, Tensor a) {
    // 确保输入和输出在同一个设备上
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, a);

    // 切换当前上下文到目标设备
    infinicore::context::setDevice(y->device());

    // 根据设备类型（CPU/CUDA等）查找对应的实现并执行
    dispatcher().lookup(y->device().getType())(y, a);
}

// Out-of-place 接口：自动创建结果 Tensor
Tensor atanh(Tensor a) {
    // 创建一个与输入形状、类型、设备完全相同的空 Tensor
    auto y = Tensor::empty(a->shape(), a->dtype(), a->device());
    atanh_(y, a);
    return y;
}

// In-place 或指定输出接口
void atanh_(Tensor y, Tensor a) {
    Atanh::execute(y, a);
}

} // namespace infinicore::op
