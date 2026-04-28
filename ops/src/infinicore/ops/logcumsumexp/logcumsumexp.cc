#include "infinicore/ops/logcumsumexp.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

// 初始化 Dispatcher 单例
common::OpDispatcher<LogCumSumExp::schema> &LogCumSumExp::dispatcher() {
    static common::OpDispatcher<LogCumSumExp::schema> dispatcher_;
    return dispatcher_;
}

// 算子执行逻辑：校验设备并分发任务
void LogCumSumExp::execute(Tensor y, Tensor x, int axis, bool exclusive, bool reverse) {
    // 确保输入输出张量在同一设备上
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);

    // 切换到目标设备的上下文
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x, axis, exclusive, reverse);
}

// 函数式接口：自动创建输出张量并返回
Tensor logcumsumexp(Tensor x, int axis, bool exclusive, bool reverse) {
    // 创建一个与输入 x 形状、类型和设备相同的空张量作为输出
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    logcumsumexp_(y, x, axis, exclusive, reverse);
    return y;
}

// 原地/指定输出接口
void logcumsumexp_(Tensor y, Tensor x, int axis, bool exclusive, bool reverse) {
    LogCumSumExp::execute(y, x, axis, exclusive, reverse);
}

} // namespace infinicore::op
