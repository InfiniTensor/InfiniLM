#include "infinicore/ops/cdist.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

// 静态调度器实例化
common::OpDispatcher<Cdist::schema> &Cdist::dispatcher() {
    static common::OpDispatcher<Cdist::schema> dispatcher_;
    return dispatcher_;
};

/**
 * 执行核心逻辑：设备校验与后端分发
 */
void Cdist::execute(Tensor out, Tensor x1, Tensor x2, double p) {
    // 校验三个 Tensor 是否在同一设备上
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x1, x2);

    // 设置当前设备上下文
    infinicore::context::setDevice(out->device());

    // 根据设备类型（CUDA/CPU/etc.）查找并执行注册的算子实现
    dispatcher().lookup(out->device().getType())(out, x1, x2, p);
}

/**
 * Out-of-place 接口：自动创建输出 Tensor
 * x1: (M, D), x2: (N, D) -> out: (M, N)
 */
Tensor cdist(Tensor x1, Tensor x2, double p) {
    // 1. 获取输入维度
    auto shape1 = x1->shape(); // 假设为 {M, D}
    auto shape2 = x2->shape(); // 假设为 {N, D}

    // 将原来的 std::vector<int64_t> 修改为 std::vector<uint64_t>
    std::vector<uint64_t> out_shape = {
        static_cast<uint64_t>(shape1[0]),
        static_cast<uint64_t>(shape2[0])};

    // 或者使用更简洁的初始化列表方式，强制转换类型
    auto out = Tensor::empty({(uint64_t)shape1[0], (uint64_t)shape2[0]}, x1->dtype(), x1->device());

    // 5. 调用执行接口
    cdist_(out, x1, x2, p);

    return out;
}

/**
 * 显式指定输出接口
 */
void cdist_(Tensor out, Tensor x1, Tensor x2, double p) {
    Cdist::execute(out, x1, x2, p);
}

} // namespace infinicore::op
