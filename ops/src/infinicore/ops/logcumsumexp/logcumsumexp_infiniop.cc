#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/logcumsumexp.hpp"
#include <infiniop.h>

namespace infinicore::op::logcumsumexp_impl::infiniop {

// 定义缓存：用于管理 infiniop 算子描述符的生命周期
thread_local common::OpCache<size_t, infiniopLogCumSumExpDescriptor_t> caches(
    100, // 缓存容量
    [](infiniopLogCumSumExpDescriptor_t &desc) {
        if (desc != nullptr) {
            // 使用 API 定义中的销毁函数
            INFINICORE_CHECK_ERROR(infiniopDestroyLogCumSumExpDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x, int axis, bool exclusive, bool reverse) {
    // 1. 生成 Hash Key：必须包含张量信息和算子特有参数 (axis, exclusive, reverse)
    size_t seed = hash_combine(y, x, axis, exclusive, reverse);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    // 2. 获取当前设备对应的缓存实例
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopLogCumSumExpDescriptor_t desc = nullptr;

    // 3. 如果缓存未命中，创建新的描述符
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLogCumSumExpDescriptor(
            context::getInfiniopHandle(y->device()),
            &desc,
            y->desc(), // 输出张量描述符
            x->desc(), // 输入张量描述符
            axis,
            exclusive,
            reverse));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 获取算子执行所需的 Workspace 大小并分配内存
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLogCumSumExpWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 5. 执行算子
    INFINICORE_CHECK_ERROR(infiniopLogCumSumExp(
        desc,
        workspace->data(),
        workspace_size,
        y->data(),
        x->data(),
        context::getStream()));
}

// 6. 自动注册：将此实现注册到 LogCumSumExp 的调度器中
static bool registered = []() {
    LogCumSumExp::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::logcumsumexp_impl::infiniop
