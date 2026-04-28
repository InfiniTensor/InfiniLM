#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/atanh.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::atanh_impl::infiniop {

// 定义线程局部的算子描述符缓存，避免重复创建 Descriptor 带来的开销
thread_local common::OpCache<size_t, infiniopAtanhDescriptor_t> caches(
    100, // 缓存容量
    [](infiniopAtanhDescriptor_t &desc) {
        if (desc != nullptr) {
            // 缓存释放时的回调：销毁 infiniop 算子描述符
            INFINICORE_CHECK_ERROR(infiniopDestroyAtanhDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor a) {
    // 1. 根据 Tensor 的形状、步长、类型等信息生成唯一 Hash 值
    size_t seed = hash_combine(y, a);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    // 2. 尝试从缓存中获取已存在的描述符
    auto desc_opt = cache.get(seed);
    infiniopAtanhDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 如果缓存未命中，创建新的描述符
        INFINICORE_CHECK_ERROR(infiniopCreateAtanhDescriptor(
            context::getInfiniopHandle(device), &desc,
            y->desc(), a->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取并分配必要的 Workspace 空间（如果有的话）
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAtanhWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 4. 执行底层计算
    INFINICORE_CHECK_ERROR(infiniopAtanh(
        desc, workspace->data(), workspace_size,
        y->data(), a->data(), context::getStream()));
}

// 5. 自动注册逻辑：程序启动时将此实现注册到分发器中
static bool registered = []() {
    Atanh::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::atanh_impl::infiniop
