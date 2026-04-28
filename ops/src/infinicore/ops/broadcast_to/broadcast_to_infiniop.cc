#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::broadcast_to_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopBroadcastToDescriptor_t> caches(
    100, // capacity
    [](infiniopBroadcastToDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyBroadcastToDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopBroadcastToDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符
        INFINICORE_CHECK_ERROR(infiniopCreateBroadcastToDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            y->desc(),
            x->desc()));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetBroadcastToWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopBroadcastTo(
        desc,
        workspace->data(),
        workspace_size,
        y->data(),
        x->data(),
        context::getStream()));
}

// 4. 注册算子实现
static bool registered = []() {
    BroadcastTo::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::broadcast_to_impl::infiniop
