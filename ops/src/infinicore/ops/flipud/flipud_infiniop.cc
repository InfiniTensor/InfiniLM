#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/flipud.hpp"
#include <infiniop.h>

namespace infinicore::op::flipud_impl::infiniop {

thread_local common::OpCache<size_t, infiniopFlipudDescriptor_t> caches(
    100, // capacity
    [](infiniopFlipudDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyFlipudDescriptor(desc));
            desc = nullptr;
        }
    });

// 执行函数
void calculate(Tensor output, Tensor input) {
    // 1. 计算缓存 Key
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopFlipudDescriptor_t desc = nullptr;

    // 2. 获取或创建描述符
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateFlipudDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc()));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetFlipudWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);
    INFINICORE_CHECK_ERROR(infiniopFlipud(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Flipud::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::flipud_impl::infiniop
