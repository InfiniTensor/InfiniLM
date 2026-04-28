#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/huber_loss.hpp"
#include <infiniop.h>

namespace infinicore::op::huber_loss_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopHuberLossDescriptor_t> caches(
    100, // capacity
    [](infiniopHuberLossDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyHuberLossDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, Tensor target, float delta, int64_t reduction) {
    size_t seed = hash_combine(output, input, target, delta, reduction);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopHuberLossDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 3. 创建描述符
        INFINICORE_CHECK_ERROR(infiniopCreateHuberLossDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            target->desc(),
            delta,
            static_cast<int>(reduction)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetHuberLossWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopHuberLoss(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        target->data(),
        context::getStream()));
}

static bool registered = []() {
    HuberLoss::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::huber_loss_impl::infiniop
