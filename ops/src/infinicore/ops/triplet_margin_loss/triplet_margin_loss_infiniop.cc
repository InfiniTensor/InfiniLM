#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/triplet_margin_loss.hpp"
#include <infiniop.h>

namespace infinicore::op::triplet_margin_loss_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopTripletMarginLossDescriptor_t> caches(
    100, // capacity
    [](infiniopTripletMarginLossDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyTripletMarginLossDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction) {
    // 1. 计算 Hash Seed 作为 Cache Key
    size_t seed = hash_combine(output, anchor, positive, negative, margin, p, eps, swap, reduction);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopTripletMarginLossDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符
        INFINICORE_CHECK_ERROR(infiniopCreateTripletMarginLossDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            anchor->desc(),
            positive->desc(),
            negative->desc(),
            margin,
            static_cast<int>(p),
            eps,
            static_cast<int>(swap), // bool -> int
            static_cast<int>(reduction)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetTripletMarginLossWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopTripletMarginLoss(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        anchor->data(),
        positive->data(),
        negative->data(),
        context::getStream()));
}

static bool registered = []() {
    TripletMarginLoss::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::triplet_margin_loss_impl::infiniop
