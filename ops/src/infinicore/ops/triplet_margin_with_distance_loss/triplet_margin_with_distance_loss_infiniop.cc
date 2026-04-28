#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/triplet_margin_with_distance_loss.hpp"
#include <infiniop.h>

namespace infinicore::op::triplet_margin_with_distance_loss_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopTripletMarginWithDistanceLossDescriptor_t> caches(
    100, // capacity
    [](infiniopTripletMarginWithDistanceLossDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyTripletMarginWithDistanceLossDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor anchor, Tensor positive, Tensor negative, double margin, bool swap, int64_t reduction) {
    size_t seed = hash_combine(output, anchor, positive, negative, margin, swap, reduction);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopTripletMarginWithDistanceLossDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateTripletMarginWithDistanceLossDescriptor(
            context::getInfiniopHandle(anchor->device()),
            &desc,
            output->desc(),
            anchor->desc(),
            positive->desc(),
            negative->desc(),
            static_cast<float>(margin),
            static_cast<int>(swap),
            static_cast<int>(reduction)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetTripletMarginWithDistanceLossWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopTripletMarginWithDistanceLoss(
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
    TripletMarginWithDistanceLoss::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::triplet_margin_with_distance_loss_impl::infiniop
