#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/smooth_l1_loss.hpp"
#include <infiniop.h>

namespace infinicore::op::smooth_l1_loss_impl::infiniop {

thread_local common::OpCache<size_t, infiniopSmoothL1LossDescriptor_t> caches(
    100, // capacity
    [](infiniopSmoothL1LossDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySmoothL1LossDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, Tensor target, float beta, int64_t reduction) {
    size_t seed = hash_combine(output, input, target, beta, reduction);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopSmoothL1LossDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSmoothL1LossDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            target->desc(),
            beta,
            static_cast<int>(reduction)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSmoothL1LossWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopSmoothL1Loss(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        target->data(),
        context::getStream()));
}

static bool registered = []() {
    SmoothL1Loss::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::smooth_l1_loss_impl::infiniop
