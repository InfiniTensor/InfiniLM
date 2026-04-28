#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/argwhere.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infiniop/ops/argwhere.h"

namespace infinicore::op::argwhere_impl::infiniop {
thread_local common::OpCache<size_t, infiniopArgwhereDescriptor_t> caches(
    100, // capacity
    [](infiniopArgwhereDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyArgwhereDescriptor(desc));
            desc = nullptr;
        }
    });
void calculate(void **y, size_t *count, Tensor x) {
    size_t seed = hash_combine(x);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopArgwhereDescriptor_t desc = nullptr;
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateArgwhereDescriptor(
            context::getInfiniopHandle(x->device()),
            &desc,
            x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetArgwhereWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopArgwhere(
        desc,
        workspace->data(), workspace_size,
        y,
        count,
        x->data(),
        context::getStream()));
}
static bool registered = []() {
    Argwhere::dispatcher().registerAll(&calculate, false);
    return true;
}();
} // namespace infinicore::op::argwhere_impl::infiniop
