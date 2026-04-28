#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/adaptive_max_pool1d.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::adaptive_max_pool1d_impl::infiniop {

thread_local common::OpCache<size_t, infiniopAdaptiveMaxPool1dDescriptor_t> caches(
    100, // capacity
    [](infiniopAdaptiveMaxPool1dDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAdaptiveMaxPool1dDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x, size_t out) {
    size_t seed = hash_combine(y, x, out);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAdaptiveMaxPool1dDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAdaptiveMaxPool1dDescriptor(
            context::getInfiniopHandle(y->device()), &desc,
            y->desc(), x->desc(), out));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAdaptiveMaxPool1dWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopAdaptiveMaxPool1d(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}

static bool registered = []() {
    AdaptiveMaxPool1d::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::adaptive_max_pool1d_impl::infiniop
