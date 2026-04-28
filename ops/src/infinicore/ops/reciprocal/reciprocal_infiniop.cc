#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/reciprocal.hpp"
#include <infiniop.h>

namespace infinicore::op::reciprocal_impl::infiniop {

thread_local common::OpCache<size_t, infiniopReciprocalDescriptor_t> caches(
    100, // capacity
    [](infiniopReciprocalDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyReciprocalDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopReciprocalDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateReciprocalDescriptor(
            context::getInfiniopHandle(device), &desc,
            y->desc(), x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetReciprocalWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopReciprocal(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}

static bool registered = []() {
    Reciprocal::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::reciprocal_impl::infiniop
