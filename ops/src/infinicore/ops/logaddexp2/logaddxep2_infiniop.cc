#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/logaddexp2.hpp"
#include <infiniop.h>

namespace infinicore::op::logaddexp2_impl::infiniop {
thread_local common::OpCache<size_t, infiniopLogAddExp2Descriptor_t> caches(
    100, // capacity
    [](infiniopLogAddExp2Descriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyLogAddExp2Descriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor c, Tensor a, Tensor b) {
    size_t seed = hash_combine(c, a, b);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopLogAddExp2Descriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLogAddExp2Descriptor(
            context::getInfiniopHandle(device), &desc,
            c->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLogAddExp2WorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLogAddExp2(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), context::getStream()));
}

static bool registered = []() {
    LogAddExp2::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::logaddexp2_impl::infiniop
