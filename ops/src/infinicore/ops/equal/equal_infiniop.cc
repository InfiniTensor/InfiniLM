#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/equal.hpp"
#include <infiniop.h>

namespace infinicore::op::equal_impl::infiniop {

thread_local common::OpCache<size_t, infiniopEqualDescriptor_t> caches(
    100,
    [](infiniopEqualDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyEqualDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor a, Tensor b) {
    size_t seed = hash_combine(out, a, b);
    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    infiniopEqualDescriptor_t desc = nullptr;
    if (auto cached = cache.get(seed)) {
        desc = *cached;
    } else {
        INFINICORE_CHECK_ERROR(infiniopCreateEqualDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetEqualWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size != 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopEqual(
        desc,
        workspace_ptr,
        workspace_size,
        out->data(),
        a->data(),
        b->data(),
        context::getStream()));
}

static bool registered = []() {
    Equal::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::equal_impl::infiniop
