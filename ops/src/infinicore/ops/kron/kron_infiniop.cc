#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/kron.hpp"

#include <infiniop.h>

namespace infinicore::op::kron_impl::infiniop {

thread_local common::OpCache<size_t, infiniopKronDescriptor_t> caches(
    100,
    [](infiniopKronDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyKronDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor a, Tensor b) {
    size_t seed = hash_combine(output, a, b);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopKronDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateKronDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            a->desc(),
            b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetKronWorkspaceSize(desc, &workspace_size));
    auto workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopKron(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        a->data(),
        b->data(),
        context::getStream()));
}

static bool registered = []() {
    Kron::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::kron_impl::infiniop
