#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/addr.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::addr_impl::infiniop {

thread_local common::OpCache<size_t, infiniopAddrDescriptor_t> caches(
    100, // capacity
    [](infiniopAddrDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAddrDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor input, Tensor vec1, Tensor vec2, float beta, float alpha) {
    // Hash the inputs including beta and alpha
    size_t seed = hash_combine(out, input, vec1, vec2, beta, alpha);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAddrDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAddrDescriptor(
            context::getInfiniopHandle(out->device()), &desc,
            out->desc(), input->desc(), vec1->desc(), vec2->desc(), beta, alpha));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAddrWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopAddr(
        desc, workspace->data(), workspace_size,
        out->data(), input->data(), vec1->data(), vec2->data(), context::getStream()));
}

static bool registered = []() {
    Addr::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::addr_impl::infiniop
