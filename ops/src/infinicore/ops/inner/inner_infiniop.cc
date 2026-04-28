#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/inner.hpp"
#include <infiniop.h>

namespace infinicore::op::inner_impl::infiniop {

thread_local common::OpCache<size_t, infiniopInnerDescriptor_t> caches(
    100,
    [](infiniopInnerDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyInnerDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor input, Tensor other) {
    size_t seed = hash_combine(out, input, other);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopInnerDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateInnerDescriptor(
            context::getInfiniopHandle(input->device()), &desc,
            out->desc(), input->desc(), other->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetInnerWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopInner(
        desc, workspace->data(), workspace_size,
        out->data(), input->data(), other->data(), context::getStream()));
}

static bool registered = []() {
    Inner::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::inner_impl::infiniop
