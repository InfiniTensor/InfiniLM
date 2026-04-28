#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/floor_divide.hpp"
#include <infiniop.h>

namespace infinicore::op::floor_divide_impl::infiniop {

thread_local common::OpCache<size_t, infiniopFloorDivideDescriptor_t> caches(
    100, // capacity
    [](infiniopFloorDivideDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyFloorDivideDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor c, Tensor a, Tensor b) {
    size_t seed = hash_combine(c, b, a);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopFloorDivideDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateFloorDivideDescriptor(
            context::getInfiniopHandle(c->device()), &desc,
            c->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetFloorDivideWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopFloorDivide(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), context::getStream()));
}

static bool registered = []() {
    FloorDivide::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::floor_divide_impl::infiniop
