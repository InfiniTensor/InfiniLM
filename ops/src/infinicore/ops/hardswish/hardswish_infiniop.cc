#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/hardswish.hpp"
#include <infiniop.h>

namespace infinicore::op::hardswish_impl::infiniop {

thread_local common::OpCache<size_t, infiniopHardSwishDescriptor_t> caches(
    100,
    [](infiniopHardSwishDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyHardSwishDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopHardSwishDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateHardSwishDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetHardSwishWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size != 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopHardSwish(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Hardswish::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::hardswish_impl::infiniop
