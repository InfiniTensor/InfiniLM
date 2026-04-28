#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/sinh.hpp"

#include <infiniop.h>

namespace infinicore::op::sinh_impl::infiniop {

thread_local common::OpCache<size_t, infiniopSinhDescriptor_t> caches(
    100,
    [](infiniopSinhDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySinhDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopSinhDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSinhDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSinhWorkspaceSize(desc, &workspace_size));
    auto workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopSinh(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Sinh::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::sinh_impl::infiniop
