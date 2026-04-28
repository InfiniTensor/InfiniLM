#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"

#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/cross_entropy.hpp"

#include <infiniop.h>

namespace infinicore::op::cross_entropy_impl::infiniop {

thread_local common::OpCache<size_t, infiniopCrossEntropyDescriptor_t> caches(
    100,
    [](infiniopCrossEntropyDescriptor_t &desc) {
        if (desc != nullptr) {

            INFINICORE_CHECK_ERROR(infiniopDestroyCrossEntropyDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, Tensor target) {

    size_t seed = hash_combine(output, input, target);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopCrossEntropyDescriptor_t desc = nullptr;

    if (!desc_opt) {

        INFINICORE_CHECK_ERROR(infiniopCreateCrossEntropyDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            input->desc(),
            target->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetCrossEntropyWorkspaceSize(desc, &workspace_size));

    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopCrossEntropy(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        target->data(),
        context::getStream()));
}

static bool registered = []() {
    CrossEntropy::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::cross_entropy_impl::infiniop
