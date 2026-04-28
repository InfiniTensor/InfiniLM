#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/softmax.hpp"
#include <infiniop.h>

namespace infinicore::op::softmax_impl::infiniop {

thread_local common::OpCache<size_t, infiniopSoftmaxDescriptor_t> caches(
    100, // capacity
    [](infiniopSoftmaxDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySoftmaxDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, int axis) {
    size_t seed = hash_combine(output, input, axis);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopSoftmaxDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSoftmaxDescriptor(
            context::getInfiniopHandle(device), &desc,
            output->desc(), input->desc(), axis));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSoftmaxWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopSoftmax(
        desc, workspace->data(), workspace_size,
        output->data(), input->data(), context::getStream()));
}

static bool registered = []() {
    Softmax::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::softmax_impl::infiniop
