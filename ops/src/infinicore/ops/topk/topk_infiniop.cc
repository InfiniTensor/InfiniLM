#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/topk.hpp"
#include <infiniop.h>

namespace infinicore::op::topk_impl::infiniop {

thread_local common::OpCache<size_t, infiniopTopKDescriptor_t> caches(
    100, // capacity
    [](infiniopTopKDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyTopKDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor values_output, Tensor indices_output, Tensor input, size_t k, size_t dim, bool largest, bool sorted) {
    size_t seed = hash_combine(values_output, indices_output, input, k, dim, largest, sorted);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopTopKDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateTopKDescriptor(
            context::getInfiniopHandle(values_output->device()), &desc,
            values_output->desc(), indices_output->desc(), input->desc(), k, dim, largest, sorted));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetTopKWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopTopK(
        desc, workspace->data(), workspace_size,
        values_output->data(), indices_output->data(), input->data(), k, dim, largest, sorted, context::getStream()));
}

static bool registered = []() {
    TopK::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::topk_impl::infiniop
