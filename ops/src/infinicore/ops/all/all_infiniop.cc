#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/all.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::all_impl::infiniop {

thread_local common::OpCache<size_t, infiniopAllDescriptor_t> caches(
    100, // capacity
    [](infiniopAllDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAllDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    size_t seed = hash_combine(output, input, dim.size(), keepdim);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAllDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAllDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc(), dim.data(), dim.size(), keepdim));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAllWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopAll(
        desc, workspace->data(), workspace_size,
        output->data(), input->data(), dim.data(), dim.size(), keepdim, context::getStream()));
}

static bool registered = []() {
    All::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::all_impl::infiniop
