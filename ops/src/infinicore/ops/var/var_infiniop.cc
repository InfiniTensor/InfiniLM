#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/var.hpp"
#include <infiniop.h>

namespace infinicore::op::var_impl::infiniop {

thread_local common::OpCache<size_t, infiniopVarDescriptor_t> caches(
    100, // capacity
    [](infiniopVarDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyVarDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor var_output, Tensor input, std::vector<size_t> dim, bool unbiased, bool keepdim) {
    size_t seed = hash_combine(var_output, input, dim.size(), unbiased, keepdim);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopVarDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateVarDescriptor(
            context::getInfiniopHandle(var_output->device()), &desc,
            var_output->desc(), input->desc(), dim.data(), dim.size(), unbiased, keepdim));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetVarWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopVar(
        desc, workspace->data(), workspace_size,
        var_output->data(), input->data(), dim.data(), dim.size(), unbiased, keepdim, context::getStream()));
}

static bool registered = []() {
    Var::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::var_impl::infiniop
