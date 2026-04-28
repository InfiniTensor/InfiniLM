#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/var_mean.hpp"
#include <infiniop.h>

// todo 实现需要修改calculate函数

namespace infinicore::op::var_mean_impl::infiniop {

thread_local common::OpCache<size_t, infiniopVarMeanDescriptor_t> caches(
    100, // capacity
    [](infiniopVarMeanDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyVarMeanDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor var_output, Tensor mean_output, Tensor input, std::vector<size_t> dim, bool unbiased, bool keepdim) {
    size_t seed = hash_combine(var_output, mean_output, input, dim.size(), unbiased, keepdim);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopVarMeanDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateVarMeanDescriptor(
            context::getInfiniopHandle(var_output->device()), &desc,
            var_output->desc(), mean_output->desc(), input->desc(), dim.data(), dim.size(), unbiased, keepdim));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetVarMeanWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopVarMean(
        desc, workspace->data(), workspace_size,
        var_output->data(), mean_output->data(), input->data(), dim.data(), dim.size(), unbiased, keepdim, context::getStream()));
}

static bool registered = []() {
    Var_Mean::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::var_mean_impl::infiniop
