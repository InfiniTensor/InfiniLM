#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/avg_pool1d.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::avg_pool1d_impl::infiniop {

thread_local common::OpCache<size_t, infiniopAvgPool1dDescriptor_t> caches(
    100,
    [](infiniopAvgPool1dDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAvgPool1dDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(
    Tensor output,
    Tensor input,
    size_t kernel_size,
    size_t stride,
    size_t padding) {

    if (stride == 0) {
        stride = kernel_size;
    }

    size_t seed = hash_combine(output, input, kernel_size, stride, padding);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopAvgPool1dDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAvgPool1dDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            input->desc(),
            kernel_size,
            stride,
            padding));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAvgPool1dWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopAvgPool1d(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    AvgPool1d::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::avg_pool1d_impl::infiniop
