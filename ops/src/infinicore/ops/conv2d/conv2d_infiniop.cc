#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/conv2d.hpp"
#include <infiniop.h>

namespace infinicore::op::conv2d_impl::infiniop {

thread_local common::OpCache<size_t, infiniopConvDescriptor_t> caches(
    100, // capacity
    [](infiniopConvDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyConvDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output,
               Tensor input,
               Tensor weight,
               Tensor bias,
               const size_t *pads,
               const size_t *strides,
               const size_t *dilations,
               size_t n) {
    size_t seed = hash_combine(output, input, weight, bias, n);
    for (size_t i = 0; i < n; ++i) {
        hash_combine(seed, pads[i], strides[i], dilations[i]);
    }

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopConvDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateConvDescriptor(
            context::getInfiniopHandle(device), &desc,
            output->desc(), input->desc(), weight->desc(),
            bias ? bias->desc() : nullptr,
            const_cast<size_t *>(pads),
            const_cast<size_t *>(strides),
            const_cast<size_t *>(dilations),
            n));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetConvWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopConv(
        desc, workspace->data(), workspace_size,
        output->data(),
        input->data(),
        weight->data(),
        bias ? bias->data() : nullptr,
        context::getStream()));
}

static bool registered = []() {
    Conv2d::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::conv2d_impl::infiniop
