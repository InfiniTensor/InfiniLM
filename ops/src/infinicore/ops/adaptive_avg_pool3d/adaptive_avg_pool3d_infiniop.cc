#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/adaptive_avg_pool3d.hpp"
#include "infinicore/ops/common/cache.hpp"

namespace infinicore::op::adaptive_avg_pool3d_impl::infiniop {

thread_local common::OpCache<size_t, infiniopAdaptiveAvgPool3DDescriptor_t> caches(
    100, // capacity
    [](infiniopAdaptiveAvgPool3DDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAdaptiveAvgPool3DDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAdaptiveAvgPool3DDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // Convert vector to array for output_size
        std::vector<size_t> output_size_vec = {output->size(2), output->size(3), output->size(4)};

        INFINICORE_CHECK_ERROR(infiniopCreateAdaptiveAvgPool3DDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            output_size_vec.data()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // Get workspace size and allocate if needed
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAdaptiveAvgPool3DWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(
        infiniopAdaptiveAvgPool3D(
            desc,
            workspace->data(), workspace_size,
            output->data(),
            input->data(),
            context::getStream()));
}

static bool registered = []() {
    AdaptiveAvgPool3D::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::adaptive_avg_pool3d_impl::infiniop
