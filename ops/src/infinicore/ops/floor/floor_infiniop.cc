#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/floor.hpp"
#include <infiniop.h>

namespace infinicore::op::floor_impl::infiniop {

thread_local common::OpCache<size_t, infiniopFloorDescriptor_t> caches(
    100, // capacity
    [](infiniopFloorDescriptor_t &desc) {
        if (desc != nullptr) {
            // 销毁 Floor 描述符
            INFINICORE_CHECK_ERROR(infiniopDestroyFloorDescriptor(desc));
            desc = nullptr;
        }
    });

// 计算函数实现
void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopFloorDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateFloorDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetFloorWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);
    INFINICORE_CHECK_ERROR(infiniopFloor(
        desc,
        workspace->data(), workspace_size,
        output->data(), input->data(), // 参数顺序通常是 Output, Input
        context::getStream()));
}

static bool registered = []() {
    Floor::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::floor_impl::infiniop
