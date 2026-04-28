#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/take.hpp"
#include <infiniop.h>

namespace infinicore::op::take_impl::infiniop {

thread_local common::OpCache<size_t, infiniopTakeDescriptor_t> caches(
    100, // capacity
    [](infiniopTakeDescriptor_t &desc) {
        if (desc != nullptr) {
            // 销毁 Take 描述符
            INFINICORE_CHECK_ERROR(infiniopDestroyTakeDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, Tensor indices) {
    size_t seed = hash_combine(output, input, indices);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopTakeDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符：传入 input 和 indices 的 descriptor
        INFINICORE_CHECK_ERROR(infiniopCreateTakeDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc(), indices->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetTakeWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 3. 执行计算：传入 input 和 indices 的数据指针
    INFINICORE_CHECK_ERROR(infiniopTake(
        desc,
        workspace->data(), workspace_size,
        output->data(), input->data(), indices->data(),
        context::getStream()));
}

static bool registered = []() {
    // 注册到 Take 的 dispatcher
    Take::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::take_impl::infiniop
