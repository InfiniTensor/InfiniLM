#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/log_softmax.hpp"
#include <infiniop.h>

namespace infinicore::op::log_softmax_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopLogSoftmaxDescriptor_t> caches(
    100, // capacity
    [](infiniopLogSoftmaxDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyLogSoftmaxDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, int64_t dim) {
    size_t seed = hash_combine(output, input, dim);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopLogSoftmaxDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 3. 创建描述符
        INFINICORE_CHECK_ERROR(infiniopCreateLogSoftmaxDescriptor(
            context::getInfiniopHandle(input->device()),
            &desc,
            output->desc(),
            input->desc(),
            static_cast<int>(dim)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLogSoftmaxWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLogSoftmax(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    LogSoftmax::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::log_softmax_impl::infiniop
