#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/kthvalue.hpp"
#include <infiniop.h>

namespace infinicore::op::kthvalue_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopKthvalueDescriptor_t> caches(
    100, // capacity
    [](infiniopKthvalueDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyKthvalueDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor values, Tensor indices, Tensor input, int64_t k, int64_t dim, bool keepdim) {
    size_t seed = hash_combine(values, indices, input, k, dim, keepdim);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopKthvalueDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 3. 创建描述符
        INFINICORE_CHECK_ERROR(infiniopCreateKthvalueDescriptor(
            context::getInfiniopHandle(input->device()),
            &desc,
            values->desc(),
            indices->desc(),
            input->desc(),
            static_cast<int>(k),
            static_cast<int>(dim),
            static_cast<int>(keepdim)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetKthvalueWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopKthvalue(
        desc,
        workspace->data(),
        workspace_size,
        values->data(),
        indices->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Kthvalue::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::kthvalue_impl::infiniop
