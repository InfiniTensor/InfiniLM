#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/scatter.hpp"
#include <infiniop.h>

namespace infinicore::op::scatter_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopScatterDescriptor_t> caches(
    100, // capacity
    [](infiniopScatterDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyScatterDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor src, int64_t reduction) {
    // Scatter 算子输入 input, index, src 均为必须存在的 Tensor，直接参与 hash
    size_t seed = hash_combine(output, input, dim, index, src, reduction);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopScatterDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 3. 创建描述符
        // C++ Op 参数: output, input, dim, index, src, reduction
        // C API 参数: output, input, indices, updates, axis, reduction
        INFINICORE_CHECK_ERROR(infiniopCreateScatterDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            index->desc(), // 对应 C API indices
            src->desc(),   // 对应 C API updates
            static_cast<int>(dim),
            static_cast<int>(reduction)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetScatterWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopScatter(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        index->data(),
        src->data(),
        context::getStream()));
}

static bool registered = []() {
    Scatter::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::scatter_impl::infiniop
