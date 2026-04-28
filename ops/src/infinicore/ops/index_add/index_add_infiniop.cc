#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/index_add.hpp" // 引用算子定义
#include <infiniop.h>

namespace infinicore::op::index_add_impl::infiniop {

thread_local common::OpCache<size_t, infiniopIndexAddDescriptor_t> caches(
    100, // capacity
    [](infiniopIndexAddDescriptor_t &desc) {
        if (desc != nullptr) {
            // 销毁描述符
            INFINICORE_CHECK_ERROR(infiniopDestroyIndexAddDescriptor(desc));
            desc = nullptr;
        }
    });

// 计算函数实现
void calculate(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor source, float alpha) {
    size_t seed = hash_combine(output, input, dim, index, source, alpha);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopIndexAddDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateIndexAddDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            dim, // 传入 int64_t
            index->desc(),
            source->desc(),
            alpha)); // 传入 float
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 大小并分配
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetIndexAddWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);
    INFINICORE_CHECK_ERROR(infiniopIndexAdd(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        index->data(),
        source->data(),
        context::getStream()));
}

// 5. 注册算子到 Dispatcher
static bool registered = []() {
    IndexAdd::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::index_add_impl::infiniop
