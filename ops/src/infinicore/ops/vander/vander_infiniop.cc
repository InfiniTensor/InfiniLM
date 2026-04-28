#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/vander.hpp"
#include <infiniop.h>

namespace infinicore::op::vander_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopVanderDescriptor_t> caches(
    100, // capacity
    [](infiniopVanderDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyVanderDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, int64_t N, bool increasing) {
    // 1. 计算 Hash Key
    // 直接组合 output, input 以及标量参数 N, increasing
    size_t seed = hash_combine(output, input, N, increasing);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopVanderDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符
        // 注意：将 int64_t N 转换为 int，bool increasing 转换为 int 以匹配 C API 签名
        INFINICORE_CHECK_ERROR(infiniopCreateVanderDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            static_cast<int>(N),
            static_cast<int>(increasing)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetVanderWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopVander(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

// 4. 注册算子实现
static bool registered = []() {
    Vander::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::vander_impl::infiniop
