#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/upsample_bilinear.hpp"
#include <infiniop.h>

namespace infinicore::op::upsample_bilinear_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopUpsampleBilinearDescriptor_t> caches(
    100, // capacity
    [](infiniopUpsampleBilinearDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyUpsampleBilinearDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, bool align_corners) {
    // 1. 计算 Hash Seed
    // align_corners 是 bool，可以直接参与 hash
    size_t seed = hash_combine(output, input, align_corners);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopUpsampleBilinearDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 3. 创建描述符
        // 注意：C 接口中 align_corners 通常用 int 传递
        INFINICORE_CHECK_ERROR(infiniopCreateUpsampleBilinearDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            static_cast<int>(align_corners)));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetUpsampleBilinearWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopUpsampleBilinear(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    UpsampleBilinear::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::upsample_bilinear_impl::infiniop
