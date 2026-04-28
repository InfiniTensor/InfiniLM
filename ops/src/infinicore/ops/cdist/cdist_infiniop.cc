#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/cdist.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::cdist_impl::infiniop {

// 定义线程局部的 cdist 算子描述符缓存
// 缓存 key 为输入 Tensor 描述信息及参数 p 的哈希值
thread_local common::OpCache<size_t, infiniopCdistDescriptor_t> caches(
    100,
    [](infiniopCdistDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyCdistDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor x1, Tensor x2, double p) {
    // 1. 生成唯一 Hash Seed 用于缓存查找
    size_t seed = hash_combine(out, x1, x2, p);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopCdistDescriptor_t desc = nullptr;

    // 2. 如果缓存未命中，创建新的描述符并存入缓存
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateCdistDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            out->desc(),
            x1->desc(),
            x2->desc(),
            p));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 动态获取并分配 Workspace 临时内存
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetCdistWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 4. 执行底层算子
    INFINICORE_CHECK_ERROR(infiniopCdist(
        desc,
        workspace->data(),
        workspace_size,
        out->data(),
        x1->data(),
        x2->data(),
        context::getStream()));
}

// 5. 自动注册到调度器 (Dispatcher)
static bool registered = []() {
    Cdist::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::cdist_impl::infiniop
