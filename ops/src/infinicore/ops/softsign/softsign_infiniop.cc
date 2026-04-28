#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/softsign.hpp"
#include <infiniop.h>

namespace infinicore::op::softsign_impl::infiniop {
thread_local common::OpCache<size_t, infiniopSoftsignDescriptor_t> caches(
    100, // capacity
    [](infiniopSoftsignDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySoftsignDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopSoftsignDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSoftsignDescriptor(
            context::getInfiniopHandle(device), &desc,
            y->desc(), x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 获取工作空间大小
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSoftsignWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 执行计算
    INFINICORE_CHECK_ERROR(infiniopSoftsign(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}
static bool registered = []() {
    Softsign::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::softsign_impl::infiniop
