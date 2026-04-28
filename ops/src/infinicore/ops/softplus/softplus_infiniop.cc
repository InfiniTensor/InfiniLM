#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/softplus.hpp"
#include <infiniop.h>

namespace infinicore::op::softplus_impl::infiniop {
thread_local common::OpCache<size_t, infiniopSoftplusDescriptor_t> caches(
    100, // capacity
    [](infiniopSoftplusDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySoftplusDescriptor(desc));
            desc = nullptr;
        }
    });
void calculate(Tensor y, Tensor x, float beta, float threshold) {
    size_t seed = hash_combine(y, x, beta, threshold);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopSoftplusDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSoftplusDescriptor(
            context::getInfiniopHandle(device), &desc,
            y->desc(), x->desc(), beta, threshold));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 获取并分配 Workspace
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSoftplusWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 执行计算
    INFINICORE_CHECK_ERROR(infiniopSoftplus(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}
static bool registered = []() {
    Softplus::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::softplus_impl::infiniop
