#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/addcmul.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::addcmul_impl::infiniop {

// 定义线程局部的算子描述符缓存
thread_local common::OpCache<size_t, infiniopAddcmulDescriptor_t> caches(
    100,
    [](infiniopAddcmulDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAddcmulDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor input, Tensor t1, Tensor t2, float value) {
    size_t seed = hash_combine(out, input, t1, t2, value);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopAddcmulDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAddcmulDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(), input->desc(), t1->desc(), t2->desc(), value));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAddcmulWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopAddcmul(
        desc, workspace->data(), workspace_size,
        out->data(), input->data(), t1->data(), t2->data(), context::getStream()));
}

static bool registered = []() {
    Addcmul::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::addcmul_impl::infiniop
