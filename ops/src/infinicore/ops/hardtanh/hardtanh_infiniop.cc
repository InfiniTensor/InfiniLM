#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/hardtanh.hpp"
#include <infiniop.h>

namespace infinicore::op::hardtanh_impl::infiniop {

thread_local common::OpCache<size_t, infiniopHardTanhDescriptor_t> caches(
    100,
    [](infiniopHardTanhDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyHardTanhDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, float min_val, float max_val) {
    size_t seed = hash_combine(output, input, min_val, max_val);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopHardTanhDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateHardTanhDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            input->desc(),
            min_val,
            max_val));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetHardTanhWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size != 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopHardTanh(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    HardTanh::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::hardtanh_impl::infiniop
