#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/masked_select.hpp"
#include <infiniop.h>

namespace infinicore::op::masked_select_impl::infiniop {

thread_local common::OpCache<size_t, infiniopMaskedSelectDescriptor_t> caches(
    100,
    [](infiniopMaskedSelectDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyMaskedSelectDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor input, Tensor mask, void **data_ptr, size_t *dlen_ptr) {
    size_t seed = hash_combine(input, mask, (std::uintptr_t)data_ptr, (std::uintptr_t)dlen_ptr);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopMaskedSelectDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateMaskedSelectDescriptor(
            context::getInfiniopHandle(input->device()), &desc,
            input->desc(), mask->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetMaskedSelectWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopMaskedSelect(
        desc, workspace->data(), workspace_size,
        input->data(), (const bool *)mask->data(), data_ptr, dlen_ptr, context::getStream()));
}

static bool registered = []() {
    MaskedSelect::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::masked_select_impl::infiniop
