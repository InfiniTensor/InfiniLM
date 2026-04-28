#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/index_copy.hpp"
#include <infiniop.h>

namespace infinicore::op::index_copy_impl::infiniop {

thread_local common::OpCache<size_t, infiniopIndexCopyDescriptor_t> caches(
    100,
    [](infiniopIndexCopyDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyIndexCopyDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor source) {
    size_t seed = hash_combine(output, input, dim, index, source);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopIndexCopyDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateIndexCopyDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            dim,
            index->desc(),
            source->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetIndexCopyWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopIndexCopy(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        index->data(),
        source->data(),
        context::getStream()));
}

static bool registered = []() {
    IndexCopy::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::index_copy_impl::infiniop
