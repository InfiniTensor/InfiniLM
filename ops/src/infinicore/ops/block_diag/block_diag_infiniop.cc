#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/block_diag.hpp"
#include "infinicore/ops/common/cache.hpp"

#include <infiniop.h>

namespace infinicore::op::block_diag_impl::infiniop {

thread_local common::OpCache<size_t, infiniopBlockDiagDescriptor_t> caches(
    100,
    [](infiniopBlockDiagDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyBlockDiagDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, const std::vector<Tensor> &inputs) {
    if (inputs.empty()) {
        throw std::runtime_error("block_diag expects at least one input tensor");
    }

    size_t seed = 0;
    hash_combine(seed, output, static_cast<size_t>(inputs.size()));
    for (const auto &x : inputs) {
        hash_combine(seed, x);
    }

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopBlockDiagDescriptor_t desc = nullptr;

    std::vector<infiniopTensorDescriptor_t> input_descs;
    input_descs.reserve(inputs.size());
    for (const auto &x : inputs) {
        input_descs.push_back(x->desc());
    }

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateBlockDiagDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            input_descs.data(),
            input_descs.size()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetBlockDiagWorkspaceSize(desc, &workspace_size));
    auto workspace = context::allocateMemory(workspace_size);

    std::vector<const void *> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto &x : inputs) {
        input_ptrs.push_back(x->data());
    }

    INFINICORE_CHECK_ERROR(infiniopBlockDiag(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input_ptrs.data(),
        context::getStream()));
}

static bool registered = []() {
    BlockDiag::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::block_diag_impl::infiniop
