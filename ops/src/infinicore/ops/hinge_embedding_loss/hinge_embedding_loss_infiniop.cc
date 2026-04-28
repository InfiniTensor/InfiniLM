#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/hinge_embedding_loss.hpp"

#include <infiniop.h>

namespace infinicore::op::hinge_embedding_loss_impl::infiniop {

thread_local common::OpCache<size_t, infiniopHingeEmbeddingLossDescriptor_t> caches(
    100,
    [](infiniopHingeEmbeddingLossDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyHingeEmbeddingLossDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, Tensor target, double margin, int reduction) {
    size_t seed = hash_combine(output, input, target, margin, reduction);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopHingeEmbeddingLossDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateHingeEmbeddingLossDescriptor(
            context::getInfiniopHandle(device),
            &desc,
            output->desc(),
            input->desc(),
            target->desc(),
            margin,
            reduction));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetHingeEmbeddingLossWorkspaceSize(desc, &workspace_size));
    auto workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopHingeEmbeddingLoss(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        input->data(),
        target->data(),
        context::getStream()));
}

static bool registered = []() {
    HingeEmbeddingLoss::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::hinge_embedding_loss_impl::infiniop
