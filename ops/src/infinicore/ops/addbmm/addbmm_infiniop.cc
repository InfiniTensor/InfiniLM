#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/addbmm.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <vector>

namespace infinicore::op::addbmm_impl::infiniop {

struct AddbmmContext {
    infiniopAddbmmDescriptor_t desc = nullptr;
    std::shared_ptr<Memory> workspace_buf = nullptr;
    size_t workspace_size = 0;

    void *getWorkspacePtr() const {
        return workspace_buf ? workspace_buf->data() : nullptr;
    }
};

thread_local common::OpCache<size_t, AddbmmContext> caches(
    256,
    [](AddbmmContext &ctx) {
        if (ctx.desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAddbmmDescriptor(ctx.desc));
            ctx.desc = nullptr;
        }
        ctx.workspace_buf = nullptr;
    });

inline size_t compute_key(const Tensor &output, const Tensor &input,
                          const Tensor &batch1, const Tensor &batch2,
                          float beta, float alpha) {
    size_t seed = 0;
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(output.operator->()));
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(input.operator->()));
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(batch1.operator->()));
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(batch2.operator->()));
    infinicore::hash_combine(seed, beta);
    infinicore::hash_combine(seed, alpha);
    return seed;
}

void calculate(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {
    size_t seed = compute_key(output, input, batch1, batch2, beta, alpha);

    static thread_local size_t last_seed = 0;
    static thread_local bool last_ctx_valid = false;
    static thread_local AddbmmContext last_ctx;

    AddbmmContext *ctx_ptr = nullptr;

    if (last_ctx_valid && seed == last_seed) {
        ctx_ptr = &last_ctx;
    } else {
        auto device_type = context::getDevice().getType();
        auto device_index = context::getDevice().getIndex();
        auto &cache = caches.getCache(device_type, device_index);

        auto opt_ctx = cache.get(seed);
        if (opt_ctx) {
            last_ctx = *opt_ctx;
        } else {
            AddbmmContext new_ctx;

            INFINICORE_CHECK_ERROR(infiniopCreateAddbmmDescriptor(
                context::getInfiniopHandle(output->device()),
                &new_ctx.desc,
                output->desc(),
                input->desc(),
                batch1->desc(),
                batch2->desc(),
                alpha,
                beta));

            INFINICORE_CHECK_ERROR(infiniopGetAddbmmWorkspaceSize(new_ctx.desc, &new_ctx.workspace_size));

            if (new_ctx.workspace_size > 0) {
                new_ctx.workspace_buf = context::allocateMemory(new_ctx.workspace_size);
            }

            cache.put(seed, new_ctx);
            last_ctx = new_ctx;
        }

        last_seed = seed;
        last_ctx_valid = true;
        ctx_ptr = &last_ctx;
    }

    INFINICORE_CHECK_ERROR(infiniopAddbmm(
        ctx_ptr->desc,
        ctx_ptr->getWorkspacePtr(),
        ctx_ptr->workspace_size,
        output->data(),
        input->data(),
        batch1->data(),
        batch2->data(),
        context::getStream()));
}

static bool registered = []() {
    Addbmm::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::addbmm_impl::infiniop
