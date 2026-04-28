#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/adaptive_avg_pool1d.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <vector>

namespace infinicore::op::adaptive_avg_pool1d_impl::infiniop {

// 1. 资源上下文
struct AdaptiveAvgPool1dContext {
    infiniopAdaptiveAvgPool1dDescriptor_t desc = nullptr;
    std::shared_ptr<Memory> workspace_buf = nullptr;
    size_t workspace_size = 0;

    void *getWorkspacePtr() const {
        return workspace_buf ? workspace_buf->data() : nullptr;
    }
};

// 2. 缓存定义
thread_local common::OpCache<size_t, AdaptiveAvgPool1dContext> caches(
    256,
    [](AdaptiveAvgPool1dContext &ctx) {
        if (ctx.desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAdaptiveAvgPool1dDescriptor(ctx.desc));
            ctx.desc = nullptr;
        }
        ctx.workspace_buf = nullptr;
    });

// 3. 核心计算函数
void calculate(Tensor output, Tensor input) {
    size_t seed = reinterpret_cast<size_t>(input.operator->());
    if (output->ndim() >= 3) {
        seed ^= (output->shape()[2] << 1);
    }

    static thread_local size_t last_seed = 0;
    static thread_local bool last_ctx_valid = false;
    static thread_local AdaptiveAvgPool1dContext last_ctx;

    AdaptiveAvgPool1dContext *active_ctx = nullptr;

    if (last_ctx_valid && seed == last_seed) {
        active_ctx = &last_ctx;
    } else {
        auto device_type = context::getDevice().getType();
        auto device_index = context::getDevice().getIndex();
        auto &cache = caches.getCache(device_type, device_index);

        auto opt_ctx = cache.get(seed);
        if (opt_ctx) {
            last_ctx = *opt_ctx;
        } else {
            AdaptiveAvgPool1dContext new_ctx;

            INFINICORE_CHECK_ERROR(infiniopCreateAdaptiveAvgPool1dDescriptor(
                context::getInfiniopHandle(output->device()),
                &new_ctx.desc,
                output->desc(),
                input->desc()));

            INFINICORE_CHECK_ERROR(infiniopGetAdaptiveAvgPool1dWorkspaceSize(new_ctx.desc, &new_ctx.workspace_size));

            if (new_ctx.workspace_size > 0) {
                new_ctx.workspace_buf = context::allocateMemory(new_ctx.workspace_size);
            }

            cache.put(seed, new_ctx);
            last_ctx = new_ctx;
        }

        last_seed = seed;
        last_ctx_valid = true;
        active_ctx = &last_ctx;
    }

    INFINICORE_CHECK_ERROR(infiniopAdaptiveAvgPool1d(
        active_ctx->desc,
        active_ctx->getWorkspacePtr(),
        active_ctx->workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

// 注册
static bool registered = []() {
    AdaptiveAvgPool1d::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::adaptive_avg_pool1d_impl::infiniop
