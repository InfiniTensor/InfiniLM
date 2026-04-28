#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/lerp.hpp"
#include <infiniop.h>

namespace infinicore::op::lerp_impl::infiniop {

thread_local common::OpCache<size_t, infiniopLerpDescriptor_t> caches(
    100,
    [](infiniopLerpDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyLerpDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor start, Tensor end, Tensor weight) {
    size_t seed = hash_combine(output, start, end, weight);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);
    auto desc_opt = cache.get(seed);

    infiniopLerpDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLerpDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            start->desc(),
            end->desc(),
            weight->desc(),
            0.0f));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLerpWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLerp(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        start->data(),
        end->data(),
        weight->data(),
        context::getStream()));
}

void calculate(Tensor output, Tensor start, Tensor end, float weight) {
    size_t seed = hash_combine(output, start, end, weight);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);
    auto desc_opt = cache.get(seed);

    infiniopLerpDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLerpDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            start->desc(),
            end->desc(),
            nullptr,
            weight));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLerpWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLerp(
        desc,
        workspace->data(),
        workspace_size,
        output->data(),
        start->data(),
        end->data(),
        nullptr,
        context::getStream()));
}

static bool registered = []() {
    using SchemaTensor = void (*)(Tensor, Tensor, Tensor, Tensor);
    Lerp::dispatcher<SchemaTensor>().registerAll(
        static_cast<SchemaTensor>(&calculate),
        false);

    using SchemaScalar = void (*)(Tensor, Tensor, Tensor, float);
    Lerp::dispatcher<SchemaScalar>().registerAll(
        static_cast<SchemaScalar>(&calculate),
        false);

    return true;
}();

} // namespace infinicore::op::lerp_impl::infiniop
