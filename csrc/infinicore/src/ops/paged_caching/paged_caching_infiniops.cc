#include "infinicore/ops/paged_caching.hpp"

#include <string>

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/reshape_and_cache_flash.h"

namespace infinicore::op::paged_caching_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    TensorMeta k, v, slot_mapping, scale, k_cache, v_cache;
    graph::GraphTensor k_tensor, v_tensor, slot_mapping_tensor, scale_tensor, k_cache_tensor, v_cache_tensor;
    Tensor scale_owner;
};
} // namespace

void *plan(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(k_cache->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cache, v_cache, k, v, slot_mapping);

    // The canonical API requires valid scales even though the "auto" path does
    // not apply quantization.
    auto scale = Tensor::empty({1}, DataType::kFloat32, k_cache->device());
    constexpr float one = 1.0f;
    context::memcpyH2D(scale->data(), &one, sizeof(one), false);
    return new PlannedMeta{
        TensorMeta(k), TensorMeta(v), TensorMeta(slot_mapping), TensorMeta(scale), TensorMeta(k_cache), TensorMeta(v_cache),
        graph::GraphTensor(k), graph::GraphTensor(v), graph::GraphTensor(slot_mapping), graph::GraphTensor(scale), graph::GraphTensor(k_cache), graph::GraphTensor(v_cache),
        scale};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::ReshapeAndCacheFlash::Call(
        handle,
        config,
        planned->k.tensor(planned->k_tensor),
        planned->v.tensor(planned->v_tensor),
        planned->slot_mapping.tensor(planned->slot_mapping_tensor),
        planned->scale.tensor(planned->scale_tensor),
        planned->scale.tensor(planned->scale_tensor),
        std::string{"auto"},
        planned->k_cache.tensor(planned->k_cache_tensor),
        planned->v_cache.tensor(planned->v_cache_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(PagedCaching::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(PagedCaching::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(PagedCaching::cleanup_dispatcher(), &cleanup);
    return true;
}();
} // namespace infinicore::op::paged_caching_impl::infiniops
#endif
