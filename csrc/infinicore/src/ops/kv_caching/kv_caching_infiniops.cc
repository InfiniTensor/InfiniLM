#include "infinicore/ops/kv_caching.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/kv_caching_infinilm.h"

namespace infinicore::op::kv_caching_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    TensorMeta k_cache, v_cache, k, v, past_kv_lengths;
    graph::GraphTensor k_cache_tensor, v_cache_tensor, k_tensor, v_tensor, past_kv_lengths_tensor;
};
} // namespace

void *plan(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &past_kv_lengths) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(k_cache->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cache, v_cache, k, v, past_kv_lengths);
    return new PlannedMeta{TensorMeta(k_cache), TensorMeta(v_cache), TensorMeta(k), TensorMeta(v), TensorMeta(past_kv_lengths), graph::GraphTensor(k_cache), graph::GraphTensor(v_cache), graph::GraphTensor(k), graph::GraphTensor(v), graph::GraphTensor(past_kv_lengths)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::KvCachingInfinilm::Call(
        handle,
        config,
        planned->k.tensor(planned->k_tensor),
        planned->v.tensor(planned->v_tensor),
        planned->past_kv_lengths.tensor(planned->past_kv_lengths_tensor),
        planned->k_cache.tensor(planned->k_cache_tensor),
        planned->v_cache.tensor(planned->v_cache_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(KVCaching::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(KVCaching::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(KVCaching::cleanup_dispatcher(), &cleanup);
    return true;
}();
} // namespace infinicore::op::kv_caching_impl::infiniops
#endif
