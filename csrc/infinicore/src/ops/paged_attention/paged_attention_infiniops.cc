#include "infinicore/ops/paged_attention.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/paged_attention_infinilm.h"

#include <cstddef>
#include <optional>

namespace infinicore::op::paged_attention_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

constexpr std::size_t kMaxPagedAttentionSplits = 8;

std::size_t WorkspaceSizeInBytes(const Tensor &q) {
    return kMaxPagedAttentionSplits
         * static_cast<std::size_t>(q->size(0))
         * static_cast<std::size_t>(q->size(1))
         * static_cast<std::size_t>(q->size(2) + 2)
         * sizeof(float);
}

struct PlannedMeta {
    TensorMeta out, q, k_cache, v_cache, block_tables, cache_lens;
    std::optional<TensorMeta> alibi_slopes;
    graph::GraphTensor workspace, out_tensor, q_tensor, k_cache_tensor, v_cache_tensor, block_tables_tensor, cache_lens_tensor;
    std::optional<graph::GraphTensor> alibi_slopes_tensor;
    float scale;
};
} // namespace

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &block_tables,
           const Tensor &cache_lens,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(out->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, cache_lens);
    if (alibi_slopes) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, *alibi_slopes);
    }
    return new PlannedMeta{
        TensorMeta(out), TensorMeta(q), TensorMeta(k_cache), TensorMeta(v_cache), TensorMeta(block_tables), TensorMeta(cache_lens),
        alibi_slopes ? std::optional<TensorMeta>{TensorMeta(*alibi_slopes)} : std::nullopt,
        graph::GraphTensor(Tensor::empty({WorkspaceSizeInBytes(q)}, DataType::kUInt8, out->device())),
        graph::GraphTensor(out), graph::GraphTensor(q), graph::GraphTensor(k_cache), graph::GraphTensor(v_cache), graph::GraphTensor(block_tables), graph::GraphTensor(cache_lens),
        alibi_slopes ? std::optional<graph::GraphTensor>{graph::GraphTensor(*alibi_slopes)} : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    handle.set_workspace(planned->workspace->data());
    handle.set_workspace_size_in_bytes(planned->workspace->numel());
    infini::ops::Config config;
    infini::ops::PagedAttentionInfinilm::Call(
        handle,
        config,
        planned->q.tensor(planned->q_tensor),
        planned->k_cache.tensor(planned->k_cache_tensor),
        planned->v_cache.tensor(planned->v_cache_tensor),
        planned->block_tables.tensor(planned->block_tables_tensor),
        planned->cache_lens.tensor(planned->cache_lens_tensor),
        planned->alibi_slopes ? std::optional<infini::ops::Tensor>{planned->alibi_slopes->tensor(planned->alibi_slopes_tensor.value()->data())} : std::nullopt,
        planned->scale,
        planned->out.tensor(planned->out_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(PagedAttention::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(PagedAttention::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(PagedAttention::cleanup_dispatcher(), &cleanup);
    return true;
}();
} // namespace infinicore::op::paged_attention_impl::infiniops
#endif
