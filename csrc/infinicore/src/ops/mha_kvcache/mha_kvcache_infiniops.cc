#include "infinicore/ops/mha_kvcache.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/flash_attn_with_kvcache.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infinicore::op::mha_kvcache_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

bool is_supported(const Tensor &out,
                  const Tensor &q,
                  const Tensor &k_cache,
                  const Tensor &v_cache,
                  const Tensor &seqlens_k,
                  const Tensor &block_table,
                  const std::optional<Tensor> &alibi_slopes) {
    const auto dtype = q->dtype();
    const auto device_type = out->device().type();
    if ((device_type != Device::Type::kNvidia
         && device_type != Device::Type::kMetax
         && device_type != Device::Type::kMoore
         && device_type != Device::Type::kCambricon)
        || q->ndim() != 4
        || out->ndim() != 4
        || k_cache->ndim() != 4
        || v_cache->ndim() != 4
        || q->size(1) != 1
        || k_cache->shape() != v_cache->shape()
        || out->shape() != q->shape()
        || (dtype != DataType::kFloat16 && dtype != DataType::kBFloat16)
        || out->dtype() != dtype
        || k_cache->dtype() != dtype
        || v_cache->dtype() != dtype
        || q->size(0) == 0
        || q->size(2) == 0
        || k_cache->size(0) == 0
        || k_cache->size(1) == 0
        || k_cache->size(2) == 0
        || q->size(2) % k_cache->size(2) != 0
        || q->size(3) == 0
        || q->size(3) > 256
        || q->size(3) % 8 != 0
        || (device_type == Device::Type::kMoore
            && q->size(3) != 64
            && q->size(3) != 128)
        || q->size(3) != k_cache->size(3)
        || q->stride(3) != 1
        || out->stride(3) != 1
        || k_cache->stride(3) != 1
        || v_cache->stride(3) != 1
        || seqlens_k->ndim() != 1
        || seqlens_k->size(0) != q->size(0)
        || seqlens_k->dtype() != DataType::kInt32
        || !seqlens_k->is_contiguous()
        || block_table->ndim() != 2
        || block_table->size(0) != q->size(0)
        || block_table->dtype() != DataType::kInt32
        || !block_table->is_contiguous()
        || k_cache->size(1) % 256 != 0) {
        return false;
    }

    if (alibi_slopes
        && ((alibi_slopes.value()->ndim() != 1
             && alibi_slopes.value()->ndim() != 2)
            || (device_type == Device::Type::kMoore
                && alibi_slopes.value()->ndim() != 1)
            || alibi_slopes.value()->dtype() != DataType::kFloat32
            || !alibi_slopes.value()->is_contiguous()
            || alibi_slopes.value()->device() != out->device()
            || (alibi_slopes.value()->ndim() == 1
                && alibi_slopes.value()->size(0) != q->size(2))
            || (alibi_slopes.value()->ndim() == 2
                && (alibi_slopes.value()->size(0) != q->size(0)
                    || alibi_slopes.value()->size(1) != q->size(2))))) {
        return false;
    }

    return true;
}

struct PlannedMeta {
    TensorMeta out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<TensorMeta> alibi_slopes;
    graph::GraphTensor out_tensor, q_tensor, k_cache_tensor, v_cache_tensor,
        seqlens_k_tensor, block_table_tensor;
    std::optional<graph::GraphTensor> alibi_slopes_tensor;
    float scale;
};

} // namespace

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &seqlens_k,
           const Tensor &block_table,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    INFINICORE_ASSERT(is_supported(
        out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes));
    return new PlannedMeta{
        TensorMeta(out),
        TensorMeta(q),
        TensorMeta(k_cache),
        TensorMeta(v_cache),
        TensorMeta(seqlens_k),
        TensorMeta(block_table),
        alibi_slopes ? std::optional<TensorMeta>{TensorMeta(*alibi_slopes)}
                     : std::nullopt,
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(seqlens_k),
        graph::GraphTensor(block_table),
        alibi_slopes
            ? std::optional<graph::GraphTensor>{graph::GraphTensor(*alibi_slopes)}
            : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    const auto device_type = planned->q.device.type();
    const std::size_t implementation_index = device_type == infini::ops::Device::Type::kMoore ? 8 : 16;
    auto config = ::infinicore::op::infiniops::configForImplementation<
        infini::ops::FlashAttnWithKvcache>(device_type, implementation_index);

    const std::optional<infini::ops::Tensor> no_tensor;
    const std::optional<infini::ops::Tensor> cache_seqlens{
        planned->seqlens_k.tensor(planned->seqlens_k_tensor)};
    const std::optional<infini::ops::Tensor> block_table{
        planned->block_table.tensor(planned->block_table_tensor)};
    const std::optional<infini::ops::Tensor> alibi_slopes = planned->alibi_slopes
                                                              ? std::optional<infini::ops::Tensor>{
                                                                    planned->alibi_slopes->tensor(*planned->alibi_slopes_tensor)}
                                                              : std::nullopt;

    infini::ops::FlashAttnWithKvcache::Call(
        handle,
        config,
        planned->q.tensor(planned->q_tensor),
        planned->k_cache.tensor(planned->k_cache_tensor),
        planned->v_cache.tensor(planned->v_cache_tensor),
        no_tensor,
        no_tensor,
        no_tensor,
        no_tensor,
        cache_seqlens,
        no_tensor,
        no_tensor,
        block_table,
        alibi_slopes,
        std::optional<double>{planned->scale},
        true,
        std::vector<std::int64_t>{-1, -1},
        0.0,
        true,
        std::int64_t{0},
        false,
        planned->out.tensor(planned->out_tensor),
        no_tensor);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::kNvidia, &plan);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::kNvidia, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::kNvidia, &cleanup);
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::kMetax, &plan);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::kMetax, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::kMetax, &cleanup);
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::kMoore, &plan);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::kMoore, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::kMoore, &cleanup);
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::kCambricon, &plan);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::kCambricon, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::kCambricon, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_kvcache_impl::infiniops
#endif
