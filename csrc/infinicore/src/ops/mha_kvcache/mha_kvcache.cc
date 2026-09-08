#include "infinicore/ops/mha_kvcache.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MhaKVCache);

bool MhaKVCache::supports_device_graph_capture(
    const Tensor &out,
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &v_cache,
    const Tensor &seqlens_k,
    const Tensor &block_table,
    const std::optional<Tensor> &alibi_slopes) {
    if (!out
        || !q
        || !k_cache
        || !v_cache
        || !seqlens_k
        || !block_table
        || alibi_slopes.has_value()) {
        return false;
    }

    const auto device = out->device();
    const auto same_device = [&](const Tensor &tensor) {
        return tensor->device() == device;
    };
    if (out->ndim() != 4
        || q->ndim() != 4
        || k_cache->ndim() != 4
        || v_cache->ndim() != 4
        || seqlens_k->ndim() != 1
        || block_table->ndim() != 2) {
        return false;
    }

    // Keep graph capture limited to shapes validated with the persistent
    // InfiniOps provider. Other decode shapes retain the eager fallback.
    const bool p12_shape =
        q->shape() == Shape({1, 1, 32, 128})
        && out->shape() == Shape({1, 1, 32, 128})
        && k_cache->shape() == Shape({512, 256, 2, 128})
        && seqlens_k->shape() == Shape({1})
        && block_table->shape() == Shape({1, 8});
    const bool p13_shape =
        q->shape() == Shape({1, 1, 32, 128})
        && k_cache->size(1) == 256
        && k_cache->size(2) == 2
        && k_cache->size(3) == 128
        && seqlens_k->shape() == Shape({1})
        && block_table->shape() == Shape({1, 8});
    const bool p09_shape =
        q->shape() == Shape({16, 1, 24, 128})
        && k_cache->size(0) == 128
        && k_cache->size(1) == 256
        && k_cache->size(2) == 8
        && k_cache->size(3) == 128
        && seqlens_k->shape() == Shape({16})
        && block_table->shape() == Shape({16, 128});
    const bool p11_shape =
        q->shape() == Shape({1, 1, 16, 128})
        && k_cache->size(0) == 1
        && k_cache->size(1) == 256
        && k_cache->size(2) == 16
        && k_cache->size(3) == 128
        && seqlens_k->shape() == Shape({1})
        && block_table->shape() == Shape({1, 1});
    const bool p14_shape =
        q->shape() == Shape({16, 1, 32, 128})
        && k_cache->size(0) == 512
        && k_cache->size(1) == 256
        && k_cache->size(2) == 2
        && k_cache->size(3) == 128
        && seqlens_k->shape() == Shape({16})
        && block_table->shape() == Shape({16, 512});
    const bool p13_layout = q->is_contiguous() && out->is_contiguous();
    // P09 reads Q directly from the fused QKV projection while output is dense.
    const bool p09_layout =
        q->strides() == Strides({5120, 5120, 128, 1})
        && out->strides() == Strides({3072, 3072, 128, 1});
    // P11 has the same fused-QKV view pattern at batch size one.
    const bool p11_layout =
        q->strides() == Strides({6144, 6144, 128, 1})
        && out->strides() == Strides({2048, 2048, 128, 1});
    // P14 reads Q from MiniCPM4's fused QKV projection at batch size 16.
    const bool p14_layout =
        q->strides() == Strides({4608, 4608, 128, 1})
        && out->strides() == Strides({4096, 4096, 128, 1});
    // P12 has the same fused QKV projection layout at batch size one.
    const bool p12_layout =
        q->strides() == Strides({4608, 4608, 128, 1})
        && out->strides() == Strides({4096, 4096, 128, 1});
    const bool reviewed_shape_and_layout =
        (p12_shape && p12_layout)
        || (p13_shape && p13_layout)
        || (p09_shape && p09_layout)
        || (p11_shape && p11_layout)
        || (p14_shape && p14_layout);
    const auto dtype = q->dtype();
    return device.type() == Device::Type::kNvidia
        && same_device(q)
        && same_device(k_cache)
        && same_device(v_cache)
        && same_device(seqlens_k)
        && same_device(block_table)
        && reviewed_shape_and_layout
        && out->shape() == q->shape()
        && k_cache->size(0) > 0
        && v_cache->shape() == k_cache->shape()
        && (dtype == DataType::kFloat16
            || dtype == DataType::kBFloat16)
        && out->dtype() == dtype
        && k_cache->dtype() == dtype
        && v_cache->dtype() == dtype
        && seqlens_k->dtype() == DataType::kInt32
        && block_table->dtype() == DataType::kInt32
        && k_cache->is_contiguous()
        && v_cache->is_contiguous()
        && seqlens_k->is_contiguous()
        && block_table->is_contiguous();
}

MhaKVCache::MhaKVCache(Tensor out,
                       const Tensor &q,
                       const Tensor &k_cache,
                       const Tensor &v_cache,
                       const Tensor &seqlens_k,
                       const Tensor &block_table,
                       std::optional<Tensor> alibi_slopes,
                       float scale)
    : device_graph_capture_safe_(
          context::isGraphRecording()
          && supports_device_graph_capture(
              out,
              q,
              k_cache,
              v_cache,
              seqlens_k,
              block_table,
              alibi_slopes)) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, seqlens_k, block_table);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(),
                                 out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

void MhaKVCache::execute(Tensor out,
                         const Tensor &q,
                         const Tensor &k_cache,
                         const Tensor &v_cache,
                         const Tensor &seqlens_k,
                         const Tensor &block_table,
                         std::optional<Tensor> alibi_slopes,
                         float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MhaKVCache,
        out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

void mha_kvcache_(Tensor out,
                  const Tensor &q,
                  const Tensor &k_cache,
                  const Tensor &v_cache,
                  const Tensor &seqlens_k,
                  const Tensor &block_table,
                  std::optional<Tensor> alibi_slopes,
                  float scale) {
    MhaKVCache::execute(out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

Tensor mha_kvcache(const Tensor &q,
                   const Tensor &k_cache,
                   const Tensor &v_cache,
                   const Tensor &seqlens_k,
                   const Tensor &block_table,
                   std::optional<Tensor> alibi_slopes,
                   float scale) {
    // Output shape matches q: [batch_size, seqlen_q, num_heads, head_size]
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    mha_kvcache_(out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
    return out;
}

} // namespace infinicore::op
