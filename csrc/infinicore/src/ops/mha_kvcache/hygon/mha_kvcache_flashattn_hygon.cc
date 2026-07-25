#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)
#include "infinicore/ops/mha_kvcache.hpp"

#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/TensorIndexing.h>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include <c10/hip/HIPFunctions.h>
#include <c10/hip/HIPGuard.h>

namespace infinicore::op::mha_kvcache_impl::flashattn {

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &seqlens_k,
           const Tensor &block_table,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(seqlens_k),
        graph::GraphTensor(block_table),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    // Paged KV caches must be contiguous for flash-attn; avoid extra copies for q/metadata when already dense.
    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor = infinicore::adaptor::to_aten_tensor(out_work);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto seqlens_k = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(p->seqlens_k));
    auto block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

    if (std::getenv("INFINICORE_HYGON_ATEN_FALLBACK")) {
        namespace idx = at::indexing;
        auto seqlens_t = infinicore::adaptor::to_aten_tensor(p->seqlens_k);
        auto block_table_t = infinicore::adaptor::to_aten_tensor(p->block_table);
        auto seqlens_cpu = seqlens_t.to(at::kCPU);
        auto block_table_cpu = block_table_t.to(at::kCPU);

        auto result = at::empty_like(out_tensor);
        const int64_t batch_size = q.size(0);
        const int64_t seqlen_q = q.size(1);
        const int64_t num_heads = q.size(2);
        const int64_t block_size = k_cache.size(1);
        const int64_t num_kv_heads = k_cache.size(2);
        const int64_t group_size = num_heads / num_kv_heads;

        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const int64_t seq_len = seqlens_cpu.index({batch_idx}).item<int64_t>();
            std::vector<at::Tensor> keys;
            std::vector<at::Tensor> values;
            keys.reserve(seq_len);
            values.reserve(seq_len);
            for (int64_t logical_pos = 0; logical_pos < seq_len; ++logical_pos) {
                const int64_t block_id = block_table_cpu.index({batch_idx, logical_pos / block_size}).item<int64_t>();
                const int64_t off = logical_pos % block_size;
                keys.push_back(k_cache.index({block_id, off, idx::Slice(), idx::Slice()}));
                values.push_back(v_cache.index({block_id, off, idx::Slice(), idx::Slice()}));
            }
            auto K = at::stack(keys, 0);
            auto V = at::stack(values, 0);
            if (group_size > 1) {
                K = K.repeat_interleave(group_size, 1);
                V = V.repeat_interleave(group_size, 1);
            }
            auto cur_q = q.index({batch_idx});
            auto scores = at::matmul(cur_q.permute({1, 0, 2}).to(at::kFloat), K.permute({1, 2, 0}).to(at::kFloat)) * p->scale;
            auto mask = at::full({seqlen_q, seq_len}, -std::numeric_limits<float>::infinity(), q.options().dtype(at::kFloat));
            const int64_t prefix_len = seq_len - seqlen_q;
            for (int64_t query_pos = 0; query_pos < seqlen_q; ++query_pos) {
                mask.index_put_({query_pos, idx::Slice(0, prefix_len + query_pos + 1)}, 0.0);
            }
            auto attn = at::softmax(scores + mask.unsqueeze(0), -1).to(q.dtype());
            auto cur_out = at::matmul(attn, V.permute({1, 0, 2})).permute({1, 0, 2});
            result.index_put_({batch_idx}, cur_out);
        }

        out_tensor.copy_(result);
        if (out_need_copy_back) {
            p->out->copy_from(out_work);
        }
        return;
    }

    std::optional<const at::Tensor> k_new = std::nullopt;
    std::optional<const at::Tensor> v_new = std::nullopt;
    std::optional<const at::Tensor> rotary_cos = std::nullopt;
    std::optional<const at::Tensor> rotary_sin = std::nullopt;
    std::optional<const at::Tensor> cache_batch_idx = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    const bool use_dynamic_out = q.dim() == 4 && k_cache.dim() == 4
                              && q.size(1) == 1 && q.size(2) > k_cache.size(2)
                              && q.size(3) % 8 == 0 && !alibi_slopes.has_value();

    auto out = use_dynamic_out ? std::optional<at::Tensor>(std::nullopt)
                               : std::optional<at::Tensor>(out_tensor);

    auto result = flash::mha_fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out,
        p->scale,
        true,
        -1,
        -1,
        0.0f,
        false,
        0);

    if (!result.empty() && result[0].defined()) {
        out_tensor.copy_(result[0]);
    }
    if (out_need_copy_back) {
        p->out->copy_from(out_work);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::kHygon, &plan);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::kHygon, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::kHygon, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_kvcache_impl::flashattn
#endif // ENABLE_HYGON_API && ENABLE_FLASH_ATTN
