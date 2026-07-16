#include "infinicore/ops/mha_kvcache.hpp"

#include "infinicore/adaptor/flash_attention_adaptor.hpp"

#include <stdexcept>

#ifdef ENABLE_FLASH_ATTN
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#if defined(ENABLE_METAX_API)
#define INFINICORE_FLASH_OP(name) ::name
#else
#define INFINICORE_FLASH_OP(name) flash::name
#endif

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
#ifdef ENABLE_FLASH_ATTN
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    // Paged KV caches must be contiguous for flash-attn; avoid extra copies for q/metadata when already dense.
    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor = infinicore::adaptor::to_aten_tensor(out_work);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
#elif defined(ENABLE_QY_API)
    Tensor k_cache_work = p->k_cache->contiguous();
    Tensor v_cache_work = p->v_cache->contiguous();
    auto k_cache = infinicore::adaptor::to_aten_tensor(k_cache_work);
    auto v_cache = infinicore::adaptor::to_aten_tensor(v_cache_work);
#endif
    auto seqlens_k = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(p->seqlens_k));
    auto block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

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

#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
    std::optional<at::Tensor> flash_attn_mars_ext = std::nullopt;
#endif

    auto result = INFINICORE_FLASH_OP(mha_fwd_kvcache)(
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
        0
#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
        ,
        flash_attn_mars_ext
#endif
    );

    if (use_dynamic_out) {
        out_tensor.copy_(result[0]);
    }
    if (out_need_copy_back) {
        p->out->copy_from(out_work);
    }
#else
    throw std::runtime_error("FlashAttention is not enabled in this build");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MhaKVCache, &plan, &run, &cleanup);

} // namespace infinicore::op::mha_kvcache_impl::flashattn
