#include "infinicore/ops/mha_varlen.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#ifdef ENABLE_FLASH_ATTN
#include "infinicore/adaptor/flash_attention_adaptor.hpp"
#endif

#include <stdexcept>

namespace infinicore::op::mha_varlen_impl::flashattn {

struct PlannedMeta {
    graph::GraphTensor out, q, k, v, cum_seqlens_q, cum_seqlens_k;
    std::optional<graph::GraphTensor> block_table;
    int max_seqlen_q, max_seqlen_k;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &cum_seqlens_q,
           const Tensor &cum_seqlens_k,
           std::optional<Tensor> block_table,
           int max_seqlen_q,
           int max_seqlen_k,
           std::optional<Tensor> alibi_slopes,
           float scale) {

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(cum_seqlens_q),
        graph::GraphTensor(cum_seqlens_k),
        block_table ? std::optional<graph::GraphTensor>(graph::GraphTensor(*block_table)) : std::nullopt,
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

namespace {

#ifdef ENABLE_FLASH_ATTN
// MetaX/hpcc pip `flash_attn_2_cuda` exports `mha_varlen_fwd` at global scope (no namespace),
// while NVIDIA `flash-attn-nvidia.so` uses `flash::mha_varlen_fwd`.
#if defined(ENABLE_METAX_API)
#define INFINICORE_FLASH_OP(name) ::name
#else
#define INFINICORE_FLASH_OP(name) flash::name
#endif

#endif // ENABLE_FLASH_ATTN
} // namespace

void run(void *planned_meta) {
#ifndef ENABLE_ATEN
    (void)planned_meta;
    throw std::runtime_error("ATen is not enabled in this build");
#else
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);

    auto cu_seqlens_q = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_q);
    auto cu_seqlens_kv = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_k);

    const bool dense_sdpa = !p->block_table.has_value()
                         && !p->alibi_slopes.has_value()
                         && q.dim() == 3 && k.dim() == 3 && v.dim() == 3
                         && p->max_seqlen_q > 0 && p->max_seqlen_k > 0
                         && p->max_seqlen_q == p->max_seqlen_k
                         && cu_seqlens_q.dim() == 1
                         && cu_seqlens_q.size(0) == cu_seqlens_kv.size(0)
                         && q.size(0) == (cu_seqlens_q.size(0) - 1) * p->max_seqlen_q
                         && k.size(0) == (cu_seqlens_kv.size(0) - 1) * p->max_seqlen_k
                         && ((q.size(2) > 256) || (v.size(2) != q.size(2)));
    if (dense_sdpa) {
        const int64_t batch_size = cu_seqlens_q.size(0) - 1;
        const int64_t seqlen = p->max_seqlen_q;
        const int64_t num_heads = q.size(1);
        const int64_t num_kv_heads = k.size(1);
        const int64_t head_dim = q.size(2);
        const int64_t value_dim = v.size(2);
        auto q_4d = q.reshape({batch_size, seqlen, num_heads, head_dim}).permute({0, 2, 1, 3});
        auto k_4d = k.reshape({batch_size, seqlen, num_kv_heads, head_dim}).permute({0, 2, 1, 3});
        auto v_4d = v.reshape({batch_size, seqlen, num_kv_heads, value_dim}).permute({0, 2, 1, 3});
        if (num_heads != num_kv_heads) {
            if (num_heads % num_kv_heads != 0) {
                throw std::runtime_error("mha_varlen dense SDPA fallback requires num_heads to be divisible by num_kv_heads");
            }
            const int64_t groups = num_heads / num_kv_heads;
            k_4d = k_4d.unsqueeze(2).expand({batch_size, num_kv_heads, groups, seqlen, head_dim}).reshape({batch_size, num_heads, seqlen, head_dim});
            v_4d = v_4d.unsqueeze(2).expand({batch_size, num_kv_heads, groups, seqlen, value_dim}).reshape({batch_size, num_heads, seqlen, value_dim});
        }
        auto result = at::scaled_dot_product_attention(
            q_4d,
            k_4d,
            v_4d,
            std::nullopt,
            0.0,
            true,
            std::optional<double>(static_cast<double>(p->scale)));
        out_work.copy_(result.permute({0, 2, 1, 3}).reshape({q.size(0), num_heads, value_dim}));
        if (out_need_copy_back) {
            p->out->copy_from(out_work_ic);
        }
        return;
    }

#ifdef ENABLE_FLASH_ATTN
    auto out = std::optional<at::Tensor>(out_work);
    std::optional<at::Tensor> seqused_k = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    auto block_table = p->block_table ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->block_table)) : std::nullopt;
    auto max_seqlen_q = p->max_seqlen_q;
    auto max_seqlen_k = p->max_seqlen_k;
    auto alibi_slopes = p->alibi_slopes ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes)) : std::nullopt;
    auto scale = p->scale;

#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
    std::optional<at::Tensor> flash_attn_mars_ext = std::nullopt;
#endif

    INFINICORE_FLASH_OP(mha_varlen_fwd)
    (
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        scale,
        false,
        true,
        -1,
        -1,
        0.0,
        false,
        std::nullopt
#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
        ,
        flash_attn_mars_ext
#endif
    );

    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }

#else
    throw std::runtime_error("FlashAttention is not enabled in this build and dense SDPA fallback is not applicable");
#endif
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MultiheadAttentionVarlen, &plan, &run, &cleanup);

} // namespace infinicore::op::mha_varlen_impl::flashattn
