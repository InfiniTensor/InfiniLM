#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)
#include "infinicore/ops/mha_varlen.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/hip/HIPFunctions.h>
#include <c10/hip/HIPGuard.h>
#endif

#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"

#include <ATen/TensorIndexing.h>
#include <cstdlib>
#include <limits>
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

void run(void *planned_meta) {
#ifndef ENABLE_ATEN
    (void)planned_meta;
    throw std::runtime_error("ATen is not enabled in this build");
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
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

    if (std::getenv("INFINICORE_HYGON_ATEN_FALLBACK")) {
        namespace idx = at::indexing;
        auto cu_q_cpu = cu_seqlens_q.to(at::kCPU);
        auto cu_k_cpu = cu_seqlens_kv.to(at::kCPU);
        const int64_t num_seqs = cu_q_cpu.size(0) - 1;
        auto result = at::zeros_like(out_work);

        if (!p->block_table.has_value()) {
            for (int64_t i = 0; i < num_seqs; ++i) {
                const int64_t q_start = cu_q_cpu.index({i}).item<int64_t>();
                const int64_t q_end = cu_q_cpu.index({i + 1}).item<int64_t>();
                const int64_t k_start = cu_k_cpu.index({i}).item<int64_t>();
                const int64_t k_end = cu_k_cpu.index({i + 1}).item<int64_t>();
                auto cur_q = q.index({idx::Slice(q_start, q_end)}).unsqueeze(0).transpose(1, 2);
                auto cur_k = k.index({idx::Slice(k_start, k_end)}).unsqueeze(0).transpose(1, 2);
                auto cur_v = v.index({idx::Slice(k_start, k_end)}).unsqueeze(0).transpose(1, 2);
                auto cur_out = at::scaled_dot_product_attention(
                    cur_q, cur_k, cur_v, std::nullopt, 0.0, true, std::optional<double>(static_cast<double>(p->scale)));
                result.index_put_({idx::Slice(q_start, q_end)}, cur_out.transpose(1, 2).squeeze(0));
            }
        } else {
            auto block_table_t = infinicore::adaptor::to_aten_tensor(*p->block_table);
            auto block_table_cpu = block_table_t.to(at::kCPU);
            const int64_t block_size = k.size(1);
            for (int64_t i = 0; i < num_seqs; ++i) {
                const int64_t q_start = cu_q_cpu.index({i}).item<int64_t>();
                const int64_t q_end = cu_q_cpu.index({i + 1}).item<int64_t>();
                const int64_t q_len = q_end - q_start;
                const int64_t h_len = (cu_k_cpu.index({i + 1}).item<int64_t>() - cu_k_cpu.index({i}).item<int64_t>()) - q_len;
                const int64_t total_len = h_len + q_len;
                auto cur_q = q.index({idx::Slice(q_start, q_end)});
                std::vector<at::Tensor> keys;
                std::vector<at::Tensor> values;
                keys.reserve(total_len);
                values.reserve(total_len);
                for (int64_t j = 0; j < total_len; ++j) {
                    const int64_t b_id = block_table_cpu.index({i, j / block_size}).item<int64_t>();
                    const int64_t off = j % block_size;
                    keys.push_back(k.index({b_id, off, idx::Slice(), idx::Slice()}));
                    values.push_back(v.index({b_id, off, idx::Slice(), idx::Slice()}));
                }
                auto K = at::stack(keys, 0);
                auto V = at::stack(values, 0);
                const int64_t q_heads = cur_q.size(1);
                const int64_t kv_heads = K.size(1);
                if (q_heads != kv_heads) {
                    const int64_t repeat = q_heads / kv_heads;
                    K = K.repeat_interleave(repeat, 1);
                    V = V.repeat_interleave(repeat, 1);
                }
                auto scores = at::matmul(cur_q.permute({1, 0, 2}).to(at::kFloat), K.permute({1, 2, 0}).to(at::kFloat)) * p->scale;
                auto mask = at::full({q_len, total_len}, -std::numeric_limits<float>::infinity(), q.options().dtype(at::kFloat));
                for (int64_t t = 0; t < q_len; ++t) {
                    mask.index_put_({t, idx::Slice(0, h_len + t + 1)}, 0.0);
                }
                auto attn = at::softmax(scores + mask.unsqueeze(0), -1).to(q.dtype());
                auto cur_out = at::matmul(attn, V.permute({1, 0, 2})).permute({1, 0, 2});
                result.index_put_({idx::Slice(q_start, q_end)}, cur_out);
            }
        }

        out_work.copy_(result);
        if (out_need_copy_back) {
            p->out->copy_from(out_work_ic);
        }
        return;
    }

    auto out = std::optional<at::Tensor>(out_work);
    std::optional<at::Tensor> seqused_k = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    auto block_table = p->block_table ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->block_table)) : std::nullopt;
    auto max_seqlen_q = p->max_seqlen_q;
    auto max_seqlen_k = p->max_seqlen_k;
    auto alibi_slopes = p->alibi_slopes ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes)) : std::nullopt;
    auto scale = p->scale;

    if (alibi_slopes.has_value()) {
        throw std::runtime_error("[mha_varlen/hygon] ALiBi is not supported by the direct libflash_attention varlen_fwd ABI");
    }
    auto q_work = q.contiguous();
    auto k_work = k.contiguous();
    auto v_work = v.contiguous();
    if (block_table.has_value() && k.dim() == 4 && v.dim() == 4) {
        const int64_t num_blocks = k.size(0);
        const int64_t block_size = k.size(1);
        const int64_t num_kv_heads = k.size(2);
        const int64_t head_dim = k.size(3);
        if (block_size % 64 != 0) {
            throw std::runtime_error("[mha_varlen/hygon] flash-attn requires paged KV block size to be divisible by 64");
        }
        const int64_t pages_per_block = block_size / 64;
        k_work = k_work.reshape({num_blocks, pages_per_block, 64, num_kv_heads, head_dim})
                     .reshape({num_blocks * pages_per_block, 64, num_kv_heads, head_dim})
                     .contiguous();
        v_work = v_work.reshape({num_blocks, pages_per_block, 64, num_kv_heads, head_dim})
                     .reshape({num_blocks * pages_per_block, 64, num_kv_heads, head_dim})
                     .contiguous();
        if (pages_per_block != 1) {
            auto offsets = at::arange(pages_per_block, block_table->options()).view({1, 1, pages_per_block});
            block_table = ((*block_table).unsqueeze(-1) * pages_per_block + offsets)
                              .reshape({block_table->size(0), block_table->size(1) * pages_per_block})
                              .contiguous();
        }
    }
    auto result = flash::vllm_mha_varlen_fwd(
        q_work,
        k_work,
        v_work,
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
        std::nullopt);
    if (!result.empty() && result[0].defined()) {
        out_work.copy_(result[0]);
    }
    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }

#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(Device::Type::kHygon, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(Device::Type::kHygon, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(Device::Type::kHygon, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_varlen_impl::flashattn
#endif // ENABLE_HYGON_API && ENABLE_FLASH_ATTN
