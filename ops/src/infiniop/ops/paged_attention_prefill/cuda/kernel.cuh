#ifndef __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
#define __PAGED_ATTENTION_PREFILL_KERNEL_CUH__

namespace op::paged_attention_prefill::cuda {

__device__ __forceinline__ size_t find_seq_id(size_t token_idx, const int64_t *cum_seq_lens_q, size_t num_seqs) {
    size_t low = 0, high = num_seqs - 1;
    while (low <= high) {
        size_t mid = (low + high) >> 1;
        if (token_idx >= (size_t)cum_seq_lens_q[mid] && token_idx < (size_t)cum_seq_lens_q[mid + 1]) {
            return mid;
        } else if (token_idx < (size_t)cum_seq_lens_q[mid]) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return 0;
}

template <typename Tdata, typename Tcompute>
__global__ void pagedAttentionPrefillKernel(
    Tdata *out_, const Tdata *q_, const Tdata *k_cache_, const Tdata *v_cache_,
    const int64_t *block_tables_,
    const int64_t *total_kv_lens_,
    const int64_t *cum_seq_lens_q_,
    const float *alibi_slopes_,
    const size_t num_heads, const size_t num_kv_heads, const float scale,
    const size_t max_num_blocks_per_seq, const size_t block_size,
    const ptrdiff_t kv_block_stride, const ptrdiff_t kv_head_stride,
    const ptrdiff_t q_stride, const ptrdiff_t q_head_stride,
    const size_t head_size,
    const size_t num_seqs) {

    // Grid : x -> token, y -> head
    const size_t global_token_idx = blockIdx.x;
    const size_t head_idx = blockIdx.y;
    const size_t dim_idx = threadIdx.x;

    if (dim_idx >= head_size) {
        return;
    }

    size_t seq_idx = find_seq_id(global_token_idx, cum_seq_lens_q_, num_seqs);

    size_t q_token_idx = global_token_idx - cum_seq_lens_q_[seq_idx];

    const size_t total_kv_len = total_kv_lens_[seq_idx];
    const size_t q_len = cum_seq_lens_q_[seq_idx + 1] - cum_seq_lens_q_[seq_idx];
    const size_t history_len = total_kv_len - q_len;
    const size_t causal_limit = history_len + q_token_idx;

    const Tdata *q_vec = q_ + global_token_idx * q_stride + head_idx * q_head_stride;
    Tdata *out_ptr = out_ + global_token_idx * num_heads * head_size + head_idx * head_size;

    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;
    const int64_t *block_table = block_tables_ + seq_idx * max_num_blocks_per_seq;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];

    Tcompute max_score = -FLT_MAX;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t b_idx = t / block_size;
        const size_t t_off = t % block_size;
        const ptrdiff_t physical_block_id = block_table[b_idx];
        const Tdata *k_vec = k_cache_ + physical_block_id * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size;

        Tcompute score = 0.0f;
        for (size_t d = 0; d < head_size; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += alibi_slope * static_cast<float>(t - causal_limit);
        }
        if (score > max_score) {
            max_score = score;
        }
    }

    Tcompute sum_exp = 0.0f;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t b_idx = t / block_size;
        const size_t t_off = t % block_size;
        const ptrdiff_t physical_block_id = block_table[b_idx];
        const Tdata *k_vec = k_cache_ + physical_block_id * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size;

        Tcompute score = 0.0f;
        for (size_t d = 0; d < head_size; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += alibi_slope * static_cast<float>(t - causal_limit);
        }
        sum_exp += expf(static_cast<float>(score - max_score));
    }

    Tcompute acc = 0.0f;
    Tcompute inv_sum = 1.0f / (sum_exp + 1e-6f);
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t b_idx = t / block_size;
        const size_t t_off = t % block_size;
        const ptrdiff_t physical_block_id = block_table[b_idx];

        const Tdata *k_vec = k_cache_ + physical_block_id * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size;
        Tcompute score = 0.0f;
        for (size_t d = 0; d < head_size; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += alibi_slope * static_cast<float>(t - causal_limit);
        }
        Tcompute prob = expf(static_cast<float>(score - max_score)) * inv_sum;

        const Tdata *v_vec = v_cache_ + physical_block_id * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size;
        acc += prob * static_cast<Tcompute>(v_vec[dim_idx]);
    }

    out_ptr[dim_idx] = static_cast<Tdata>(acc);
}

} // namespace op::paged_attention_prefill::cuda

#endif
