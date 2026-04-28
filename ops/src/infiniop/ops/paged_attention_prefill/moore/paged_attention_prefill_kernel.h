#ifndef __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
#define __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
namespace op::paged_attention_prefill::cuda {

template <typename Tindex>
__device__ __forceinline__ size_t find_seq_id(size_t token_idx, const Tindex *cum_seq_lens_q, size_t num_seqs) {
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
// Warp-level sum reduction with an explicit active mask (safe for partial warps).
__device__ __forceinline__ float warpReduceSum(float val, unsigned mask) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}
// Block-level sum reduction. Returns the sum to all threads in the block.
// Supports blockDim.x up to 1024.
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // max 32 warps per block
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const unsigned mask = __activemask();
    val = warpReduceSum(val, mask);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    const int num_warps = (blockDim.x + 31) >> 5;
    float sum = 0.0f;
    if (wid == 0) {
        sum = (lane < num_warps) ? shared[lane] : 0.0f;
        const unsigned mask0 = (num_warps >= 32) ? 0xffffffffu : ((1u << num_warps) - 1u);
        sum = warpReduceSum(sum, mask0);
        if (lane == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();
    return shared[0];
}

template <typename Tindex, typename Tdata, typename Tcompute>
__global__ void pagedAttentionPrefillKernel(
    Tdata *out_, const Tdata *q_, const Tdata *k_cache_, const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cum_seq_lens_q_,
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
    __shared__ size_t sh_seq_idx;
    __shared__ size_t sh_causal_limit;
    __shared__ size_t sh_kv_head_idx;
    __shared__ float sh_scale_acc;
    __shared__ float sh_w;
    __shared__ float sh_inv_l;
    if (dim_idx == 0) {
        sh_seq_idx = find_seq_id<Tindex>(global_token_idx, cum_seq_lens_q_, num_seqs);
        const size_t q_token_idx = global_token_idx - static_cast<size_t>(cum_seq_lens_q_[sh_seq_idx]);
        const size_t total_kv_len = static_cast<size_t>(total_kv_lens_[sh_seq_idx]);
        const size_t q_len = static_cast<size_t>(cum_seq_lens_q_[sh_seq_idx + 1] - cum_seq_lens_q_[sh_seq_idx]);
        const size_t history_len = total_kv_len - q_len;
        sh_causal_limit = history_len + q_token_idx;
        const size_t num_queries_per_kv = num_heads / num_kv_heads;
        sh_kv_head_idx = head_idx / num_queries_per_kv;
    }
    __syncthreads();
    const size_t seq_idx = sh_seq_idx;
    const size_t causal_limit = sh_causal_limit;
    const size_t kv_head_idx = sh_kv_head_idx;
    const Tdata *q_vec = q_ + global_token_idx * q_stride + head_idx * q_head_stride;
    Tdata *out_ptr = out_ + global_token_idx * num_heads * head_size + head_idx * head_size;
    const Tindex *block_table = block_tables_ + seq_idx * max_num_blocks_per_seq;
    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    const float qv = static_cast<float>(q_vec[dim_idx]);
    Tcompute acc = 0.0f;
    float m = -FLT_MAX;
    float l = 0.0f;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t b_idx = t / block_size;
        const size_t t_off = t % block_size;
        const ptrdiff_t physical_block_id = block_table[b_idx];
        const Tdata *k_vec = k_cache_ + physical_block_id * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size;
        const float dot = blockReduceSum(qv * static_cast<float>(k_vec[dim_idx]));
        if (dim_idx == 0) {
            float score = dot * static_cast<float>(scale);
            if (alibi_slope != 0.0f) {
                score += alibi_slope * static_cast<float>(t - causal_limit);
            }
            const float m_new = fmaxf(m, score);
            const float scale_acc = expf(m - m_new);
            const float w = expf(score - m_new);
            l = l * scale_acc + w;
            m = m_new;
            sh_scale_acc = scale_acc;
            sh_w = w;
        }
        __syncthreads();
        const float scale_acc = sh_scale_acc;
        const float w = sh_w;
        const Tdata *v_vec = v_cache_ + physical_block_id * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size;
        acc = acc * static_cast<Tcompute>(scale_acc) + static_cast<Tcompute>(w) * static_cast<Tcompute>(v_vec[dim_idx]);
        __syncthreads();
    }
    if (dim_idx == 0) {
        sh_inv_l = 1.0f / (l + 1e-6f);
    }
    __syncthreads();
    out_ptr[dim_idx] = static_cast<Tdata>(acc * static_cast<Tcompute>(sh_inv_l));
}
} // namespace op::paged_attention_prefill::cuda
#endif
