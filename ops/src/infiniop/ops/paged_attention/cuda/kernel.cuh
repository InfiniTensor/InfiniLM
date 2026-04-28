#ifndef __PAGED_ATTENTION_KERNEL_CUH__
#define __PAGED_ATTENTION_KERNEL_CUH__

// This kernel is refactored to be high-performance, adopting parallelism strategies
// from industry-standard implementations like vLLM. It fixes functional and performance
// issues in the original draft.

namespace op::paged_attention::cuda {

template <typename Tdata, typename Tcompute, size_t HEAD_SIZE, size_t NUM_THREADS>
__device__ void pagedAttentionKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const int64_t *block_tables_,
    const int64_t *seq_lens_,
    const float *alibi_slopes_,
    const size_t num_kv_heads,
    const float scale,
    const size_t max_num_blocks_per_seq,
    const size_t block_size,
    const ptrdiff_t q_stride,
    const ptrdiff_t kv_block_stride,
    const ptrdiff_t kv_head_stride,
    const ptrdiff_t o_stride) {
    //================================================================================
    // 1. Setup & Query Loading (No changes in this section)
    //================================================================================
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int num_heads = gridDim.x;
    const int64_t seq_len = seq_lens_[seq_idx];
    if (seq_len == 0) {
        return;
    }

    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;
    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];

    const int64_t *block_table = block_tables_ + seq_idx * max_num_blocks_per_seq;

    const Tdata *q_ptr = q_ + seq_idx * q_stride + head_idx * HEAD_SIZE;
    Tdata *out_ptr = out_ + seq_idx * o_stride + head_idx * HEAD_SIZE;

    extern __shared__ char shared_mem_char[];
    Tcompute *shared_mem = reinterpret_cast<Tcompute *>(shared_mem_char);
    Tcompute *q_shared = shared_mem;
    Tcompute *logits = shared_mem + HEAD_SIZE;

    // printf("static_cast<Tcompute>(q_ptr[i]);");
    for (size_t i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
        q_shared[i] = static_cast<Tcompute>(q_ptr[i]);
    }
    __syncthreads();
    //================================================================================
    // 2. Compute QK Dot Product & Find Max Logit
    //================================================================================
    for (size_t token_idx = threadIdx.x; token_idx < seq_len; token_idx += NUM_THREADS) {
        const int64_t block_idx = token_idx / block_size;
        const int64_t token_in_block_idx = token_idx % block_size;
        const int64_t physical_block_num = block_table[block_idx];

        const Tdata *k_vec_ptr = k_cache_ + physical_block_num * kv_block_stride + kv_head_idx * kv_head_stride + token_in_block_idx * HEAD_SIZE;

        Tcompute qk = 0.0f;
#pragma unroll
        for (size_t i = 0; i < HEAD_SIZE / 8; ++i) {
            const size_t offset = i * 8;

            // 手动展开8次计算
            qk += q_shared[offset + 0] * static_cast<Tcompute>(k_vec_ptr[offset + 0]);
            qk += q_shared[offset + 1] * static_cast<Tcompute>(k_vec_ptr[offset + 1]);
            qk += q_shared[offset + 2] * static_cast<Tcompute>(k_vec_ptr[offset + 2]);
            qk += q_shared[offset + 3] * static_cast<Tcompute>(k_vec_ptr[offset + 3]);
            qk += q_shared[offset + 4] * static_cast<Tcompute>(k_vec_ptr[offset + 4]);
            qk += q_shared[offset + 5] * static_cast<Tcompute>(k_vec_ptr[offset + 5]);
            qk += q_shared[offset + 6] * static_cast<Tcompute>(k_vec_ptr[offset + 6]);
            qk += q_shared[offset + 7] * static_cast<Tcompute>(k_vec_ptr[offset + 7]);
        }

        qk *= scale;
        if (alibi_slope != 0.0f) {
            qk += alibi_slope * (token_idx - seq_len + 1);
        }

        logits[token_idx] = qk;
    }
    __syncthreads();

    __shared__ Tcompute global_qk_max;
    Tcompute global_qk_max_0 = op::common_cuda::reduce_op::max<NUM_THREADS, Tcompute>(logits, seq_len);

    if (threadIdx.x == 0) {
        global_qk_max = global_qk_max_0;
    }
    __syncthreads();

    //================================================================================
    // 3. Compute Softmax (No changes in this section)
    //================================================================================

    for (size_t i = threadIdx.x; i < seq_len; i += NUM_THREADS) {
        Tcompute val = expf(logits[i] - global_qk_max); // 使用全局最大值
        logits[i] = val;
    }
    __syncthreads();

    __shared__ Tcompute inv_sum;
    Tcompute exp_sum_0 = op::common_cuda::reduce_op::sum<NUM_THREADS, Tcompute, Tcompute>(logits, seq_len);
    if (threadIdx.x == 0) {
        inv_sum = 1.0f / (exp_sum_0 + 1e-6f);
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < seq_len; i += NUM_THREADS) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    //================================================================================
    // 4. Aggregate Values (V) weighted by probabilities
    //================================================================================

    for (size_t h_dim = threadIdx.x; h_dim < HEAD_SIZE; h_dim += NUM_THREADS) {
        Tcompute acc = 0.0f;

        for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
            const size_t block_idx = token_idx / block_size;
            const size_t token_in_block_idx = token_idx % block_size;
            const int64_t physical_block_num = block_table[block_idx];
            const Tcompute prob = logits[token_idx];

            const Tdata *v_vec_ptr = v_cache_
                                   + physical_block_num * kv_block_stride
                                   + kv_head_idx * kv_head_stride
                                   + token_in_block_idx * HEAD_SIZE;

            const Tdata v_val = v_vec_ptr[h_dim];
            acc += prob * static_cast<Tcompute>(v_val);
        }
        out_ptr[h_dim] = static_cast<Tdata>(acc);
    }
}

} // namespace op::paged_attention::cuda

#endif // __PAGED_ATTENTION_KERNEL_CUH__
