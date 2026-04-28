#ifndef __KV_CACHING_KERNEL_CUH__
#define __KV_CACHING_KERNEL_CUH__

template <typename Tdata, typename Tidx>
__device__ void kvCachingKernel(
    Tdata *__restrict__ k_cache,
    Tdata *__restrict__ v_cache,
    const Tdata *__restrict__ k,
    const Tdata *__restrict__ v,
    const Tidx *__restrict__ past_kv_lengths,
    int batch_size,
    int num_kv_heads,
    int max_seq_len,
    int seq_len,
    int hidden_dim,
    ptrdiff_t k_cache_strides_0,
    ptrdiff_t k_cache_strides_1,
    ptrdiff_t k_cache_strides_2,
    ptrdiff_t k_cache_strides_3,
    ptrdiff_t v_cache_strides_0,
    ptrdiff_t v_cache_strides_1,
    ptrdiff_t v_cache_strides_2,
    ptrdiff_t v_cache_strides_3,
    ptrdiff_t k_strides_0,
    ptrdiff_t k_strides_1,
    ptrdiff_t k_strides_2,
    ptrdiff_t k_strides_3,
    ptrdiff_t v_strides_0,
    ptrdiff_t v_strides_1,
    ptrdiff_t v_strides_2,
    ptrdiff_t v_strides_3) {
    // num of ele = B * H * seq_len * D
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_kv_heads * seq_len * hidden_dim;

    const int grid_size = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += grid_size) {
        // unravel index

        int d = idx % hidden_dim;
        idx /= hidden_dim;

        int s = idx % seq_len;
        idx /= seq_len;

        int h = idx % num_kv_heads;
        int b = idx / num_kv_heads;

        int past_len = static_cast<int>(past_kv_lengths[b]); // Cast to int for both types
        // write position
        int cache_s = past_len + s;
        int k_cache_offset = d * (int)k_cache_strides_3 + cache_s * (int)k_cache_strides_2 + h * (int)k_cache_strides_1 + b * (int)k_cache_strides_0;
        int v_cache_offset = d * (int)v_cache_strides_3 + cache_s * (int)v_cache_strides_2 + h * (int)v_cache_strides_1 + b * (int)v_cache_strides_0;

        int k_src_offset = d * (int)k_strides_3 + s * (int)k_strides_2 + h * (int)k_strides_1 + b * (int)k_strides_0;
        int v_src_offset = d * (int)v_strides_3 + s * (int)v_strides_2 + h * (int)v_strides_1 + b * (int)v_strides_0;
        k_cache[k_cache_offset] = k[k_src_offset];
        v_cache[v_cache_offset] = v[v_src_offset];
    }
}

#endif // __KV_CACHING_KERNEL_CUH__
