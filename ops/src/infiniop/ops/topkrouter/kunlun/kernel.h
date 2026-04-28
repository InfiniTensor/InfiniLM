#ifndef __TOPKROUTER_KUNLUN_KERNEL_H__
#define __TOPKROUTER_KUNLUN_KERNEL_H__

#include "../../../devices/kunlun/kunlun_kernel_common.h"
#include "../../../sort/kunlun/heap.h"
#include <float.h>

using namespace device::kunlun::kernel;

template <typename T>
inline __device__ float expf_(T x) {
    float data;
    if constexpr (std::is_same_v<T, float>) {
        data = x;
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        data = __bfloat162float(x);
    } else if constexpr (std::is_same_v<T, half>) {
        data = __half2float(x);
    }
    return exp(data);
}

template <typename T>
inline __device__ float sigmoidf_(T x) {
    return 1.0f / (1.0f + expf_<T>(-x));
}

template <typename T, typename TID>
inline __device__ void descending_sort(T *x, TID *idx, int32_t n) {
    make_lm_min_heap(x, idx, n);
    mfence_lm();
    sort_lm_min_heap(x, idx, n);
    mfence_lm();
}

template <typename T, int32_t BLOCK_THREADS = 64, int32_t MAX_EXPERTS = 256,
          int32_t N_GROUPS = 8, int32_t TOPK_GROUP = 4, int32_t TOPK_PER_GROUP = 2>
__global__ void topkrouter_kernel(
    float *values_topk,             // 输出数据, 形状[N, topk]
    int32_t *indices_topk,          // 输出索引, 形状[N, topk]
    const T *input,                 // 输入数据 [N, n_experts]
    const float *d_correction_bias, // 输入数据 [n_experts]
    const float routed_scaling_factor,
    const int32_t N,         // N tokens
    const int32_t n_experts, // n_experts <= MAX_EXPERTS
    const int32_t topk) {

    const int32_t block_idx = cluster_id();
    if (block_idx >= N) {
        return;
    }
    const int32_t thread_idx = core_id();

    const int32_t GROUP_SIZE = n_experts / N_GROUPS; // 32 in DeepSeek-V3

    __shared__ T input_shm[MAX_EXPERTS]; // input shm for i-th token, total N
    __shared__ float correction_bias_sm[MAX_EXPERTS];

    // Copy data into SM
    if (thread_idx == 0) {
        GM2SM_ASYNC(input + block_idx * n_experts, input_shm, n_experts * sizeof(T));
        GM2SM_ASYNC(d_correction_bias, correction_bias_sm, n_experts * sizeof(float));
    }
    sync_cluster();

    // Calculate sigmoid scores and add bias
    __shared__ float scores[MAX_EXPERTS];
    __shared__ float scores_with_bias_shm[MAX_EXPERTS];
    for (int32_t i = thread_idx; i < n_experts; i += BLOCK_THREADS) {
        float v = sigmoidf_<T>(input_shm[i]);
        scores[i] = v;
        scores_with_bias_shm[i] = v + correction_bias_sm[i];
    }
    sync_cluster();

    // 按N_GROUPS分组，每组统计TOPK_PER_GROUP最大分数和
    __shared__ float values_grouped_topk_shm[N_GROUPS];
    if (thread_idx < N_GROUPS) {
        int32_t base = thread_idx * GROUP_SIZE;
        float tmp[TOPK_PER_GROUP];
// 初始化为负无穷，便于找topk
#pragma unroll
        for (int32_t k = 0; k < TOPK_PER_GROUP; ++k) {
            tmp[k] = -FLT_MAX;
        }
        // 维护一个TOPK_PER_GROUP大小的降序队列
        for (int32_t i = 0; i < GROUP_SIZE; ++i) {
            float val = scores_with_bias_shm[base + i];
            // 插入到队列
            if (val > tmp[TOPK_PER_GROUP - 1]) {
                int pos = TOPK_PER_GROUP - 1;
                while (pos > 0 && val > tmp[pos - 1]) {
                    tmp[pos] = tmp[pos - 1];
                    --pos;
                }
                tmp[pos] = val;
            }
        }
        float group_sum = 0.f;
        for (int32_t k = 0; k < TOPK_PER_GROUP; ++k) {
            group_sum += tmp[k];
        }
        values_grouped_topk_shm[thread_idx] = group_sum;
    }
    sync_cluster();

    // Select TOPK_GROUP in N_GROUPS according to sum of TOPK_PER_GROUP values in each group
    __shared__ int32_t indices_group[TOPK_GROUP];
    if (thread_idx == 0) {
        float values_group[TOPK_GROUP];
        int32_t indices_tmp[TOPK_GROUP];

// 初始化为负无穷和-1
#pragma unroll
        for (int32_t k = 0; k < TOPK_GROUP; ++k) {
            values_group[k] = -FLT_MAX;
            indices_tmp[k] = -1;
        }

        for (int32_t i = 0; i < N_GROUPS; i++) {
            float val = values_grouped_topk_shm[i];
            if (val > values_group[TOPK_GROUP - 1]) {
                int32_t pos = TOPK_GROUP - 1;
                while (pos > 0 && val > values_group[pos - 1]) {
                    values_group[pos] = values_group[pos - 1];
                    indices_tmp[pos] = indices_tmp[pos - 1];
                    pos--;
                }
                values_group[pos] = val;
                indices_tmp[pos] = i;
            }
        }
// 写入共享内存
#pragma unroll
        for (int32_t k = 0; k < TOPK_GROUP; ++k) {
            indices_group[k] = indices_tmp[k];
        }
    }
    sync_cluster();

    // 拷贝被选中的group的数据 values_group_select和 indices_group_select
    __shared__ float values_group_select[MAX_EXPERTS];
    __shared__ int32_t indices_group_select[MAX_EXPERTS];
    if (thread_idx < TOPK_GROUP) {
        int32_t group_id = indices_group[thread_idx];
        // 用于本线程复制group数据的临时buffer
        float local_buffer[GROUP_SIZE];
        // 拷贝选中group的所有分数到local_buffer
        __builtin_memcpy(local_buffer, scores_with_bias_shm + group_id * GROUP_SIZE, GROUP_SIZE * sizeof(float));
        mfence_lm();
        // 写回到共享内存选取buffer，对齐排列
        __builtin_memcpy(values_group_select + thread_idx * GROUP_SIZE, local_buffer, GROUP_SIZE * sizeof(float));
        // 记录原始索引
        for (int32_t i = 0; i < GROUP_SIZE; i++) {
            indices_group_select[thread_idx * GROUP_SIZE + i] = group_id * GROUP_SIZE + i;
        }
    }
    sync_cluster();

    // Global topk and copy to GM
    if (thread_idx == 0) {
        int32_t len = GROUP_SIZE * TOPK_GROUP;
        float values[len];
        int32_t indices[len];
        // COPY to LM
        __builtin_memcpy(values, values_group_select, len * sizeof(float));
        __builtin_memcpy(indices, indices_group_select, len * sizeof(int32_t));
        mfence_lm();
        // Sort
        descending_sort<float, int32_t>(values, indices, len);
        // Last scaling
        float sum = 1e-9f;
        for (int32_t k = 0; k < topk; k++) {
            int32_t idx = indices[k];
            sum += scores[idx];
        }
        for (int32_t k = 0; k < topk; k++) {
            int32_t idx = indices[k];
            values[k] = routed_scaling_factor * scores[idx] / sum;
        }
        mfence_lm();
        // COPY to GM
        LM2GM_ASYNC(values, values_topk, topk * sizeof(float));
        LM2GM_ASYNC(indices, indices_topk, topk * sizeof(int32_t));
    }
    sync_cluster();
}

#endif // __TOPKROUTER_KUNLUN_KERNEL_H__
