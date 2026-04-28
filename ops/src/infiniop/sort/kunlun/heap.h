#ifndef __INFINIOP_HEAP_KUNLUN_H__
#define __INFINIOP_HEAP_KUNLUN_H__
#include "xpu/kernel/xtdk_simd_xpu2.h"

template <typename TK, typename TV>
static __device__ inline void sm_swap_kv(_shared_ptr_ TK *k0, _shared_ptr_ TV *v0,
                                         _shared_ptr_ TK *k1, _shared_ptr_ TV *v1) {
    TK tmpk = *k0;
    TV tmpv = *v0;
    *k0 = *k1;
    *v0 = *v1;
    *k1 = tmpk;
    *v1 = tmpv;
}

template <typename TK, typename TV>
static __device__ inline void update_sm_min_heap(_shared_ptr_ TK *heap_key,
                                                 _shared_ptr_ TV *heap_value, int idx, int heap_capacity) {
    while (idx < heap_capacity) {
        int child_l = idx * 2 + 1;
        int child_r = idx * 2 + 2;
        int child_min = child_l;
        if (child_r >= heap_capacity) {
            if (child_l >= heap_capacity) { // idx is leaf node, shift finished
                break;
            } else { // if child_r does not exist while child_l does, choose child_l
                child_min = child_l;
            }
        } else { // both child L & R exists
            child_min = child_l + (heap_key[child_l] > heap_key[child_r]);
        }
        if (heap_key[idx] <= heap_key[child_min]) {
            break;
        }
        sm_swap_kv(&heap_key[idx], &heap_value[idx], &heap_key[child_min], &heap_value[child_min]);
        idx = child_min;
    }
}

template <typename TK, typename TV>
static __device__ inline void make_sm_min_heap(
    _shared_ptr_ TK *heap_key, _shared_ptr_ TV *heap_value, int size) {
    for (int i = size / 2 - 1; i >= 0; i--) {
        update_sm_min_heap(heap_key, heap_value, i, size);
    }
}

template <typename TK, typename TV>
static __device__ inline void sort_sm_min_heap(
    _shared_ptr_ TK *heap_key, _shared_ptr_ TV *heap_value, int heap_capacity) {
    for (int i = heap_capacity - 1; i > 0; i--) {
        sm_swap_kv(&heap_key[0], &heap_value[0], &heap_key[i], &heap_value[i]);
        update_sm_min_heap(heap_key, heap_value, 0, i);
    }
}

template <typename TK, typename TV>
static __device__ inline void update_sm_max_heap(_shared_ptr_ TK *heap_key,
                                                 _shared_ptr_ TV *heap_value, int idx, int heap_capacity) {
    while (idx < heap_capacity) {
        int child_l = idx * 2 + 1;
        int child_r = idx * 2 + 2;
        int child_max = child_l;
        if (child_r >= heap_capacity) {
            if (child_l >= heap_capacity) { // idx is leaf node, shift finished
                break;
            } else { // if child_r does not exist while child_l does, choose child_l
                child_max = child_l;
            }
        } else { // both child L & R exists
            child_max = child_l + (heap_key[child_l] < heap_key[child_r]);
        }
        if (heap_key[idx] >= heap_key[child_max]) {
            break;
        }
        sm_swap_kv(&heap_key[idx], &heap_value[idx], &heap_key[child_max], &heap_value[child_max]);
        idx = child_max;
    }
}

template <typename TK, typename TV>
static __device__ inline void make_sm_max_heap(
    _shared_ptr_ TK *heap_key, _shared_ptr_ TV *heap_value, int size) {
    for (int i = size / 2 - 1; i >= 0; i--) {
        update_sm_max_heap(heap_key, heap_value, i, size);
    }
}

template <typename TK, typename TV>
static __device__ inline void sort_sm_max_heap(_shared_ptr_ TK *heap_key,
                                               _shared_ptr_ TV *heap_value, int heap_capacity) {
    for (int i = heap_capacity - 1; i > 0; i--) {
        sm_swap_kv(&heap_key[0], &heap_value[0], &heap_key[i], &heap_value[i]);
        update_sm_max_heap(heap_key, heap_value, 0, i);
    }
}

template <typename TK, typename TV>
static __device__ inline void lm_swap_kv(TK *k0, TV *v0,
                                         TK *k1, TV *v1) {
    TK tmpk = *k0;
    TV tmpv = *v0;
    *k0 = *k1;
    *v0 = *v1;
    *k1 = tmpk;
    *v1 = tmpv;
}

template <typename TK, typename TV>
static __device__ inline void update_lm_min_heap(TK *heap_key, TV *heap_value, int idx, int heap_capacity) {
    while (idx < heap_capacity) {
        int child_l = idx * 2 + 1;
        int child_r = idx * 2 + 2;
        int child_min = child_l;
        if (child_r >= heap_capacity) {
            if (child_l >= heap_capacity) { // idx is leaf node, shift finished
                break;
            } else { // if child_r does not exist while child_l does, choose child_l
                child_min = child_l;
            }
        } else { // both child L & R exists
            child_min = child_l + (heap_key[child_l] > heap_key[child_r]);
        }
        if (heap_key[idx] <= heap_key[child_min]) {
            break;
        }
        lm_swap_kv(&heap_key[idx], &heap_value[idx], &heap_key[child_min], &heap_value[child_min]);
        idx = child_min;
    }
}

template <typename TK, typename TV>
static __device__ inline void make_lm_min_heap(
    TK *heap_key, TV *heap_value, int size) {
    for (int i = size / 2 - 1; i >= 0; i--) {
        update_lm_min_heap(heap_key, heap_value, i, size);
    }
}

template <typename TK, typename TV>
static __device__ inline void sort_lm_min_heap(TK *heap_key, TV *heap_value, int heap_capacity) {
    for (int i = heap_capacity - 1; i > 0; i--) {
        lm_swap_kv(&heap_key[0], &heap_value[0], &heap_key[i], &heap_value[i]);
        update_lm_min_heap(heap_key, heap_value, 0, i);
    }
}

template <typename TK, typename TV>
static __device__ inline void update_lm_max_heap(TK *heap_key, TV *heap_value, int idx, int heap_capacity) {
    while (idx < heap_capacity) {
        int child_l = idx * 2 + 1;
        int child_r = idx * 2 + 2;
        int child_max = child_l;
        if (child_r >= heap_capacity) {
            if (child_l >= heap_capacity) { // idx is leaf node, shift finished
                break;
            } else { // if child_r does not exist while child_l does, choose child_l
                child_max = child_l;
            }
        } else { // both child L & R exists
            child_max = child_l + (heap_key[child_l] < heap_key[child_r]);
        }
        if (heap_key[idx] >= heap_key[child_max]) {
            break;
        }
        lm_swap_kv(&heap_key[idx], &heap_value[idx], &heap_key[child_max], &heap_value[child_max]);
        idx = child_max;
    }
}

template <typename TK, typename TV>
static __device__ inline void make_lm_max_heap(
    TK *heap_key, TV *heap_value, int size) {
    for (int i = size / 2 - 1; i >= 0; i--) {
        update_lm_max_heap(heap_key, heap_value, i, size);
    }
}

template <typename TK, typename TV>
static __device__ inline void sort_lm_max_heap(TK *heap_key, TV *heap_value, int heap_capacity) {
    for (int i = heap_capacity - 1; i > 0; i--) {
        lm_swap_kv(&heap_key[0], &heap_value[0], &heap_key[i], &heap_value[i]);
        update_lm_max_heap(heap_key, heap_value, 0, i);
    }
}

template <typename TID>
__device__ TID roundup_div_p(TID a, TID b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ T min_p(T a, T b) {
    return a < b ? a : b;
}

template <typename TID>
static __device__ inline void partition(int tid, int nthreads, TID len, int align, TID *start, TID *end) {
    TID block_cnt = roundup_div_p<TID>(len, align);
    TID remain_block = block_cnt % nthreads;
    TID start_block = block_cnt / nthreads * static_cast<TID>(tid) + min_p<TID>(tid, remain_block);
    TID end_block = start_block + block_cnt / nthreads + (tid < remain_block);
    *start = min_p<TID>(start_block * align, len);
    *end = min_p<TID>(end_block * align, len);
}

template <typename TX, typename TY>
static __device__ void primitive_cast(const TX *x, TY *y, int len) {
    return;
}

template <>
__device__ void primitive_cast(const float *x, int *y, int len) {
    for (int i = 0; i < len; i += 16) {
        float32x16_t Y = vload_lm_float32x16(x);
        __asm__ __volatile__("vfloat2fix.rz vr0, %0\t\n"
                             "vstore_mask16.mz vr0{mr1}, 0(%1)" ::"v"(Y),
                             "r"(y)
                             : "vr0");
        x += 16;
        y += 16;
    }
    mfence_lm();
}
template <>
__device__ void primitive_cast(const int *x, float *y, int len) {
    for (int i = 0; i < len; i += 16) {
        int32x16_t Y = vload_lm_int32x16(x);
        __asm__ __volatile__("vfix2float.rn vr0, %0\t\n"
                             "vstore_mask16.mz vr0{mr1}, 0(%1)" ::"v"(Y),
                             "r"(y)
                             : "vr0");
        x += 16;
        y += 16;
    }
    mfence_lm();
}

static __device__ inline void vload2_lm(const float *ptr, float32x16_t &vl, float32x16_t &vh) {
    vl = __builtin_xpu2_vload_mask16_mr1(ptr, 0);
    vh = __builtin_xpu2_vload_mask16_mr1(ptr + 16, 0);
}

static __device__ inline void vstore2_lm(float *ptr, float32x16_t &vl, float32x16_t &vh) {
    vstore_lm_float32x16(ptr, vl);
    vstore_lm_float32x16(ptr + 16, vh);
}

template <>
__device__ void primitive_cast(const float *x, float *y, int len) {
    if (x == y) {
        return;
    } else { // just copy
        float32x16_t vec_x_0;
        float32x16_t vec_x_1;
        for (int i = 0; i < len; i += 32) {
            vload2_lm(x + i, vec_x_0, vec_x_1);
            vstore2_lm(y + i, vec_x_0, vec_x_1);
        }
        mfence_lm();
    }
}

#endif
