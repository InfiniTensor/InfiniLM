// Adapted from InfiniCore 1ab85ef1d3c2bce49edf1fc73a5b48becaf7552e:
// src/infiniop/ops/deepseek_moe/nvidia/deepseek_moe_nvidia.cu.
// Modified for InfiniOps: backend, layout and resource integration.
// Copyright 2025 InfiniTensor; MIT notice: LICENSES/InfiniCore-MIT.txt.
// Additional upstream notices are retained; see repository THIRD_PARTY.txt.
#include "native/cuda/nvidia/ops/deepseek_moe/kernel.h"
#include "native/cuda/nvidia/ops/moe_fused_gate/device_guard.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <type_traits>
#include <vector>
#include <string>
#include <cublas_v2.h>
namespace infini::ops { namespace {

enum MoeStatus { MOE_SUCCESS=0, MOE_BAD_PARAM, MOE_BAD_DTYPE, MOE_WORKSPACE, MOE_INTERNAL, MOE_DEVICE };
#define MOE_CHECK_STATUS(expr) do { auto s=(expr); if(s!=MOE_SUCCESS) return s; } while(0)
#define MOE_CHECK_CUDA(expr) do { auto s=(expr); if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); } while(0)
#define MOE_CHECK_CUBLAS(expr) do { auto s=(expr); if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("MoE cuBLAS failure: "+std::to_string(int(s))); } while(0)
inline void Check(MoeStatus s) { if(s!=MOE_SUCCESS) throw std::runtime_error("MoE native failure: "+std::to_string(int(s))); }
struct DeepseekMoeInfo { DataType dtype; size_t ntokens, hidden_size, topk, intermediate_size, num_experts; };

// Native by-value kernel arguments are copied by the CUDA launch. No temporary
// pageable/pinned host pointer array survives submission; the old FP16/BF16 weights are not repacked.
constexpr size_t kPointerChunk=128;
struct WeightPointerChunk { const void* gate[kPointerChunk]; const void* up[kPointerChunk]; const void* down[kPointerChunk]; };
static_assert(sizeof(WeightPointerChunk)==3072,"Fits pre-Volta CUDA's 4096-byte argument limit with outputs/scalars");
__global__ void StageWeightPointers(WeightPointerChunk pointers,const void** gate,const void** up,const void** down,size_t begin,size_t count) {
 const auto j=size_t(threadIdx.x); if(j<count) { gate[begin+j]=pointers.gate[j]; up[begin+j]=pointers.up[j]; down[begin+j]=pointers.down[j]; }
}

constexpr size_t ROUTE_BATCHED_GEMM_MAX_TOKENS = 16384;
constexpr size_t EXPERT_GROUPED_GEMM_MIN_TOKENS = 2048;
constexpr size_t FUSED_DECODE_MAX_TOKENS = 32;
constexpr int FUSED_KERNEL_THREADS = 64;
constexpr size_t CHUNKED_GROUPED_GEMM_TOKENS = 4096;

constexpr size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline char *advance_workspace(char *&ptr, size_t bytes, size_t alignment = 256) {
    ptr = reinterpret_cast<char *>(align_up(reinterpret_cast<uintptr_t>(ptr), alignment));
    char *out = ptr;
    ptr += bytes;
    return out;
}

template <typename T>
size_t route_batched_workspace_size(const DeepseekMoeInfo &info, size_t base_offset) {
    const size_t routes = info.ntokens * info.topk;
    size_t bytes = base_offset;
    bytes = align_up(bytes, alignof(void *));
    bytes += routes * sizeof(void *) * 9;
    bytes = align_up(bytes, 256);
    bytes += routes * info.intermediate_size * sizeof(T) * 3;
    bytes = align_up(bytes, 256);
    bytes += routes * info.hidden_size * sizeof(T);
    return align_up(bytes, 256);
}

template <typename T>
size_t expert_grouped_workspace_size(const DeepseekMoeInfo &info, size_t base_offset) {
    const size_t routes = info.ntokens * info.topk;
    size_t bytes = base_offset;
    for (int i = 0; i < 3; ++i) {
        bytes = align_up(bytes, 256);
        bytes += (info.num_experts + 1) * sizeof(int);
    }
    for (int i = 0; i < 2; ++i) {
        bytes = align_up(bytes, 256);
        bytes += routes * sizeof(int);
    }
    bytes = align_up(bytes, 256);
    bytes += routes * info.hidden_size * sizeof(T);
    for (int i = 0; i < 2; ++i) {
        bytes = align_up(bytes, 256);
        bytes += routes * info.intermediate_size * sizeof(T);
    }
    return align_up(bytes, 256);
}

template <typename T>
__device__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ float to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float dot_bf16x8(const uint4 &a, const uint4 &b) {
    const auto *a2 = reinterpret_cast<const __nv_bfloat162 *>(&a);
    const auto *b2 = reinterpret_cast<const __nv_bfloat162 *>(&b);
    float sum = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 av = __bfloat1622float2(a2[pair]);
        const float2 bv = __bfloat1622float2(b2[pair]);
        sum += av.x * bv.x + av.y * bv.y;
    }
    return sum;
}

template <typename T>
__global__ void gate_up_kernel(
    T *intermediate,
    const T *hidden,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    size_t ntokens,
    size_t hidden_size,
    size_t topk,
    size_t intermediate_size,
    size_t num_experts) {

    const size_t route = blockIdx.x / intermediate_size;
    const size_t j = blockIdx.x - route * intermediate_size;
    if (route >= ntokens * topk) {
        return;
    }
    const int expert = topk_indices[route];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        return;
    }
    const size_t token = route / topk;
    const T *x = hidden + token * hidden_size;
    const T *gate = reinterpret_cast<const T *>(gate_weights[expert]) + j * hidden_size;
    const T *up = reinterpret_cast<const T *>(up_weights[expert]) + j * hidden_size;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if ((hidden_size & 7) == 0) {
            const auto *x8 = reinterpret_cast<const uint4 *>(x);
            const auto *gate8 = reinterpret_cast<const uint4 *>(gate);
            const auto *up8 = reinterpret_cast<const uint4 *>(up);
            const size_t hidden_chunks = hidden_size / 8;
            for (size_t h8 = threadIdx.x; h8 < hidden_chunks; h8 += blockDim.x) {
                const uint4 xv = x8[h8];
                gate_sum += dot_bf16x8(xv, gate8[h8]);
                up_sum += dot_bf16x8(xv, up8[h8]);
            }
        } else if ((hidden_size & 1) == 0) {
            const auto *x2 = reinterpret_cast<const __nv_bfloat162 *>(x);
            const auto *gate2 = reinterpret_cast<const __nv_bfloat162 *>(gate);
            const auto *up2 = reinterpret_cast<const __nv_bfloat162 *>(up);
            const size_t hidden_pairs = hidden_size / 2;
            for (size_t h2 = threadIdx.x; h2 < hidden_pairs; h2 += blockDim.x) {
                const float2 xv = __bfloat1622float2(x2[h2]);
                const float2 gv = __bfloat1622float2(gate2[h2]);
                const float2 uv = __bfloat1622float2(up2[h2]);
                gate_sum += xv.x * gv.x + xv.y * gv.y;
                up_sum += xv.x * uv.x + xv.y * uv.y;
            }
        } else {
            for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
                const float xv = to_float<T>(x[h]);
                gate_sum += xv * to_float<T>(gate[h]);
                up_sum += xv * to_float<T>(up[h]);
            }
        }
    } else {
        for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            const float xv = to_float<T>(x[h]);
            gate_sum += xv * to_float<T>(gate[h]);
            up_sum += xv * to_float<T>(up[h]);
        }
    }

    __shared__ float gate_shared[FUSED_KERNEL_THREADS];
    __shared__ float up_shared[FUSED_KERNEL_THREADS];
    gate_shared[threadIdx.x] = gate_sum;
    up_shared[threadIdx.x] = up_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gate_shared[threadIdx.x] += gate_shared[threadIdx.x + stride];
            up_shared[threadIdx.x] += up_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float g = gate_shared[0];
        const float silu = g / (1.0f + __expf(-g));
        intermediate[route * intermediate_size + j] = from_float<T>(silu * up_shared[0] * topk_weights[route]);
    }
}

template <typename T>
__global__ void down_kernel(
    T *out,
    const T *intermediate,
    const int *topk_indices,
    const void *const *down_weights,
    size_t ntokens,
    size_t hidden_size,
    size_t topk,
    size_t intermediate_size,
    size_t num_experts) {

    const size_t linear = blockIdx.x;
    const size_t token = linear / hidden_size;
    const size_t h = linear - token * hidden_size;
    if (token >= ntokens) {
        return;
    }

    float acc = 0.0f;
    const size_t route_base = token * topk;
    const size_t count = topk * intermediate_size;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if ((intermediate_size & 7) == 0) {
            const size_t chunks_per_expert = intermediate_size / 8;
            const size_t chunk_count = topk * chunks_per_expert;
            for (size_t chunk_idx = threadIdx.x; chunk_idx < chunk_count; chunk_idx += blockDim.x) {
                const size_t k = chunk_idx / chunks_per_expert;
                const size_t j8 = chunk_idx - k * chunks_per_expert;
                const size_t route = route_base + k;
                const int expert = topk_indices[route];
                if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
                    const auto *intermediate8 = reinterpret_cast<const uint4 *>(
                        intermediate + route * intermediate_size);
                    const auto *down8 = reinterpret_cast<const uint4 *>(
                        reinterpret_cast<const T *>(down_weights[expert]) + h * intermediate_size);
                    acc += dot_bf16x8(intermediate8[j8], down8[j8]);
                }
            }
        } else if ((intermediate_size & 1) == 0) {
            const size_t pairs_per_expert = intermediate_size / 2;
            const size_t pair_count = topk * pairs_per_expert;
            for (size_t pair_idx = threadIdx.x; pair_idx < pair_count; pair_idx += blockDim.x) {
                const size_t k = pair_idx / pairs_per_expert;
                const size_t j2 = pair_idx - k * pairs_per_expert;
                const size_t route = route_base + k;
                const int expert = topk_indices[route];
                if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
                    const auto *intermediate2 = reinterpret_cast<const __nv_bfloat162 *>(
                        intermediate + route * intermediate_size);
                    const auto *down2 = reinterpret_cast<const __nv_bfloat162 *>(
                        reinterpret_cast<const T *>(down_weights[expert]) + h * intermediate_size);
                    const float2 iv = __bfloat1622float2(intermediate2[j2]);
                    const float2 dv = __bfloat1622float2(down2[j2]);
                    acc += iv.x * dv.x + iv.y * dv.y;
                }
            }
        } else {
            for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
                const size_t k = idx / intermediate_size;
                const size_t j = idx - k * intermediate_size;
                const size_t route = route_base + k;
                const int expert = topk_indices[route];
                if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
                    const T *down = reinterpret_cast<const T *>(down_weights[expert]) + h * intermediate_size;
                    acc += to_float<T>(intermediate[route * intermediate_size + j]) * to_float<T>(down[j]);
                }
            }
        }
    } else {
        for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
            const size_t k = idx / intermediate_size;
            const size_t j = idx - k * intermediate_size;
            const size_t route = route_base + k;
            const int expert = topk_indices[route];
            if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
                const T *down = reinterpret_cast<const T *>(down_weights[expert]) + h * intermediate_size;
                acc += to_float<T>(intermediate[route * intermediate_size + j]) * to_float<T>(down[j]);
            }
        }
    }

    __shared__ float shared[FUSED_KERNEL_THREADS];
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[token * hidden_size + h] = from_float<T>(shared[0]);
    }
}

__global__ void count_experts_kernel(
    const int *topk_indices,
    int *counts,
    int routes,
    int num_experts) {
    int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) {
        return;
    }
    int expert = topk_indices[route];
    if (expert >= 0 && expert < num_experts) {
        atomicAdd(counts + expert, 1);
    }
}

__global__ void prefix_counts_kernel(
    const int *counts,
    int *offsets,
    int num_experts) {
    extern __shared__ int scan[];
    int tid = threadIdx.x;
    if (tid < num_experts) {
        scan[tid] = counts[tid];
    } else {
        scan[tid] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int value = 0;
        if (tid >= stride) {
            value = scan[tid - stride];
        }
        __syncthreads();
        scan[tid] += value;
        __syncthreads();
    }

    if (tid == 0) {
        offsets[0] = 0;
    }
    if (tid < num_experts) {
        offsets[tid + 1] = scan[tid];
    }
}

template <typename T>
__global__ void pack_grouped_hidden_kernel(
    const T *hidden,
    const int *topk_indices,
    const int *offsets,
    int *positions,
    int *row_to_route,
    int *route_to_row,
    T *packed_hidden,
    int routes,
    int hidden_size,
    int topk,
    int num_experts) {
    int route = blockIdx.x;
    int tid = threadIdx.x;
    if (route >= routes) {
        return;
    }
    int expert = topk_indices[route];
    if (expert < 0 || expert >= num_experts) {
        return;
    }
    __shared__ int row;
    if (tid == 0) {
        int local = atomicAdd(positions + expert, 1);
        row = offsets[expert] + local;
        row_to_route[row] = route;
        route_to_row[route] = row;
    }
    __syncthreads();
    int token = route / topk;
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        packed_hidden[static_cast<size_t>(row) * hidden_size + h] = hidden[static_cast<size_t>(token) * hidden_size + h];
    }
}

template <typename T>
__global__ void swiglu_weight_grouped_kernel(
    T *activated,
    const T *gate,
    const T *up,
    const int *row_to_route,
    const float *topk_weights,
    int routes,
    int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = routes * intermediate_size;
    if (idx >= total) {
        return;
    }
    int row = idx / intermediate_size;
    int route = row_to_route[row];
    float g = to_float<T>(gate[idx]);
    float u = to_float<T>(up[idx]);
    float silu = g / (1.0f + __expf(-g));
    activated[idx] = from_float<T>(silu * u * topk_weights[route]);
}

template <typename T>
__global__ void sum_grouped_out_kernel(
    T *out,
    const T *packed_out,
    const int *route_to_row,
    int ntokens,
    int hidden_size,
    int topk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ntokens * hidden_size;
    if (idx >= total) {
        return;
    }
    int token = idx / hidden_size;
    int h = idx - token * hidden_size;
    float acc = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int route = token * topk + k;
        int row = route_to_row[route];
        acc += to_float<T>(packed_out[static_cast<size_t>(row) * hidden_size + h]);
    }
    out[idx] = from_float<T>(acc);
}

template <typename T>
__global__ void setup_route_batched_ptrs_kernel(
    const void **a_array,
    const void **b_array,
    void **c_array,
    const T *hidden,
    const int *topk_indices,
    const void *const *weights,
    T *gemm_out,
    int routes,
    int hidden_size,
    int topk,
    int out_size,
    int num_experts) {
    int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) {
        return;
    }
    int expert = topk_indices[route];
    if (expert < 0 || expert >= num_experts) {
        expert = 0;
    }
    int token = route / topk;
    a_array[route] = weights[expert];
    b_array[route] = hidden + static_cast<size_t>(token) * hidden_size;
    c_array[route] = gemm_out + static_cast<size_t>(route) * out_size;
}

template <typename T>
__global__ void setup_down_batched_ptrs_kernel(
    const void **a_array,
    const void **b_array,
    void **c_array,
    const T *activated,
    const int *topk_indices,
    const void *const *down_weights,
    T *route_out,
    int routes,
    int hidden_size,
    int topk,
    int intermediate_size,
    int num_experts) {
    int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) {
        return;
    }
    int expert = topk_indices[route];
    if (expert < 0 || expert >= num_experts) {
        expert = 0;
    }
    a_array[route] = down_weights[expert];
    b_array[route] = activated + static_cast<size_t>(route) * intermediate_size;
    c_array[route] = route_out + static_cast<size_t>(route) * hidden_size;
}

template <typename T>
__global__ void setup_all_batched_ptrs_kernel(
    const void **gate_a_array,
    const void **gate_b_array,
    void **gate_c_array,
    const void **up_a_array,
    const void **up_b_array,
    void **up_c_array,
    const void **down_a_array,
    const void **down_b_array,
    void **down_c_array,
    const T *hidden,
    const int *topk_indices,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    T *gate_buf,
    T *up_buf,
    T *activated,
    T *route_out,
    int routes,
    int hidden_size,
    int topk,
    int intermediate_size,
    int num_experts) {
    int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) {
        return;
    }
    int expert = topk_indices[route];
    if (expert < 0 || expert >= num_experts) {
        expert = 0;
    }
    int token = route / topk;
    const T *token_hidden = hidden + static_cast<size_t>(token) * hidden_size;

    gate_a_array[route] = gate_weights[expert];
    gate_b_array[route] = token_hidden;
    gate_c_array[route] = gate_buf + static_cast<size_t>(route) * intermediate_size;

    up_a_array[route] = up_weights[expert];
    up_b_array[route] = token_hidden;
    up_c_array[route] = up_buf + static_cast<size_t>(route) * intermediate_size;

    down_a_array[route] = down_weights[expert];
    down_b_array[route] = activated + static_cast<size_t>(route) * intermediate_size;
    down_c_array[route] = route_out + static_cast<size_t>(route) * hidden_size;
}

template <typename T>
__global__ void swiglu_weight_kernel(
    T *activated,
    const T *gate,
    const T *up,
    const float *topk_weights,
    int routes,
    int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = routes * intermediate_size;
    if (idx >= total) {
        return;
    }
    int route = idx / intermediate_size;
    float g = to_float<T>(gate[idx]);
    float u = to_float<T>(up[idx]);
    float silu = g / (1.0f + __expf(-g));
    activated[idx] = from_float<T>(silu * u * topk_weights[route]);
}

template <typename T>
__global__ void sum_route_out_kernel(
    T *out,
    const T *route_out,
    int ntokens,
    int hidden_size,
    int topk) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ntokens * hidden_size;
    if (linear >= total) {
        return;
    }
    int token = linear / hidden_size;
    int h = linear - token * hidden_size;
    float acc = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int route = token * topk + k;
        acc += to_float<T>(route_out[static_cast<size_t>(route) * hidden_size + h]);
    }
    out[linear] = from_float<T>(acc);
}

template <typename T>
MoeStatus run_batched_gemm(
    const NvidiaGemmHandle &internal,
    cudaStream_t stream,
    const void **a_array,
    const void **b_array,
    void **c_array,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int batch_count,
    cudaDataType dtype) {
    float alpha = 1.0f;
    float beta = 0.0f;
    const auto handle=internal.get();
    MOE_CHECK_CUBLAS(cublasSetStream(handle,stream));
    MOE_CHECK_CUBLAS(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
            MOE_CHECK_CUBLAS(cublasGemmBatchedEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                reinterpret_cast<const void *const *>(a_array),
                dtype,
                lda,
                reinterpret_cast<const void *const *>(b_array),
                dtype,
                ldb,
                &beta,
                reinterpret_cast<void *const *>(c_array),
                dtype,
                ldc,
                batch_count,
                compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return MOE_SUCCESS;
}

template <typename T>
MoeStatus run_gemm(
    const NvidiaGemmHandle &internal,
    cudaStream_t stream,
    const void *a,
    const void *b,
    void *c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    cudaDataType dtype) {
    float alpha = 1.0f;
    float beta = 0.0f;
    const auto handle=internal.get();
    MOE_CHECK_CUBLAS(cublasSetStream(handle,stream));
    MOE_CHECK_CUBLAS(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
            MOE_CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                a,
                dtype,
                lda,
                b,
                dtype,
                ldb,
                &beta,
                c,
                dtype,
                ldc,
                compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return MOE_SUCCESS;
}

template <typename T>
MoeStatus run_chunked_grouped_gemm(
    char *base,
    size_t workspace_size,
    size_t intermediate_offset,
    const DeepseekMoeInfo &info,
    T *out,
    const T *hidden,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    cudaStream_t stream,
    const NvidiaGemmHandle &internal) {

    const int hidden_size = static_cast<int>(info.hidden_size);
    const int topk = static_cast<int>(info.topk);
    const int intermediate_size = static_cast<int>(info.intermediate_size);
    const int num_experts = static_cast<int>(info.num_experts);
    const cudaDataType blas_dtype = info.dtype == DataType::kFloat16 ? CUDA_R_16F : CUDA_R_16BF;

    std::vector<int> host_counts(info.num_experts + 1);
    std::vector<int> host_offsets(info.num_experts + 1);
    std::vector<const void *> host_gate(info.num_experts);
    std::vector<const void *> host_up(info.num_experts);
    std::vector<const void *> host_down(info.num_experts);
        MOE_CHECK_CUDA(cudaMemcpyAsync(host_gate.data(), gate_weights, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
        MOE_CHECK_CUDA(cudaMemcpyAsync(host_up.data(), up_weights, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
        MOE_CHECK_CUDA(cudaMemcpyAsync(host_down.data(), down_weights, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
        MOE_CHECK_CUDA(cudaStreamSynchronize(stream));

    for (size_t token_begin = 0; token_begin < info.ntokens; token_begin += CHUNKED_GROUPED_GEMM_TOKENS) {
        const int chunk_tokens = static_cast<int>(std::min(
            CHUNKED_GROUPED_GEMM_TOKENS, info.ntokens - token_begin));
        const int routes = chunk_tokens * topk;

        char *gptr = base + intermediate_offset;
        auto *counts = reinterpret_cast<int *>(advance_workspace(gptr, (info.num_experts + 1) * sizeof(int)));
        auto *offsets = reinterpret_cast<int *>(advance_workspace(gptr, (info.num_experts + 1) * sizeof(int)));
        auto *positions = reinterpret_cast<int *>(advance_workspace(gptr, (info.num_experts + 1) * sizeof(int)));
        auto *row_to_route = reinterpret_cast<int *>(advance_workspace(gptr, static_cast<size_t>(routes) * sizeof(int)));
        auto *route_to_row = reinterpret_cast<int *>(advance_workspace(gptr, static_cast<size_t>(routes) * sizeof(int)));
        auto *packed_hidden = reinterpret_cast<T *>(advance_workspace(gptr, static_cast<size_t>(routes) * info.hidden_size * sizeof(T)));
        auto *gate_buf = reinterpret_cast<T *>(advance_workspace(gptr, static_cast<size_t>(routes) * info.intermediate_size * sizeof(T)));
        auto *up_buf = reinterpret_cast<T *>(advance_workspace(gptr, static_cast<size_t>(routes) * info.intermediate_size * sizeof(T)));
        if (static_cast<size_t>(gptr - base) > workspace_size) {
            return MOE_WORKSPACE;
        }

        const T *hidden_chunk = hidden + token_begin * info.hidden_size;
        const int *indices_chunk = topk_indices + token_begin * info.topk;
        const float *weights_chunk = topk_weights + token_begin * info.topk;
        T *out_chunk = out + token_begin * info.hidden_size;

        MOE_CHECK_CUDA(cudaMemsetAsync(counts, 0, (info.num_experts + 1) * sizeof(int), stream));
        MOE_CHECK_CUDA(cudaMemsetAsync(positions, 0, (info.num_experts + 1) * sizeof(int), stream));
        count_experts_kernel<<<(routes + 255) / 256, 256, 0, stream>>>(
            indices_chunk, counts, routes, num_experts);
        prefix_counts_kernel<<<1, 256, 256 * sizeof(int), stream>>>(
            counts, offsets, num_experts);
        pack_grouped_hidden_kernel<T><<<routes, 256, 0, stream>>>(
            hidden_chunk, indices_chunk, offsets, positions,
            row_to_route, route_to_row, packed_hidden,
            routes, hidden_size, topk, num_experts);

        MOE_CHECK_CUDA(cudaMemcpyAsync(host_counts.data(), counts, (info.num_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
        MOE_CHECK_CUDA(cudaMemcpyAsync(host_offsets.data(), offsets, (info.num_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
        MOE_CHECK_CUDA(cudaStreamSynchronize(stream));

        for (int e = 0; e < num_experts; ++e) {
            const int count = host_counts[e];
            if (count == 0) {
                continue;
            }
            const int offset = host_offsets[e];
            MOE_CHECK_STATUS(run_gemm<T>(
                internal, stream, host_gate[e], packed_hidden + static_cast<size_t>(offset) * hidden_size,
                gate_buf + static_cast<size_t>(offset) * intermediate_size,
                intermediate_size, count, hidden_size,
                hidden_size, hidden_size, intermediate_size, blas_dtype));
            MOE_CHECK_STATUS(run_gemm<T>(
                internal, stream, host_up[e], packed_hidden + static_cast<size_t>(offset) * hidden_size,
                up_buf + static_cast<size_t>(offset) * intermediate_size,
                intermediate_size, count, hidden_size,
                hidden_size, hidden_size, intermediate_size, blas_dtype));
        }

        swiglu_weight_grouped_kernel<T><<<(static_cast<size_t>(routes) * info.intermediate_size + 255) / 256, 256, 0, stream>>>(
            gate_buf, gate_buf, up_buf, row_to_route, weights_chunk, routes, intermediate_size);

        for (int e = 0; e < num_experts; ++e) {
            const int count = host_counts[e];
            if (count == 0) {
                continue;
            }
            const int offset = host_offsets[e];
            MOE_CHECK_STATUS(run_gemm<T>(
                internal, stream, host_down[e], gate_buf + static_cast<size_t>(offset) * intermediate_size,
                packed_hidden + static_cast<size_t>(offset) * hidden_size,
                hidden_size, count, intermediate_size,
                intermediate_size, intermediate_size, hidden_size, blas_dtype));
        }

        sum_grouped_out_kernel<T><<<(static_cast<size_t>(chunk_tokens) * info.hidden_size + 255) / 256, 256, 0, stream>>>(
            out_chunk, packed_hidden, route_to_row, chunk_tokens, hidden_size, topk);
    }

    // Keep the large-prefill path memory-bounded across transformer layers.
    // All earlier chunks are synchronized by the next chunk's routing copy;
    // this final sync drains the last chunk before its workspace can be reused.
    MOE_CHECK_CUDA(cudaStreamSynchronize(stream));

    return MOE_SUCCESS;
}

template <typename T>
MoeStatus launch_typed(
    void *workspace,
    size_t workspace_size,
    const DeepseekMoeInfo &info,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    cudaStream_t stream,
    const NvidiaGemmHandle &internal) {

    const size_t ptr_bytes = align_up(info.num_experts * sizeof(void *), 256);
    const size_t ptr_workspace = ptr_bytes * 3;
    const size_t intermediate_offset = align_up(ptr_workspace, 256);
    size_t intermediate_tokens = info.ntokens;
    if (info.ntokens > ROUTE_BATCHED_GEMM_MAX_TOKENS && info.num_experts <= 256) {
        intermediate_tokens = std::min(info.ntokens, CHUNKED_GROUPED_GEMM_TOKENS);
    }
    const size_t intermediate_bytes = intermediate_tokens * info.topk * info.intermediate_size * sizeof(T);
    if (workspace_size < intermediate_offset + intermediate_bytes) {
        return MOE_WORKSPACE;
    }

    auto *base = reinterpret_cast<char *>(workspace);
    const void *const *gate_ptrs = gate_weights;
    const void *const *up_ptrs = up_weights;
    const void *const *down_ptrs = down_weights;
    auto *intermediate = reinterpret_cast<T *>(base + intermediate_offset);

    if (info.ntokens > ROUTE_BATCHED_GEMM_MAX_TOKENS && info.num_experts <= 256) {
        return run_chunked_grouped_gemm<T>(
            base, workspace_size, intermediate_offset, info,
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(hidden),
            reinterpret_cast<const int *>(topk_indices), reinterpret_cast<const float *>(topk_weights),
            gate_weights, up_weights, down_weights,
            stream, internal);
    }

    if (info.ntokens > FUSED_DECODE_MAX_TOKENS
        && info.ntokens <= ROUTE_BATCHED_GEMM_MAX_TOKENS) {
        const int routes = static_cast<int>(info.ntokens * info.topk);
        const int hidden_size_i = static_cast<int>(info.hidden_size);
        const int topk_i = static_cast<int>(info.topk);
        const int intermediate_size_i = static_cast<int>(info.intermediate_size);
        const int num_experts_i = static_cast<int>(info.num_experts);
        const cudaDataType blas_dtype = info.dtype == DataType::kFloat16 ? CUDA_R_16F : CUDA_R_16BF;

        if (info.ntokens >= EXPERT_GROUPED_GEMM_MIN_TOKENS && info.num_experts <= 256) {
            char *gptr = base + intermediate_offset;
            auto *counts = reinterpret_cast<int *>(advance_workspace(gptr, (info.num_experts + 1) * sizeof(int)));
            auto *offsets = reinterpret_cast<int *>(advance_workspace(gptr, (info.num_experts + 1) * sizeof(int)));
            auto *positions = reinterpret_cast<int *>(advance_workspace(gptr, (info.num_experts + 1) * sizeof(int)));
            auto *row_to_route = reinterpret_cast<int *>(advance_workspace(gptr, static_cast<size_t>(routes) * sizeof(int)));
            auto *route_to_row = reinterpret_cast<int *>(advance_workspace(gptr, static_cast<size_t>(routes) * sizeof(int)));
            auto *packed_hidden = reinterpret_cast<T *>(advance_workspace(gptr, static_cast<size_t>(routes) * info.hidden_size * sizeof(T)));
            auto *gate_buf_g = reinterpret_cast<T *>(advance_workspace(gptr, static_cast<size_t>(routes) * info.intermediate_size * sizeof(T)));
            auto *up_buf_g = reinterpret_cast<T *>(advance_workspace(gptr, static_cast<size_t>(routes) * info.intermediate_size * sizeof(T)));
            if (static_cast<size_t>(gptr - base) <= workspace_size) {
                MOE_CHECK_CUDA(cudaMemsetAsync(counts, 0, (info.num_experts + 1) * sizeof(int), stream));
                MOE_CHECK_CUDA(cudaMemsetAsync(positions, 0, (info.num_experts + 1) * sizeof(int), stream));
                count_experts_kernel<<<(routes + 255) / 256, 256, 0, stream>>>(
                    reinterpret_cast<const int *>(topk_indices), counts, routes, num_experts_i);
                prefix_counts_kernel<<<1, 256, 256 * sizeof(int), stream>>>(
                    counts, offsets, num_experts_i);
                pack_grouped_hidden_kernel<T><<<routes, 256, 0, stream>>>(
                    reinterpret_cast<const T *>(hidden), reinterpret_cast<const int *>(topk_indices),
                    offsets, positions, row_to_route, route_to_row, packed_hidden,
                    routes, hidden_size_i, topk_i, num_experts_i);

                std::vector<int> host_counts(info.num_experts + 1);
                std::vector<int> host_offsets(info.num_experts + 1);
                std::vector<const void *> host_gate(info.num_experts);
                std::vector<const void *> host_up(info.num_experts);
                std::vector<const void *> host_down(info.num_experts);
                MOE_CHECK_CUDA(cudaMemcpyAsync(host_counts.data(), counts, (info.num_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
                MOE_CHECK_CUDA(cudaMemcpyAsync(host_offsets.data(), offsets, (info.num_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
                    MOE_CHECK_CUDA(cudaMemcpyAsync(host_gate.data(), gate_weights, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
                    MOE_CHECK_CUDA(cudaMemcpyAsync(host_up.data(), up_weights, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
                    MOE_CHECK_CUDA(cudaMemcpyAsync(host_down.data(), down_weights, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
                MOE_CHECK_CUDA(cudaStreamSynchronize(stream));

                for (int e = 0; e < num_experts_i; ++e) {
                    int count = host_counts[e];
                    if (count == 0) {
                        continue;
                    }
                    int offset = host_offsets[e];
                    MOE_CHECK_STATUS(run_gemm<T>(
                        internal, stream, host_gate[e], packed_hidden + static_cast<size_t>(offset) * hidden_size_i,
                        gate_buf_g + static_cast<size_t>(offset) * intermediate_size_i,
                        intermediate_size_i, count, hidden_size_i,
                        hidden_size_i, hidden_size_i, intermediate_size_i, blas_dtype));
                    MOE_CHECK_STATUS(run_gemm<T>(
                        internal, stream, host_up[e], packed_hidden + static_cast<size_t>(offset) * hidden_size_i,
                        up_buf_g + static_cast<size_t>(offset) * intermediate_size_i,
                        intermediate_size_i, count, hidden_size_i,
                        hidden_size_i, hidden_size_i, intermediate_size_i, blas_dtype));
                }

                swiglu_weight_grouped_kernel<T><<<(static_cast<size_t>(routes) * info.intermediate_size + 255) / 256, 256, 0, stream>>>(
                    gate_buf_g, gate_buf_g, up_buf_g, row_to_route, reinterpret_cast<const float *>(topk_weights), routes, intermediate_size_i);

                for (int e = 0; e < num_experts_i; ++e) {
                    int count = host_counts[e];
                    if (count == 0) {
                        continue;
                    }
                    int offset = host_offsets[e];
                    MOE_CHECK_STATUS(run_gemm<T>(
                        internal, stream, host_down[e], gate_buf_g + static_cast<size_t>(offset) * intermediate_size_i,
                        packed_hidden + static_cast<size_t>(offset) * hidden_size_i,
                        hidden_size_i, count, intermediate_size_i,
                        intermediate_size_i, intermediate_size_i, hidden_size_i, blas_dtype));
                }

                sum_grouped_out_kernel<T><<<(info.ntokens * info.hidden_size + 255) / 256, 256, 0, stream>>>(
                    reinterpret_cast<T *>(out), packed_hidden, route_to_row, static_cast<int>(info.ntokens), hidden_size_i, topk_i);
                return MOE_SUCCESS;
            }
        }

        char *bptr = base + intermediate_offset;
        auto **gate_a_array = reinterpret_cast<const void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto **gate_b_array = reinterpret_cast<const void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto **gate_c_array = reinterpret_cast<void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto **up_a_array = reinterpret_cast<const void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto **up_b_array = reinterpret_cast<const void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto **up_c_array = reinterpret_cast<void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto **down_a_array = reinterpret_cast<const void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto **down_b_array = reinterpret_cast<const void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto **down_c_array = reinterpret_cast<void **>(advance_workspace(bptr, static_cast<size_t>(routes) * sizeof(void *), alignof(void *)));
        auto *gate_buf = reinterpret_cast<T *>(advance_workspace(bptr, static_cast<size_t>(routes) * info.intermediate_size * sizeof(T)));
        auto *up_buf = reinterpret_cast<T *>(advance_workspace(bptr, static_cast<size_t>(routes) * info.intermediate_size * sizeof(T)));
        auto *activated = reinterpret_cast<T *>(advance_workspace(bptr, static_cast<size_t>(routes) * info.intermediate_size * sizeof(T)));
        auto *route_out = reinterpret_cast<T *>(advance_workspace(bptr, static_cast<size_t>(routes) * info.hidden_size * sizeof(T)));
        if (static_cast<size_t>(bptr - base) <= workspace_size) {
            const int blocks = (routes + 255) / 256;
            if (info.ntokens >= EXPERT_GROUPED_GEMM_MIN_TOKENS) {
                setup_all_batched_ptrs_kernel<T><<<blocks, 256, 0, stream>>>(
                    gate_a_array, gate_b_array, gate_c_array,
                    up_a_array, up_b_array, up_c_array,
                    down_a_array, down_b_array, down_c_array,
                    reinterpret_cast<const T *>(hidden), reinterpret_cast<const int *>(topk_indices),
                    gate_ptrs, up_ptrs, down_ptrs, gate_buf, up_buf, activated, route_out,
                    routes, hidden_size_i, topk_i, intermediate_size_i, num_experts_i);
                MOE_CHECK_STATUS(run_batched_gemm<T>(
                    internal, stream, gate_a_array, gate_b_array, gate_c_array,
                    intermediate_size_i, 1, hidden_size_i, hidden_size_i, hidden_size_i, intermediate_size_i,
                    routes, blas_dtype));

                MOE_CHECK_STATUS(run_batched_gemm<T>(
                    internal, stream, up_a_array, up_b_array, up_c_array,
                    intermediate_size_i, 1, hidden_size_i, hidden_size_i, hidden_size_i, intermediate_size_i,
                    routes, blas_dtype));

                swiglu_weight_kernel<T><<<(static_cast<size_t>(routes) * info.intermediate_size + 255) / 256, 256, 0, stream>>>(
                    activated, gate_buf, up_buf, reinterpret_cast<const float *>(topk_weights), routes, intermediate_size_i);

                MOE_CHECK_STATUS(run_batched_gemm<T>(
                    internal, stream, down_a_array, down_b_array, down_c_array,
                    hidden_size_i, 1, intermediate_size_i, intermediate_size_i, intermediate_size_i, hidden_size_i,
                    routes, blas_dtype));
            } else {
                setup_route_batched_ptrs_kernel<T><<<blocks, 256, 0, stream>>>(
                    gate_a_array, gate_b_array, gate_c_array,
                    reinterpret_cast<const T *>(hidden), reinterpret_cast<const int *>(topk_indices), gate_ptrs,
                    gate_buf, routes, hidden_size_i, topk_i, intermediate_size_i, num_experts_i);
                MOE_CHECK_STATUS(run_batched_gemm<T>(
                    internal, stream, gate_a_array, gate_b_array, gate_c_array,
                    intermediate_size_i, 1, hidden_size_i, hidden_size_i, hidden_size_i, intermediate_size_i,
                    routes, blas_dtype));

                setup_route_batched_ptrs_kernel<T><<<blocks, 256, 0, stream>>>(
                    gate_a_array, gate_b_array, gate_c_array,
                    reinterpret_cast<const T *>(hidden), reinterpret_cast<const int *>(topk_indices), up_ptrs,
                    up_buf, routes, hidden_size_i, topk_i, intermediate_size_i, num_experts_i);
                MOE_CHECK_STATUS(run_batched_gemm<T>(
                    internal, stream, gate_a_array, gate_b_array, gate_c_array,
                    intermediate_size_i, 1, hidden_size_i, hidden_size_i, hidden_size_i, intermediate_size_i,
                    routes, blas_dtype));

                swiglu_weight_kernel<T><<<(static_cast<size_t>(routes) * info.intermediate_size + 255) / 256, 256, 0, stream>>>(
                    activated, gate_buf, up_buf, reinterpret_cast<const float *>(topk_weights), routes, intermediate_size_i);

                setup_down_batched_ptrs_kernel<T><<<blocks, 256, 0, stream>>>(
                    gate_a_array, gate_b_array, gate_c_array,
                    activated, reinterpret_cast<const int *>(topk_indices), down_ptrs,
                    route_out, routes, hidden_size_i, topk_i, intermediate_size_i, num_experts_i);
                MOE_CHECK_STATUS(run_batched_gemm<T>(
                    internal, stream, gate_a_array, gate_b_array, gate_c_array,
                    hidden_size_i, 1, intermediate_size_i, intermediate_size_i, intermediate_size_i, hidden_size_i,
                    routes, blas_dtype));
            }

            sum_route_out_kernel<T><<<(info.ntokens * info.hidden_size + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<T *>(out), route_out, static_cast<int>(info.ntokens), hidden_size_i, topk_i);
            return MOE_SUCCESS;
        }
    }

    constexpr int threads = FUSED_KERNEL_THREADS;
    const dim3 gate_blocks(static_cast<unsigned int>(info.ntokens * info.topk * info.intermediate_size));
    gate_up_kernel<T><<<gate_blocks, threads, 0, stream>>>(
        intermediate,
        reinterpret_cast<const T *>(hidden),
        reinterpret_cast<const int *>(topk_indices),
        reinterpret_cast<const float *>(topk_weights),
        gate_ptrs,
        up_ptrs,
        info.ntokens,
        info.hidden_size,
        info.topk,
        info.intermediate_size,
        info.num_experts);

    const dim3 down_blocks(static_cast<unsigned int>(info.ntokens * info.hidden_size));
    down_kernel<T><<<down_blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        intermediate,
        reinterpret_cast<const int *>(topk_indices),
        down_ptrs,
        info.ntokens,
        info.hidden_size,
        info.topk,
        info.intermediate_size,
        info.num_experts);

    return MOE_SUCCESS;
}

DeepseekMoeInfo Info(const std::vector<Tensor>& v,size_t m,size_t e) { return {v[1].dtype(),v[1].size(0),v[1].size(1),v[2].size(1),m,e}; }
size_t WorkspaceBytes(const DeepseekMoeInfo& info) {
    const size_t dtype_size = info.dtype == DataType::kFloat16 ? sizeof(half) : sizeof(__nv_bfloat16);
    const size_t ptr_bytes = align_up(info.num_experts * sizeof(void *), 256);
    const size_t intermediate_offset = align_up(ptr_bytes * 3, 256);
    size_t workspace_tokens = info.ntokens;
    if (info.ntokens > ROUTE_BATCHED_GEMM_MAX_TOKENS && info.num_experts <= 256) {
        workspace_tokens = std::min(info.ntokens, CHUNKED_GROUPED_GEMM_TOKENS);
    }
    const size_t intermediate_bytes = workspace_tokens * info.topk * info.intermediate_size * dtype_size;
    const size_t old_workspace_size = intermediate_offset + intermediate_bytes;
    size_t batched_workspace_size = old_workspace_size;
    size_t grouped_workspace_size = old_workspace_size;
    if (info.ntokens > ROUTE_BATCHED_GEMM_MAX_TOKENS && info.num_experts <= 256) {
        auto chunk_info = info;
        chunk_info.ntokens = workspace_tokens;
        if (info.dtype == DataType::kFloat16) {
            grouped_workspace_size = expert_grouped_workspace_size<half>(chunk_info, intermediate_offset);
        } else {
            grouped_workspace_size = expert_grouped_workspace_size<__nv_bfloat16>(chunk_info, intermediate_offset);
        }
    } else if (info.ntokens > FUSED_DECODE_MAX_TOKENS
               && info.ntokens <= ROUTE_BATCHED_GEMM_MAX_TOKENS) {
        if (info.dtype == DataType::kFloat16) {
            batched_workspace_size = route_batched_workspace_size<half>(info, intermediate_offset);
            grouped_workspace_size = expert_grouped_workspace_size<half>(info, intermediate_offset);
        } else {
            batched_workspace_size = route_batched_workspace_size<__nv_bfloat16>(info, intermediate_offset);
            grouped_workspace_size = expert_grouped_workspace_size<__nv_bfloat16>(info, intermediate_offset);
        }
    }
    const size_t workspace_size = std::max(old_workspace_size, std::max(batched_workspace_size, grouped_workspace_size));

return workspace_size + 4096; // per-region alignment headroom in the old pointer/buffer layout
}

} // namespace
Operator<DeepseekMoe, Device::Type::kNvidia, 0>::Operator(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) : DeepseekMoe(output, hidden, ids, weights, gate, up, down, intermediate, num_experts), blas_(NvidiaGemmHandle::Acquire(output.device().index())) { moe_detail::Require(SupportsLayout(hidden,gate,up,down,intermediate)); }
size_t Operator<DeepseekMoe, Device::Type::kNvidia, 0>::workspace_size_in_bytes() const { return WorkspaceBytes(Info(metadata_,intermediate_,num_experts_)); }
void Operator<DeepseekMoe, Device::Type::kNvidia, 0>::operator()(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) const {
  MoeDeviceGuard device_guard{metadata_[0].device().index()};
  Invocation(output, hidden, ids, weights, gate, up, down, intermediate, num_experts); if(hidden.size(0)==0) return;
  moe_detail::Require(SupportsLayout(hidden,gate,up,down,intermediate));
  const auto info=Info(metadata_,intermediate,num_experts);
  moe_detail::Workspace(workspace_,workspace_size_in_bytes_,WorkspaceBytes(info),Collect(output,hidden,ids,weights,gate,up,down));
  const auto stream=static_cast<cudaStream_t>(stream_);
  const size_t ptr_bytes=align_up(num_experts*sizeof(void*),256);
  auto base=static_cast<char*>(workspace_);
  auto gp=reinterpret_cast<const void**>(base);
  auto uptr=reinterpret_cast<const void**>(base+ptr_bytes);
  auto dp=reinterpret_cast<const void**>(base+2*ptr_bytes);
  for(size_t begin=0;begin<num_experts;begin+=kPointerChunk) {
    WeightPointerChunk pointers{};
    const auto count=std::min(kPointerChunk,num_experts-begin);
    for(size_t j=0;j<count;++j) { pointers.gate[j]=gate[begin+j].data(); pointers.up[j]=up[begin+j].data(); pointers.down[j]=down[begin+j].data(); }
    StageWeightPointers<<<1,kPointerChunk,0,stream>>>(pointers,gp,uptr,dp,begin,count);
  }
  if(hidden.dtype()==DataType::kFloat16) Check(launch_typed<half>(workspace_,workspace_size_in_bytes_,info,output.data(),hidden.data(),ids.data(),weights.data(),gp,uptr,dp,stream,*blas_));
  else Check(launch_typed<__nv_bfloat16>(workspace_,workspace_size_in_bytes_,info,output.data(),hidden.data(),ids.data(),weights.data(),gp,uptr,dp,stream,*blas_));
  MOE_CHECK_CUDA(cudaGetLastError());

}

#undef MOE_CHECK_STATUS
#undef MOE_CHECK_CUDA
#undef MOE_CHECK_CUBLAS
} // namespace infini::ops
