// CUDA/CUTLASS body ported from frozen InfiniCore 0.2.9. See THIRD_PARTY.txt for source and license details.
#include "native/cuda/nvidia/ops/moe_fused_dense/kernel.h"
#include "native/cuda/nvidia/ops/moe_fused_gate/device_guard.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
namespace infini::ops { namespace {

enum MoeStatus { MOE_SUCCESS=0, MOE_BAD_PARAM, MOE_BAD_DTYPE, MOE_WORKSPACE, MOE_INTERNAL, MOE_DEVICE };
#define MOE_CHECK_STATUS(expr) do { auto s=(expr); if(s!=MOE_SUCCESS) return s; } while(0)
#define MOE_CHECK_CUDA(expr) do { auto s=(expr); if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); } while(0)
inline void Check(MoeStatus s) { if(s!=MOE_SUCCESS) throw std::runtime_error("MoE native failure: "+std::to_string(int(s))); }
struct MoeFusedDenseInfo { size_t num_tokens, hidden_size, num_experts, intermediate_size, topk, max_num_tokens_padded, max_num_blocks, block_size; DataType dtype; };


size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void *advance_workspace(uint8_t *&ptr, size_t &remaining, size_t bytes, size_t alignment = 16) {
    auto address = reinterpret_cast<uintptr_t>(ptr);
    auto aligned = align_up(address, alignment);
    auto padding = aligned - address;
    if (padding + bytes > remaining) {
        return nullptr;
    }
    ptr += padding;
    remaining -= padding;
    void *out = ptr;
    ptr += bytes;
    remaining -= bytes;
    return out;
}

size_t workspace_size(const MoeFusedDenseInfo &info) {
    const size_t pairs = info.num_tokens * info.topk;
    const size_t rows = info.max_num_tokens_padded;
    const size_t dtype_size = info.dtype == DataType::kFloat16 ? sizeof(half) : sizeof(__nv_bfloat16);
    size_t bytes = 0;
    bytes += align_up((info.num_experts + 1) * sizeof(int), 16);
    bytes += align_up((info.num_experts + 1) * sizeof(int), 16);
    bytes += align_up(pairs * sizeof(int), 16);
    bytes += align_up(rows * info.hidden_size * dtype_size, 16);
    bytes += align_up(rows * info.intermediate_size * 2 * dtype_size, 16);
    bytes += align_up(rows * info.intermediate_size * dtype_size, 16);
    bytes += align_up(rows * info.hidden_size * dtype_size, 16);
    bytes += align_up(info.num_experts * sizeof(cutlass::gemm::GemmCoord), 16);
    bytes += align_up(info.num_experts * sizeof(void *), 16) * 4;
    bytes += align_up(info.num_experts * sizeof(int64_t), 16) * 4;
    bytes += align_up(sizeof(int), 16);
    return bytes + 256;
}

template <typename T>
__device__ float to_float(T v);

template <>
__device__ float to_float<half>(half v) {
    return __half2float(v);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ T from_float(float v);

template <>
__device__ half from_float<half>(float v) {
    return __float2half(v);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__global__ void exclusive_prefix_counts_kernel(const int *counts, int *offsets, int num_experts) {
    extern __shared__ int scan[];
    int tid = threadIdx.x;
    if (tid < num_experts) {
        scan[tid] = counts[tid];
    }
    if (tid >= num_experts && tid < blockDim.x) {
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

__global__ void count_aligned_experts_kernel(const int *expert_ids,
                                             const int *num_tokens_post_padded,
                                             int *counts,
                                             int num_experts,
                                             int block_size) {
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = (*num_tokens_post_padded + block_size - 1) / block_size;
    if (block >= num_blocks) {
        return;
    }
    int expert = expert_ids[block];
    if (expert >= 0 && expert < num_experts) {
        atomicAdd(counts + expert, block_size);
    }
}

template <typename T>
__global__ void pack_hidden_aligned_kernel(const T *hidden,
                                           const int *sorted_token_ids,
                                           int *output_permutation,
                                           T *packed_hidden,
                                           int pairs,
                                           int topk,
                                           int hidden_size,
                                           int max_num_tokens_padded) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= max_num_tokens_padded) {
        return;
    }
    int pair = sorted_token_ids[row];
    if (pair >= 0 && pair < pairs) {
        if (tid == 0) {
            output_permutation[pair] = row;
        }
        int token = pair / topk;
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            packed_hidden[static_cast<size_t>(row) * hidden_size + h] = hidden[static_cast<size_t>(token) * hidden_size + h];
        }
    } else {
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            packed_hidden[static_cast<size_t>(row) * hidden_size + h] = from_float<T>(0.0f);
        }
    }
}

template <typename T>
__global__ void fill_prefill_bias_kernel(T *gate_up, T *expert_out,
                                         const T *bias1, const T *bias2,
                                         const int *counts, const int *offsets,
                                         int intermediate_size, int hidden_size) {
    const int expert = blockIdx.x;
    const int rows = counts[expert];
    const int offset = offsets[expert];
    const int gate_columns = intermediate_size * 2;
    if (bias1) {
        for (size_t i = threadIdx.x; i < static_cast<size_t>(rows) * gate_columns; i += blockDim.x) {
            gate_up[static_cast<size_t>(offset) * gate_columns + i] =
                bias1[static_cast<size_t>(expert) * gate_columns + i % gate_columns];
        }
    }
    if (bias2) {
        for (size_t i = threadIdx.x; i < static_cast<size_t>(rows) * hidden_size; i += blockDim.x) {
            expert_out[static_cast<size_t>(offset) * hidden_size + i] =
                bias2[static_cast<size_t>(expert) * hidden_size + i % hidden_size];
        }
    }
}

template <typename T>
__global__ void fill_decode_bias_kernel(T *gate_up, T *expert_out,
                                        const T *bias1, const T *bias2,
                                        const int *sorted_token_ids, const int *expert_ids,
                                        const int *num_tokens_post_padded, const int *output_permutation,
                                        int pairs, int num_experts, int intermediate_size,
                                        int hidden_size, int block_size) {
    const int row = blockIdx.x;
    if (row >= *num_tokens_post_padded) return;
    const int pair = sorted_token_ids[row];
    if (pair < 0 || pair >= pairs) return;
    const int expert = expert_ids[row / block_size];
    if (expert < 0 || expert >= num_experts) return;
    const int idx = output_permutation[pair];
    if (idx < 0 || idx >= pairs) return;
    const int gate_columns = intermediate_size * 2;
    if (bias1) {
        for (size_t col = threadIdx.x; col < static_cast<size_t>(gate_columns); col += blockDim.x) {
            gate_up[static_cast<size_t>(idx) * gate_columns + col] =
                bias1[static_cast<size_t>(expert) * gate_columns + col];
        }
    }
    if (bias2) {
        for (size_t col = threadIdx.x; col < static_cast<size_t>(hidden_size); col += blockDim.x) {
            expert_out[static_cast<size_t>(idx) * hidden_size + col] =
                bias2[static_cast<size_t>(expert) * hidden_size + col];
        }
    }
}

template <typename T>
__global__ void swiglu_kernel(const T *gate_up, T *activated, int rows, int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * intermediate_size;
    if (idx >= total) {
        return;
    }
    int row = idx / intermediate_size;
    int col = idx - row * intermediate_size;
    const T *base = gate_up + static_cast<size_t>(row) * intermediate_size * 2;
    float gate = to_float<T>(base[col]);
    float up = to_float<T>(base[intermediate_size + col]);
    float silu = gate / (1.0f + expf(-gate));
    activated[idx] = from_float<T>(up * silu);
}

template <typename T>
__global__ void apply_shuffle_mul_sum_kernel(const T *__restrict__ expert_out,
                                             T *__restrict__ out,
                                             const int *__restrict__ output_permutation,
                                             const float *__restrict__ topk_weights,
                                             int num_tokens,
                                             int topk,
                                             int hidden_size) {
    int token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < topk; ++k) {
            int pair = token * topk + k;
            int src_row = output_permutation[pair];
            if (src_row >= 0) {
                sum += to_float<T>(expert_out[static_cast<size_t>(src_row) * hidden_size + h]) * topk_weights[pair];
            }
        }
        out[static_cast<size_t>(token) * hidden_size + h] = from_float<T>(sum);
    }
}

template <typename T>
__global__ void setup_prefill_gemm1_kernel(cutlass::gemm::GemmCoord *problems,
                                           void **ptr_a,
                                           void **ptr_b,
                                           void **ptr_c,
                                           void **ptr_d,
                                           int64_t *lda,
                                           int64_t *ldb,
                                           int64_t *ldc,
                                           int64_t *ldd,
                                           const int *counts,
                                           const int *offsets,
                                           const T *packed_hidden,
                                           const T *w13,
                                           T *gate_up,
                                           int num_experts,
                                           int hidden_size,
                                           int intermediate_size) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }
    int m = counts[expert];
    int off = offsets[expert];
    problems[expert] = cutlass::gemm::GemmCoord(m, intermediate_size * 2, hidden_size);
    ptr_a[expert] = const_cast<T *>(packed_hidden + static_cast<size_t>(off) * hidden_size);
    ptr_b[expert] = const_cast<T *>(w13 + static_cast<size_t>(expert) * intermediate_size * 2 * hidden_size);
    ptr_c[expert] = gate_up + static_cast<size_t>(off) * intermediate_size * 2;
    ptr_d[expert] = gate_up + static_cast<size_t>(off) * intermediate_size * 2;
    lda[expert] = hidden_size;
    ldb[expert] = hidden_size;
    ldc[expert] = intermediate_size * 2;
    ldd[expert] = intermediate_size * 2;
}

template <typename T>
__global__ void setup_prefill_gemm2_kernel(cutlass::gemm::GemmCoord *problems,
                                           void **ptr_a,
                                           void **ptr_b,
                                           void **ptr_c,
                                           void **ptr_d,
                                           int64_t *lda,
                                           int64_t *ldb,
                                           int64_t *ldc,
                                           int64_t *ldd,
                                           const int *counts,
                                           const int *offsets,
                                           const T *activated,
                                           const T *w2,
                                           T *expert_out,
                                           int num_experts,
                                           int hidden_size,
                                           int intermediate_size) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }
    int m = counts[expert];
    int off = offsets[expert];
    problems[expert] = cutlass::gemm::GemmCoord(m, hidden_size, intermediate_size);
    ptr_a[expert] = const_cast<T *>(activated + static_cast<size_t>(off) * intermediate_size);
    ptr_b[expert] = const_cast<T *>(w2 + static_cast<size_t>(expert) * hidden_size * intermediate_size);
    ptr_c[expert] = expert_out + static_cast<size_t>(off) * hidden_size;
    ptr_d[expert] = expert_out + static_cast<size_t>(off) * hidden_size;
    lda[expert] = intermediate_size;
    ldb[expert] = intermediate_size;
    ldc[expert] = hidden_size;
    ldd[expert] = hidden_size;
}

template <typename T>
__global__ void setup_decode_gemm1_aligned_compact_kernel(cutlass::gemm::GemmCoord *problems,
                                                          void **ptr_a,
                                                          void **ptr_b,
                                                          void **ptr_c,
                                                          void **ptr_d,
                                                          int64_t *lda,
                                                          int64_t *ldb,
                                                          int64_t *ldc,
                                                          int64_t *ldd,
                                                          int *active_count,
                                                          int *output_permutation,
                                                          const T *hidden,
                                                          const T *w13,
                                                          T *gate_up,
                                                          const int *sorted_token_ids,
                                                          const int *expert_ids,
                                                          const int *num_tokens_post_padded,
                                                          int pairs,
                                                          int topk,
                                                          int num_experts,
                                                          int hidden_size,
                                                          int intermediate_size,
                                                          int block_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = *num_tokens_post_padded;
    if (row >= total_rows) {
        return;
    }
    int pair = sorted_token_ids[row];
    if (pair < 0 || pair >= pairs) {
        return;
    }
    int expert = expert_ids[row / block_size];
    if (expert < 0 || expert >= num_experts) {
        return;
    }

    int idx = atomicAdd(active_count, 1);
    int token = pair / topk;
    output_permutation[pair] = idx;
    problems[idx] = cutlass::gemm::GemmCoord(1, intermediate_size * 2, hidden_size);
    ptr_a[idx] = const_cast<T *>(hidden + static_cast<size_t>(token) * hidden_size);
    ptr_b[idx] = const_cast<T *>(w13 + static_cast<size_t>(expert) * intermediate_size * 2 * hidden_size);
    ptr_c[idx] = gate_up + static_cast<size_t>(idx) * intermediate_size * 2;
    ptr_d[idx] = gate_up + static_cast<size_t>(idx) * intermediate_size * 2;
    lda[idx] = hidden_size;
    ldb[idx] = hidden_size;
    ldc[idx] = intermediate_size * 2;
    ldd[idx] = intermediate_size * 2;
}

template <typename T>
__global__ void setup_decode_gemm2_aligned_compact_kernel(cutlass::gemm::GemmCoord *problems,
                                                          void **ptr_a,
                                                          void **ptr_b,
                                                          void **ptr_c,
                                                          void **ptr_d,
                                                          int64_t *lda,
                                                          int64_t *ldb,
                                                          int64_t *ldc,
                                                          int64_t *ldd,
                                                          const T *activated,
                                                          const T *w2,
                                                          T *expert_out,
                                                          const int *sorted_token_ids,
                                                          const int *expert_ids,
                                                          const int *num_tokens_post_padded,
                                                          const int *output_permutation,
                                                          int pairs,
                                                          int num_experts,
                                                          int hidden_size,
                                                          int intermediate_size,
                                                          int block_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = *num_tokens_post_padded;
    if (row >= total_rows) {
        return;
    }
    int pair = sorted_token_ids[row];
    if (pair < 0 || pair >= pairs) {
        return;
    }
    int expert = expert_ids[row / block_size];
    if (expert < 0 || expert >= num_experts) {
        return;
    }
    int idx = output_permutation[pair];
    if (idx < 0) {
        return;
    }
    problems[idx] = cutlass::gemm::GemmCoord(1, hidden_size, intermediate_size);
    ptr_a[idx] = const_cast<T *>(activated + static_cast<size_t>(idx) * intermediate_size);
    ptr_b[idx] = const_cast<T *>(w2 + static_cast<size_t>(expert) * hidden_size * intermediate_size);
    ptr_c[idx] = expert_out + static_cast<size_t>(idx) * hidden_size;
    ptr_d[idx] = expert_out + static_cast<size_t>(idx) * hidden_size;
    lda[idx] = intermediate_size;
    ldb[idx] = intermediate_size;
    ldc[idx] = hidden_size;
    ldd[idx] = hidden_size;
}

template <typename CutlassT>
MoeStatus launch_cutlass_gemm_grouped_device_meta(int problem_count,
                                                       cutlass::gemm::GemmCoord *d_problems,
                                                       void **d_ptr_a,
                                                       void **d_ptr_b,
                                                       void **d_ptr_c,
                                                       void **d_ptr_d,
                                                       int64_t *d_lda,
                                                       int64_t *d_ldb,
                                                       int64_t *d_ldc,
                                                       int64_t *d_ldd,
                                                       bool full_occupancy_wave,
                                                       bool use_bias,
                                                       cudaStream_t stream) {
    if (problem_count == 0) {
        return MOE_SUCCESS;
    }

    using Element = CutlassT;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using OutputOp = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        Element,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        8,
        Element,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        8,
        Element,
        LayoutC,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        OutputOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    const int occupancy_wave = Gemm::sufficient();
    // Decode routes to only top-k experts, but each expert still contains many
    // N-dimension tiles. A full persistent wave lets CUTLASS distribute those
    // tiles across the GPU instead of leaving most SMs idle. Prefill retains
    // the previous problem-count cap to avoid scheduler overhead from excess
    // threadblocks when most experts receive no tokens.
    const int threadblock_count = full_occupancy_wave
                                    ? occupancy_wave
                                    : std::min(occupancy_wave, problem_count);
    if (threadblock_count <= 0) {
        return MOE_DEVICE;
    }

    typename Gemm::EpilogueOutputOp::Params epilogue_op(1.0f, use_bias ? 1.0f : 0.0f);
    typename Gemm::Arguments args(
        d_problems,
        problem_count,
        threadblock_count,
        epilogue_op,
        reinterpret_cast<Element **>(d_ptr_a),
        reinterpret_cast<Element **>(d_ptr_b),
        reinterpret_cast<Element **>(d_ptr_c),
        reinterpret_cast<Element **>(d_ptr_d),
        d_lda,
        d_ldb,
        d_ldc,
        d_ldd,
        nullptr);

    Gemm gemm;
    auto status = gemm(args, nullptr, stream);
    return status == cutlass::Status::kSuccess ? MOE_SUCCESS : MOE_INTERNAL;
}

template <typename T, typename CutlassT>
MoeStatus calculate_typed(const MoeFusedDenseInfo &info,
                               void *workspace,
                               size_t workspace_size,
                               void *output,
                               const void *hidden_states,
                               const void *w13,
                               const void *w2,
                               const void *topk_weights,
                               const void *topk_ids,
                               const void *sorted_token_ids,
                               const void *expert_ids,
                               const void *num_tokens_post_padded,
                               const void *bias1,
                               const void *bias2,
                               cudaStream_t stream) {
    const int num_tokens = static_cast<int>(info.num_tokens);
    const int hidden_size = static_cast<int>(info.hidden_size);
    const int num_experts = static_cast<int>(info.num_experts);
    const int intermediate_size = static_cast<int>(info.intermediate_size);
    const int topk = static_cast<int>(info.topk);
    const int pairs = num_tokens * topk;
    const int max_num_tokens_padded = static_cast<int>(info.max_num_tokens_padded);
    const int block_size = static_cast<int>(info.block_size);

    uint8_t *ptr = reinterpret_cast<uint8_t *>(workspace);
    size_t remaining = workspace_size;
    auto counts = reinterpret_cast<int *>(advance_workspace(ptr, remaining, (num_experts + 1) * sizeof(int)));
    auto offsets = reinterpret_cast<int *>(advance_workspace(ptr, remaining, (num_experts + 1) * sizeof(int)));
    auto output_permutation = reinterpret_cast<int *>(advance_workspace(ptr, remaining, pairs * sizeof(int)));
    auto packed_hidden = reinterpret_cast<T *>(advance_workspace(ptr, remaining, static_cast<size_t>(max_num_tokens_padded) * hidden_size * sizeof(T)));
    auto gate_up = reinterpret_cast<T *>(advance_workspace(ptr, remaining, static_cast<size_t>(max_num_tokens_padded) * intermediate_size * 2 * sizeof(T)));
    auto activated = reinterpret_cast<T *>(advance_workspace(ptr, remaining, static_cast<size_t>(max_num_tokens_padded) * intermediate_size * sizeof(T)));
    auto expert_out = reinterpret_cast<T *>(advance_workspace(ptr, remaining, static_cast<size_t>(max_num_tokens_padded) * hidden_size * sizeof(T)));
    auto grouped_problems = reinterpret_cast<cutlass::gemm::GemmCoord *>(
        advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(cutlass::gemm::GemmCoord), alignof(cutlass::gemm::GemmCoord)));
    auto grouped_ptr_a = reinterpret_cast<void **>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_b = reinterpret_cast<void **>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_c = reinterpret_cast<void **>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_d = reinterpret_cast<void **>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_lda = reinterpret_cast<int64_t *>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldb = reinterpret_cast<int64_t *>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldc = reinterpret_cast<int64_t *>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldd = reinterpret_cast<int64_t *>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto active_count = reinterpret_cast<int *>(advance_workspace(ptr, remaining, sizeof(int), alignof(int)));

    if (!counts || !offsets || !output_permutation || !packed_hidden || !gate_up || !activated || !expert_out || !grouped_problems || !grouped_ptr_a || !grouped_ptr_b || !grouped_ptr_c || !grouped_ptr_d || !grouped_lda || !grouped_ldb || !grouped_ldc || !grouped_ldd || !active_count) {
        return MOE_WORKSPACE;
    }

    auto w13_t = reinterpret_cast<const T *>(w13);
    auto w2_t = reinterpret_cast<const T *>(w2);
    if (num_tokens == 1 && topk <= num_experts) {
        cudaMemsetAsync(output_permutation, 0xff, pairs * sizeof(int), stream);
        cudaMemsetAsync(grouped_problems, 0, topk * sizeof(cutlass::gemm::GemmCoord), stream);
        cudaMemsetAsync(active_count, 0, sizeof(int), stream);
        setup_decode_gemm1_aligned_compact_kernel<T><<<(max_num_tokens_padded + 255) / 256, 256, 0, stream>>>(
            grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
            grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, active_count, output_permutation,
            reinterpret_cast<const T *>(hidden_states), w13_t, gate_up,
            reinterpret_cast<const int *>(sorted_token_ids),
            reinterpret_cast<const int *>(expert_ids),
            reinterpret_cast<const int *>(num_tokens_post_padded),
            pairs, topk, num_experts, hidden_size, intermediate_size, block_size);

        if (bias1 || bias2) {
            fill_decode_bias_kernel<T><<<max_num_tokens_padded, 256, 0, stream>>>(
                gate_up, expert_out, reinterpret_cast<const T *>(bias1), reinterpret_cast<const T *>(bias2),
                reinterpret_cast<const int *>(sorted_token_ids), reinterpret_cast<const int *>(expert_ids),
                reinterpret_cast<const int *>(num_tokens_post_padded), output_permutation,
                pairs, num_experts, intermediate_size, hidden_size, block_size);
            MOE_CHECK_CUDA(cudaGetLastError());
        }

        // Decode has one token and exactly one valid aligned row for each of
        // its top-k routes, so the compact problem count is always topk.
        // A D2H count copy and stream sync cannot be captured for replay.
        MOE_CHECK_STATUS(launch_cutlass_gemm_grouped_device_meta<CutlassT>(
            topk, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
            grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, true, bias1 != nullptr, stream));

        swiglu_kernel<T><<<(topk * intermediate_size + 255) / 256, 256, 0, stream>>>(gate_up, activated, topk, intermediate_size);

        setup_decode_gemm2_aligned_compact_kernel<T><<<(max_num_tokens_padded + 255) / 256, 256, 0, stream>>>(
            grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
            grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, activated, w2_t, expert_out,
            reinterpret_cast<const int *>(sorted_token_ids),
            reinterpret_cast<const int *>(expert_ids),
            reinterpret_cast<const int *>(num_tokens_post_padded),
            output_permutation, pairs, num_experts, hidden_size, intermediate_size, block_size);
        MOE_CHECK_STATUS(launch_cutlass_gemm_grouped_device_meta<CutlassT>(
            topk, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
            grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, true, bias2 != nullptr, stream));

        apply_shuffle_mul_sum_kernel<T><<<1, std::min(hidden_size, 1024), 0, stream>>>(
            expert_out, reinterpret_cast<T *>(output), output_permutation,
            reinterpret_cast<const float *>(topk_weights), 1, topk, hidden_size);
        return MOE_SUCCESS;
    }

    cudaMemsetAsync(output_permutation, 0xff, pairs * sizeof(int), stream);
    cudaMemsetAsync(counts, 0, (num_experts + 1) * sizeof(int), stream);
    count_aligned_experts_kernel<<<(info.max_num_blocks + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const int *>(expert_ids),
        reinterpret_cast<const int *>(num_tokens_post_padded),
        counts,
        num_experts,
        block_size);

    int scan_threads = 1;
    while (scan_threads < num_experts) {
        scan_threads <<= 1;
    }
    scan_threads = std::max(32, scan_threads);
    exclusive_prefix_counts_kernel<<<1, scan_threads, scan_threads * sizeof(int), stream>>>(counts, offsets, num_experts);

    pack_hidden_aligned_kernel<T><<<max_num_tokens_padded, 256, 0, stream>>>(
        reinterpret_cast<const T *>(hidden_states),
        reinterpret_cast<const int *>(sorted_token_ids),
        output_permutation,
        packed_hidden,
        pairs,
        topk,
        hidden_size,
        max_num_tokens_padded);

    setup_prefill_gemm1_kernel<T><<<(num_experts + 255) / 256, 256, 0, stream>>>(
        grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, counts, offsets, packed_hidden,
        w13_t, gate_up, num_experts, hidden_size, intermediate_size);

    if (bias1 || bias2) {
        fill_prefill_bias_kernel<T><<<num_experts, 256, 0, stream>>>(
            gate_up, expert_out, reinterpret_cast<const T *>(bias1), reinterpret_cast<const T *>(bias2),
            counts, offsets, intermediate_size, hidden_size);
        MOE_CHECK_CUDA(cudaGetLastError());
    }

    // Keep one grouped problem per expert. Experts with no routed tokens have
    // m == 0 and are skipped by CUTLASS's device-only grouped scheduler. This
    // avoids making the runtime problem count depend on device routing data.
    MOE_CHECK_STATUS(launch_cutlass_gemm_grouped_device_meta<CutlassT>(
        num_experts, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, false, bias1 != nullptr, stream));

    swiglu_kernel<T><<<(max_num_tokens_padded * intermediate_size + 255) / 256, 256, 0, stream>>>(
        gate_up, activated, max_num_tokens_padded, intermediate_size);

    setup_prefill_gemm2_kernel<T><<<(num_experts + 255) / 256, 256, 0, stream>>>(
        grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, counts, offsets, activated,
        w2_t, expert_out, num_experts, hidden_size, intermediate_size);
    MOE_CHECK_STATUS(launch_cutlass_gemm_grouped_device_meta<CutlassT>(
        num_experts, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, false, bias2 != nullptr, stream));

    apply_shuffle_mul_sum_kernel<T><<<num_tokens, std::min(hidden_size, 1024), 0, stream>>>(
        expert_out, reinterpret_cast<T *>(output), output_permutation,
        reinterpret_cast<const float *>(topk_weights), num_tokens, topk, hidden_size);
    return MOE_SUCCESS;
}

} // namespace
Operator<MoeFusedDense, Device::Type::kNvidia, 0>::Operator(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1, const std::optional<Tensor> bias2) : MoeFusedDense(output, hidden, w13, w2, weights, ids, sorted, experts, total, block_size, bias1, bias2) { moe_detail::Require(SupportsLayout(hidden,w13,w2)); }
size_t Operator<MoeFusedDense, Device::Type::kNvidia, 0>::workspace_size_in_bytes() const {
  const auto& v=metadata_; MoeFusedDenseInfo info{v[1].size(0),v[1].size(1),v[2].size(0),v[3].size(2),v[5].size(1),v[6].numel(),v[7].numel(),block_size_,v[1].dtype()};
  return workspace_size(info);
}
void Operator<MoeFusedDense, Device::Type::kNvidia, 0>::operator()(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1, const std::optional<Tensor> bias2) const {
  MoeDeviceGuard device_guard{metadata_[0].device().index()};
  Invocation(output, hidden, w13, w2, weights, ids, sorted, experts, total, block_size, bias1, bias2); MoeFusedDenseInfo info{hidden.size(0),hidden.size(1),w13.size(0),w2.size(2),ids.size(1),sorted.numel(),experts.numel(),block_size,hidden.dtype()};
  moe_detail::Require(SupportsLayout(hidden,w13,w2));
  std::vector<Tensor> views{output, hidden, w13, w2, weights, ids, sorted, experts, total};
  if (bias1) views.push_back(*bias1);
  if (bias2) views.push_back(*bias2);
  moe_detail::Workspace(workspace_,workspace_size_in_bytes_,workspace_size(info),views);
  auto stream=static_cast<cudaStream_t>(stream_);
  if(hidden.dtype()==DataType::kFloat16) Check(calculate_typed<half,cutlass::half_t>(info,workspace_,workspace_size_in_bytes_,output.data(),hidden.data(),w13.data(),w2.data(),weights.data(),ids.data(),sorted.data(),experts.data(),total.data(),bias1 ? bias1->data() : nullptr,bias2 ? bias2->data() : nullptr,stream));
  else Check(calculate_typed<__nv_bfloat16,cutlass::bfloat16_t>(info,workspace_,workspace_size_in_bytes_,output.data(),hidden.data(),w13.data(),w2.data(),weights.data(),ids.data(),sorted.data(),experts.data(),total.data(),bias1 ? bias1->data() : nullptr,bias2 ? bias2->data() : nullptr,stream));
  MOE_CHECK_CUDA(cudaGetLastError());
}

#undef MOE_CHECK_STATUS
#undef MOE_CHECK_CUDA
} // namespace infini::ops
