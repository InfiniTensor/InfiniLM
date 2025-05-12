#include "../../../devices/cuda/cuda_kernel_common.cuh"
#include "infinicore.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

namespace op::random_sample::cuda {

// ↓↓↓ 重新封装 cub api，减少模板参数，方便调用

template <class T>
static cudaError argMax_(
    cub::KeyValuePair<int, T> *kv_pair,
    const T *logits,
    int n,
    void *workspace_ptr,
    size_t &workspace_len,
    cudaStream_t stream) {
    return cub::DeviceReduce::ArgMax(
        workspace_ptr, workspace_len,
        logits, kv_pair, n,
        stream);
}

template <class Tval, class Tidx>
static cudaError radixSort(
    void *workspace_ptr, size_t &workspace_len,
    const Tval *key_in, Tval *key_out,
    const Tidx *val_in, Tidx *val_out,
    int n,
    cudaStream_t stream) {
    return cub::DeviceRadixSort::SortPairsDescending(
        workspace_ptr, workspace_len,
        key_in, key_out,
        val_in, val_out,
        n,
        0, sizeof(Tval) * 8,
        stream);
}

template <class T>
static cudaError inclusiveSum(
    void *workspace_ptr, size_t &workspace_len,
    T *data, int n,
    cudaStream_t stream) {
    return cub::DeviceScan::InclusiveSum(
        workspace_ptr, workspace_len,
        data, data, n,
        stream);
}

// ↑↑↑ 重新封装 cub api，减少模板参数，方便调用
// ↓↓↓ 计算 workspace

// 地址对齐到 256
static constexpr size_t align256(size_t size) {
    return (size + 255) & (~255);
}

template <class Tidx, class Tval>
utils::Result<size_t> calculateWorkspace(size_t n_) {
    const auto n = static_cast<int>(n_);

    size_t argmax;
    CHECK_CUDA(argMax_<Tval>(
        nullptr, nullptr, n,
        nullptr, argmax,
        nullptr));
    // 前 256 字节用于 kv pair
    argmax += 256;

    // indices
    size_t size_random = align256(sizeof(Tidx) * n);
    // sorted
    size_random += align256(sizeof(Tval) * n);
    // indices_out
    size_random += align256(sizeof(Tidx) * n);
    // cub device api
    size_t size_radix_sort;
    CHECK_CUDA((radixSort<Tval, Tidx>(
        nullptr, size_radix_sort,
        nullptr, nullptr,
        nullptr, nullptr,
        n,
        nullptr)));

    size_t size_inclusive_sum;
    CHECK_CUDA(inclusiveSum<Tval>(
        nullptr, size_inclusive_sum,
        nullptr, n,
        nullptr));
    size_random += cub::Max()(size_radix_sort, size_inclusive_sum);

    return utils::Result<size_t>(cub::Max()(argmax, size_random));
}

// ↑↑↑ 计算 workspace
// ↓↓↓ 通过特化将 fp16_t 转换为 half

template <class Tval>
struct CudaTval {
    using Type = Tval;
};

template <>
struct CudaTval<fp16_t> {
    using Type = half;
};

// ↑↑↑ 通过特化将 fp16_t 转换为 half
// ↓↓↓ 用于采样过程的小型 kernel

// cuda toolkit 11.x 带的 cub::DeviceReduce::ArgMax 只接受 cub::KeyValuePair<int, Tval> 输出。
// 这个 kernel 用于取出序号
template <class Tidx, class Tval>
static __global__ void castIdx(Tidx *result, const cub::KeyValuePair<int, Tval> *kv_pair) {
    *result = kv_pair->key;
}

// 填充排序要求的序号数组
template <class Tidx>
static __global__ void fillIndices(Tidx *indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        indices[i] = i;
    }
}

// random sample 使用的 softmax 可以简化为一个基本的线性映射
// 由于已经排序，最大值就是第一个数字
// 第一个数字需要被多个 block 读取，不能写
template <class T>
static __global__ void partialSoftmaxKernel(
    T *__restrict__ data, int n,
    float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < i && i < n) {
        float max = __ldg(data);
        data[i] = (T)expf(((float)data[i] - max) / temperature);
    }
}

// 将第一个数字写成 1，即 exp(0)
template <class T>
static __global__ void setSoftmaxMaxKernel(
    T *__restrict__ data) {
    *data = 1;
}

// 直接 for 循环遍历采样
// 这个 kernel 仅用于避免将数据拷贝到 cpu
template <class Tval, class Tidx>
static __global__ void randomSampleKernel(
    Tidx *__restrict__ result,
    const Tval *__restrict__ sorted,
    const Tidx *__restrict__ indices_out,
    size_t n,
    float random, float topp, size_t topk) {
    topk = cub::Min()(topk, n);
    auto p = (Tval)(random * cub::Min()(topp * (float)sorted[n - 1], (float)sorted[topk - 1]));
    for (size_t i = 0;; ++i) {
        if ((sorted[i]) >= p) {
            *result = indices_out[i];
            return;
        }
    }
}

// ↑↑↑ 用于采样过程的小型 kernel

struct Algo {
    int block_size;

    template <class Tidx, class Tval_>
    infiniStatus_t argmax(
        void *workspace, size_t workspace_size,
        void *result, const void *probs, size_t n,
        void *stream_) const {

        using Tval = typename CudaTval<Tval_>::Type;

        auto stream = (cudaStream_t)stream_;
        auto logits = (Tval *)probs;
        auto kv_pair = (cub::KeyValuePair<int, Tval> *)workspace;
        workspace = (void *)((char *)workspace + 256);
        workspace_size -= 256;

        argMax_(
            kv_pair,
            logits,
            n,
            workspace,
            workspace_size, stream);
        castIdx<<<1, 1, 0, stream>>>((Tidx *)result, kv_pair);

        return INFINI_STATUS_SUCCESS;
    }

    template <class Tidx, class Tval_>
    infiniStatus_t random(
        void *workspace_, size_t workspace_size,
        void *result_, const void *probs, size_t n,
        float random_val, float topp, int topk, float temperature,
        void *stream_) const {

        using Tval = typename CudaTval<Tval_>::Type;

        auto stream = (cudaStream_t)stream_;
        auto logits = (Tval *)probs;
        auto result = (Tidx *)result_;

        auto workspace = reinterpret_cast<size_t>(workspace_);
        auto workspace_end = workspace + workspace_size;

        auto indices = reinterpret_cast<Tidx *>(workspace);
        workspace += align256(sizeof(Tidx) * n);

        auto sorted = reinterpret_cast<Tval *>(workspace);
        workspace += align256(sizeof(Tval) * n);

        auto indices_out = reinterpret_cast<Tidx *>(workspace);
        workspace += align256(sizeof(Tidx) * n);

        workspace_ = reinterpret_cast<void *>(workspace);
        workspace_size = workspace_end - workspace;

        auto block = cub::Min()((size_t)block_size, n);
        auto grid = (n + block - 1) / block;
        // sort
        fillIndices<<<grid, block, 0, stream>>>(indices, n);
        CHECK_CUDA(radixSort(
            workspace_, workspace_size,
            logits, sorted,
            indices, indices_out,
            n,
            stream));
        // softmax
        partialSoftmaxKernel<<<grid, block, 0, stream>>>(sorted, n, temperature);
        setSoftmaxMaxKernel<<<1, 1, 0, stream>>>(sorted);
        // sum
        CHECK_CUDA(inclusiveSum(
            workspace_, workspace,
            sorted, n,
            stream));
        // sample
        randomSampleKernel<<<1, 1, 0, stream>>>(
            result,
            sorted, indices_out, n,
            random_val, topp, topk);
        return INFINI_STATUS_SUCCESS;
    }
};

} // namespace op::random_sample::cuda
