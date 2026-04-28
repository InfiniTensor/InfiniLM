#include "../../../devices/moore/moore_kernel_common.h"
#include "infinicore.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <type_traits>

// 辅助函数：__mt_bfloat16 到 float 的转换核函数
__global__ void bfloat16_to_float_kernel(const __mt_bfloat16 *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (float)in[i];
    }
}

// 辅助函数：float 到 __mt_bfloat16 的转换核函数
__global__ void float_to_bfloat16_kernel(const float *in, __mt_bfloat16 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (__mt_bfloat16)in[i];
    }
}

// device-side helper：把 float 类型的 kv_pair 写入 bf16 类型的 kv_pair
__global__ void write_kv_bfloat16_kernel(
    cub::KeyValuePair<int, __mt_bfloat16> *dst,
    const cub::KeyValuePair<int, float> *src) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst->key = src->key;
        dst->value = (__mt_bfloat16)src->value;
    }
}

namespace op::random_sample::moore {

// 地址对齐到 256
static constexpr size_t align256(size_t size) {
    return (size + 255) & (~255);
}

// ↓↓↓ 重新封装 cub api，减少模板参数，方便调用

template <class T>
static musaError_t argMax_(
    cub::KeyValuePair<int, T> *kv_pair, // device ptr for result (value type T)
    const T *logits,                    // device ptr logits
    int n,
    void *workspace_ptr,
    size_t &workspace_len,
    musaStream_t stream) {

    if constexpr (std::is_same_v<T, __mt_bfloat16>) {
        // 为 device 侧 float kv 预留空间
        size_t kv_needed = align256(sizeof(cub::KeyValuePair<int, float>));
        if (workspace_len < kv_needed) {
            return musaErrorInvalidValue;
        }
        auto *temp_kv_dev = reinterpret_cast<cub::KeyValuePair<int, float> *>(workspace_ptr);
        workspace_ptr = (void *)((char *)workspace_ptr + kv_needed);
        workspace_len -= kv_needed;

        // 为 temp_logits（bf16->float）预留空间，并前移工作区，避免与 CUB 工作区重叠
        size_t logits_bytes = align256(sizeof(float) * (size_t)n);
        if (workspace_len < logits_bytes) {
            return musaErrorInvalidValue;
        }
        float *temp_logits = reinterpret_cast<float *>(workspace_ptr);
        workspace_ptr = (void *)((char *)workspace_ptr + logits_bytes);
        workspace_len -= logits_bytes;

        // 现在 workspace_ptr/workspace_len 才交给 CUB 使用，确保不覆盖 temp_logits
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        // bf16 -> float
        bfloat16_to_float_kernel<<<grid_size, block_size, 0, stream>>>(logits, temp_logits, n);

        // CUB ArgMax：输入 temp_logits(float)，输出 temp_kv_dev(float)
        musaError_t err = cub::DeviceReduce::ArgMax(
            workspace_ptr, workspace_len,
            temp_logits, temp_kv_dev, n,
            stream);
        if (err != musaSuccess) {
            return err;
        }

        // 把 float kv_pair 写回 bf16 kv_pair
        write_kv_bfloat16_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<cub::KeyValuePair<int, __mt_bfloat16> *>(kv_pair),
            temp_kv_dev);

        return musaSuccess;
    } else {
        // 非 bf16 直接调用（原有逻辑）
        return cub::DeviceReduce::ArgMax(
            workspace_ptr, workspace_len,
            logits, kv_pair, n,
            stream);
    }
}

template <class Tval, class Tidx>
static musaError_t radixSort(
    void *workspace_ptr, size_t &workspace_len,
    const Tval *key_in, Tval *key_out,
    const Tidx *val_in, Tidx *val_out,
    int n,
    musaStream_t stream) {

    if constexpr (std::is_same_v<Tval, __mt_bfloat16>) {
        // 为 float 转换缓冲做 256 对齐切分
        size_t buf = align256(sizeof(float) * (size_t)n);
        if (workspace_len < buf) {
            return musaErrorInvalidValue;
        }

        float *temp_key_in = reinterpret_cast<float *>(workspace_ptr);
        workspace_ptr = (void *)((char *)workspace_ptr + buf);
        workspace_len -= buf;

        if (workspace_len < buf) {
            return musaErrorInvalidValue;
        }

        float *temp_key_out = reinterpret_cast<float *>(workspace_ptr);
        workspace_ptr = (void *)((char *)workspace_ptr + buf);
        workspace_len -= buf;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        // bf16 -> float
        bfloat16_to_float_kernel<<<grid_size, block_size, 0, stream>>>(key_in, temp_key_in, n);

        // CUB 工作区用剩余空间
        musaError_t err = cub::DeviceRadixSort::SortPairsDescending(
            workspace_ptr, workspace_len,
            temp_key_in, temp_key_out,
            val_in, val_out,
            n,
            0, sizeof(float) * 8,
            stream);
        if (err != musaSuccess) {
            return err;
        }

        // float -> bf16 写回
        float_to_bfloat16_kernel<<<grid_size, block_size, 0, stream>>>(temp_key_out, key_out, n);

        return musaSuccess;
    } else {
        return cub::DeviceRadixSort::SortPairsDescending(
            workspace_ptr, workspace_len,
            key_in, key_out,
            val_in, val_out,
            n,
            0, sizeof(Tval) * 8,
            stream);
    }
}

template <class T>
static musaError_t inclusiveSum(
    void *workspace_ptr, size_t &workspace_len,
    T *data, int n,
    musaStream_t stream) {

    if constexpr (std::is_same_v<T, __mt_bfloat16>) {
        // 为 float 临时缓冲做 256B 对齐切分
        size_t buf = align256(sizeof(float) * (size_t)n);
        if (workspace_len < buf) {
            return musaErrorInvalidValue;
        }
        float *temp_data = reinterpret_cast<float *>(workspace_ptr);
        workspace_ptr = (void *)((char *)workspace_ptr + buf);
        workspace_len -= buf;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        // bf16 -> float
        bfloat16_to_float_kernel<<<grid_size, block_size, 0, stream>>>(data, temp_data, n);

        // CUB 用剩余空间
        musaError_t err = cub::DeviceScan::InclusiveSum(
            workspace_ptr, workspace_len,
            temp_data, temp_data, n,
            stream);

        // float -> bf16
        float_to_bfloat16_kernel<<<grid_size, block_size, 0, stream>>>(temp_data, data, n);

        return err;
    } else {
        return cub::DeviceScan::InclusiveSum(
            workspace_ptr, workspace_len,
            data, data, n,
            stream);
    }
}

// ↑↑↑ 重新封装 cub api，减少模板参数，方便调用
// ↓↓↓ 计算 workspace

template <class Tidx, class Tval>
utils::Result<size_t> calculateWorkspace(size_t n_) {
    const auto n = static_cast<int>(n_);

    size_t argmax;
    // 使用一个伪造的指针来解决 nullptr 类型推断问题
    cub::KeyValuePair<int, Tval> *fake_kv = nullptr;
    const Tval *fake_logits = nullptr;

    if constexpr (std::is_same_v<Tval, __mt_bfloat16>) {
        // bf16 用 float 版本来 query
        cub::DeviceReduce::ArgMax(
            nullptr, argmax,
            (const float *)fake_logits, (cub::KeyValuePair<int, float> *)fake_kv, n,
            nullptr);
    } else {
        cub::DeviceReduce::ArgMax(
            nullptr, argmax,
            fake_logits, fake_kv, n,
            nullptr);
    }

    // 前 256 字节用于 kv pair
    argmax += 256;

    // indices / sorted / indices_out（主缓冲）
    size_t size_random = align256(sizeof(Tidx) * n); // indices
    size_random += align256(sizeof(Tval) * n);       // sorted
    size_random += align256(sizeof(Tidx) * n);       // indices_out

    // CUB device api workspace
    size_t size_radix_sort;
    const Tval *fake_key_in = nullptr;
    Tval *fake_key_out = nullptr;
    const Tidx *fake_val_in = nullptr;
    Tidx *fake_val_out = nullptr;

    if constexpr (std::is_same_v<Tval, __mt_bfloat16>) {
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, size_radix_sort,
            (const float *)fake_key_in, (float *)fake_key_out,
            fake_val_in, fake_val_out,
            n,
            0, sizeof(float) * 8,
            nullptr);
    } else {
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, size_radix_sort,
            fake_key_in, fake_key_out,
            fake_val_in, fake_val_out,
            n,
            0, sizeof(Tval) * 8,
            nullptr);
    }

    size_t size_inclusive_sum;
    Tval *fake_data = nullptr;
    cub::DeviceScan::InclusiveSum(
        nullptr, size_inclusive_sum,
        fake_data, fake_data, n,
        nullptr);

    size_random += cub::Max()(size_radix_sort, size_inclusive_sum);

    // 额外的临时显存开销：bf16 需要 4*n*sizeof(float)
    //  - ArgMax: 1n (temp_logits)
    //  - InclusiveSum: 1n (temp_data)
    //  - RadixSort: 2n (temp_key_in/out)
    if constexpr (std::is_same_v<Tval, __mt_bfloat16>) {
        size_random += align256(sizeof(float) * (size_t)n) * 4;
    }

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

template <>
struct CudaTval<bf16_t> {
    using Type = __mt_bfloat16;
};

// ↑↑↑ 通过特化将 fp16_t 转换为 half
// ↓↓↓ 用于采样过程的小型 kernel

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
        // MUSA not support '__ldg'
        float maxv = (float)data[0];
        data[i] = (T)expf(((float)data[i] - maxv) / temperature);
    }
}

// 将第一个数字写成 1，即 exp(0)
template <class T>
static __global__ void setSoftmaxMaxKernel(T *__restrict__ data) {
    // 将 1.0f 转换为 T 类型，以消除编译器 int 到 __mt_bfloat16 赋值时的二义性
    *data = (T)(1.0f);
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

        auto stream = (musaStream_t)stream_;
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

        auto stream = (musaStream_t)stream_;
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
        CHECK_MOORE(radixSort(
            workspace_, workspace_size,
            logits, sorted,
            indices, indices_out,
            n,
            stream));
        // softmax
        partialSoftmaxKernel<<<grid, block, 0, stream>>>(sorted, n, temperature);
        setSoftmaxMaxKernel<<<1, 1, 0, stream>>>(sorted);
        // sum
        CHECK_MOORE(inclusiveSum(
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

} // namespace op::random_sample::moore
