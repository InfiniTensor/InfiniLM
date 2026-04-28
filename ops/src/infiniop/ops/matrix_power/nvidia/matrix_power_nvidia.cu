#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "matrix_power_nvidia.cuh"
#include <limits>
#include <utility>

namespace op::matrix_power::nvidia {

namespace {

template <typename T>
__forceinline__ __device__ T identityZero();

template <typename T>
__forceinline__ __device__ T identityOne();

template <>
__forceinline__ __device__ __half identityZero<__half>() {
    return __float2half(0.0f);
}

template <>
__forceinline__ __device__ __half identityOne<__half>() {
    return __float2half(1.0f);
}

template <>
__forceinline__ __device__ cuda_bfloat16 identityZero<cuda_bfloat16>() {
    return __float2bfloat16(0.0f);
}

template <>
__forceinline__ __device__ cuda_bfloat16 identityOne<cuda_bfloat16>() {
    return __float2bfloat16(1.0f);
}

template <>
__forceinline__ __device__ float identityZero<float>() {
    return 0.0f;
}

template <>
__forceinline__ __device__ float identityOne<float>() {
    return 1.0f;
}

template <>
__forceinline__ __device__ double identityZero<double>() {
    return 0.0;
}

template <>
__forceinline__ __device__ double identityOne<double>() {
    return 1.0;
}

template <typename T>
INFINIOP_CUDA_KERNEL packMatrix2dStridedToContiguous(
    const T *src,
    T *dst,
    size_t matrix_size,
    ptrdiff_t src_stride_0,
    ptrdiff_t src_stride_1) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t matrix_numel = matrix_size * matrix_size;
    if (idx < matrix_numel) {
        size_t row = idx / matrix_size;
        size_t col = idx - row * matrix_size;
        dst[idx] = src[row * src_stride_0 + col * src_stride_1];
    }
}

template <typename T>
INFINIOP_CUDA_KERNEL scatterMatrix2dContiguousToStrided(
    const T *src,
    T *dst,
    size_t matrix_size,
    ptrdiff_t dst_stride_0,
    ptrdiff_t dst_stride_1) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t matrix_numel = matrix_size * matrix_size;
    if (idx < matrix_numel) {
        size_t row = idx / matrix_size;
        size_t col = idx - row * matrix_size;
        dst[row * dst_stride_0 + col * dst_stride_1] = src[idx];
    }
}

template <typename T>
INFINIOP_CUDA_KERNEL setIdentity2dStrided(
    T *out,
    size_t matrix_size,
    ptrdiff_t out_stride_0,
    ptrdiff_t out_stride_1) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t matrix_numel = matrix_size * matrix_size;
    if (idx < matrix_numel) {
        size_t row = idx / matrix_size;
        size_t col = idx - row * matrix_size;
        out[row * out_stride_0 + col * out_stride_1] = (row == col) ? identityOne<T>() : identityZero<T>();
    }
}

INFINIOP_CUDA_KERNEL setDiagonalFp16(__half *out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx * n + idx] = __float2half(1.0f);
    }
}

INFINIOP_CUDA_KERNEL setDiagonalBf16(cuda_bfloat16 *out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx * n + idx] = __float2bfloat16(1.0f);
    }
}

INFINIOP_CUDA_KERNEL setDiagonalFp32(float *out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx * n + idx] = 1.0f;
    }
}

INFINIOP_CUDA_KERNEL setDiagonalFp64(double *out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx * n + idx] = 1.0;
    }
}

infiniStatus_t initializeIdentity(
    void *y,
    infiniDtype_t dtype,
    size_t matrix_size,
    bool y_contiguous,
    ptrdiff_t y_stride_0,
    ptrdiff_t y_stride_1,
    cudaStream_t stream) {

    if (matrix_size == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    constexpr int threads = 256;
    size_t diag_blocks = CEIL_DIV(matrix_size, static_cast<size_t>(threads));
    size_t matrix_numel = matrix_size * matrix_size;
    size_t matrix_blocks = CEIL_DIV(matrix_numel, static_cast<size_t>(threads));

    if (y_contiguous) {
        CHECK_CUDA(cudaMemsetAsync(y, 0, matrix_numel * infiniSizeOf(dtype), stream));
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        if (y_contiguous) {
            setDiagonalFp16<<<static_cast<unsigned int>(diag_blocks), threads, 0, stream>>>(
                reinterpret_cast<__half *>(y), matrix_size);
        } else {
            setIdentity2dStrided<<<static_cast<unsigned int>(matrix_blocks), threads, 0, stream>>>(
                reinterpret_cast<__half *>(y), matrix_size, y_stride_0, y_stride_1);
        }
        break;
    case INFINI_DTYPE_BF16:
        if (y_contiguous) {
            setDiagonalBf16<<<static_cast<unsigned int>(diag_blocks), threads, 0, stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y), matrix_size);
        } else {
            setIdentity2dStrided<<<static_cast<unsigned int>(matrix_blocks), threads, 0, stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y), matrix_size, y_stride_0, y_stride_1);
        }
        break;
    case INFINI_DTYPE_F32:
        if (y_contiguous) {
            setDiagonalFp32<<<static_cast<unsigned int>(diag_blocks), threads, 0, stream>>>(
                reinterpret_cast<float *>(y), matrix_size);
        } else {
            setIdentity2dStrided<<<static_cast<unsigned int>(matrix_blocks), threads, 0, stream>>>(
                reinterpret_cast<float *>(y), matrix_size, y_stride_0, y_stride_1);
        }
        break;
    case INFINI_DTYPE_F64:
        if (y_contiguous) {
            setDiagonalFp64<<<static_cast<unsigned int>(diag_blocks), threads, 0, stream>>>(
                reinterpret_cast<double *>(y), matrix_size);
        } else {
            setIdentity2dStrided<<<static_cast<unsigned int>(matrix_blocks), threads, 0, stream>>>(
                reinterpret_cast<double *>(y), matrix_size, y_stride_0, y_stride_1);
        }
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t packInputToContiguous(
    void *dst,
    const void *src,
    infiniDtype_t dtype,
    size_t matrix_size,
    ptrdiff_t src_stride_0,
    ptrdiff_t src_stride_1,
    cudaStream_t stream) {
    constexpr int threads = 256;
    size_t matrix_numel = matrix_size * matrix_size;
    size_t blocks = CEIL_DIV(matrix_numel, static_cast<size_t>(threads));
    switch (dtype) {
    case INFINI_DTYPE_F16:
        packMatrix2dStridedToContiguous<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            reinterpret_cast<const __half *>(src), reinterpret_cast<__half *>(dst),
            matrix_size, src_stride_0, src_stride_1);
        break;
    case INFINI_DTYPE_BF16:
        packMatrix2dStridedToContiguous<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            reinterpret_cast<const cuda_bfloat16 *>(src), reinterpret_cast<cuda_bfloat16 *>(dst),
            matrix_size, src_stride_0, src_stride_1);
        break;
    case INFINI_DTYPE_F32:
        packMatrix2dStridedToContiguous<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            reinterpret_cast<const float *>(src), reinterpret_cast<float *>(dst),
            matrix_size, src_stride_0, src_stride_1);
        break;
    case INFINI_DTYPE_F64:
        packMatrix2dStridedToContiguous<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            reinterpret_cast<const double *>(src), reinterpret_cast<double *>(dst),
            matrix_size, src_stride_0, src_stride_1);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t scatterContiguousToOutput(
    void *dst,
    const void *src,
    infiniDtype_t dtype,
    size_t matrix_size,
    ptrdiff_t dst_stride_0,
    ptrdiff_t dst_stride_1,
    cudaStream_t stream) {
    constexpr int threads = 256;
    size_t matrix_numel = matrix_size * matrix_size;
    size_t blocks = CEIL_DIV(matrix_numel, static_cast<size_t>(threads));
    switch (dtype) {
    case INFINI_DTYPE_F16:
        scatterMatrix2dContiguousToStrided<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            reinterpret_cast<const __half *>(src), reinterpret_cast<__half *>(dst),
            matrix_size, dst_stride_0, dst_stride_1);
        break;
    case INFINI_DTYPE_BF16:
        scatterMatrix2dContiguousToStrided<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            reinterpret_cast<const cuda_bfloat16 *>(src), reinterpret_cast<cuda_bfloat16 *>(dst),
            matrix_size, dst_stride_0, dst_stride_1);
        break;
    case INFINI_DTYPE_F32:
        scatterMatrix2dContiguousToStrided<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            reinterpret_cast<const float *>(src), reinterpret_cast<float *>(dst),
            matrix_size, dst_stride_0, dst_stride_1);
        break;
    case INFINI_DTYPE_F64:
        scatterMatrix2dContiguousToStrided<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            reinterpret_cast<const double *>(src), reinterpret_cast<double *>(dst),
            matrix_size, dst_stride_0, dst_stride_1);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
using GemmComputeType = cudaDataType;
#else
using GemmComputeType = cublasComputeType_t;
#endif

struct GemmTypeConfig {
    cudaDataType io_type;
    GemmComputeType compute_type;
};

infiniStatus_t getGemmTypeConfig(infiniDtype_t dtype, GemmTypeConfig &cfg) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        cfg.io_type = CUDA_R_16F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        cfg.compute_type = CUDA_R_32F;
#else
        cfg.compute_type = CUBLAS_COMPUTE_32F;
#endif
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        cfg.io_type = CUDA_R_16BF;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        cfg.compute_type = CUDA_R_32F;
#else
        cfg.compute_type = CUBLAS_COMPUTE_32F;
#endif
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        cfg.io_type = CUDA_R_32F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        cfg.compute_type = CUDA_R_32F;
#else
        cfg.compute_type = CUBLAS_COMPUTE_32F;
#endif
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F64:
        cfg.io_type = CUDA_R_64F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        cfg.compute_type = CUDA_R_64F;
#else
        cfg.compute_type = CUBLAS_COMPUTE_64F;
#endif
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

// Compute row-major C = A * B using cuBLAS column-major GEMM:
// C_col = B_col * A_col, where *_col views the same memory as column-major.
infiniStatus_t gemmRowMajorSquare(
    cublasHandle_t handle,
    const GemmTypeConfig &cfg,
    infiniDtype_t dtype,
    int n,
    const void *a,
    const void *b,
    void *c) {

    if (dtype == INFINI_DTYPE_F64) {
        const double alpha = 1.0;
        const double beta = 0.0;
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            n,
            &alpha,
            b,
            cfg.io_type,
            n,
            a,
            cfg.io_type,
            n,
            &beta,
            c,
            cfg.io_type,
            n,
            cfg.compute_type,
            CUBLAS_GEMM_DEFAULT));
        return INFINI_STATUS_SUCCESS;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        b,
        cfg.io_type,
        n,
        a,
        cfg.io_type,
        n,
        &beta,
        c,
        cfg.io_type,
        n,
        cfg.compute_type,
        CUBLAS_GEMM_DEFAULT));
    return INFINI_STATUS_SUCCESS;
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;

    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_)
        : internal(internal_) {}
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int n) {

    if (handle == nullptr || desc_ptr == nullptr || y_desc == nullptr || x_desc == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (n < 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    if (y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 2 || x_shape[0] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape != x_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (x_desc->hasBroadcastDim() || y_desc->hasBroadcastDim()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    if (x_shape[0] > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto x_strides = x_desc->strides();
    auto y_strides = y_desc->strides();
    bool x_contiguous = x_desc->isContiguous();
    bool y_contiguous = y_desc->isContiguous();

    size_t matrix_numel = x_desc->numel();
    size_t matrix_bytes = matrix_numel * infiniSizeOf(dtype);
    size_t workspace_size = 0;
    if (n != 0) {
        workspace_size = matrix_bytes * (y_contiguous ? 2 : 3);
    }

    auto handle_nvidia = reinterpret_cast<device::nvidia::Handle *>(handle);
    Descriptor *desc = new Descriptor(dtype, x_shape[0], static_cast<size_t>(n),
                                      matrix_numel, y_desc->numel(), workspace_size,
                                      x_strides[0], x_strides[1], y_strides[0], y_strides[1],
                                      x_contiguous, y_contiguous,
                                      handle->device, handle->device_id);
    desc->_opaque = new Opaque(handle_nvidia->internal());
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    if (x == nullptr || y == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (workspace_size < this->workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (this->workspaceSize() != 0 && workspace == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (matrix_size == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    if (n == 0) {
        CHECK_STATUS(initializeIdentity(
            y, _dtype, matrix_size, y_contiguous, y_stride_0, y_stride_1, cuda_stream));
        return INFINI_STATUS_SUCCESS;
    }

    size_t matrix_bytes = input_size * infiniSizeOf(_dtype);
    char *workspace_ptr = reinterpret_cast<char *>(workspace);
    void *base = workspace_ptr;
    void *temp = workspace_ptr + matrix_bytes;
    void *contiguous_output = y_contiguous ? y : (workspace_ptr + matrix_bytes * 2);

    CHECK_STATUS(initializeIdentity(
        contiguous_output, _dtype, matrix_size, true, 0, 0, cuda_stream));

    if (x_contiguous) {
        CHECK_CUDA(cudaMemcpyAsync(base, x, matrix_bytes, cudaMemcpyDeviceToDevice, cuda_stream));
    } else {
        CHECK_STATUS(packInputToContiguous(
            base, x, _dtype, matrix_size, x_stride_0, x_stride_1, cuda_stream));
    }

    GemmTypeConfig cfg;
    CHECK_STATUS(getGemmTypeConfig(_dtype, cfg));

    void *result = contiguous_output;
    void *scratch = temp;
    void *base_matrix = base;
    size_t power = n;
    int matrix_dim = static_cast<int>(matrix_size);

    CHECK_STATUS(_opaque->internal->useCublas(
        cuda_stream,
        [&](cublasHandle_t handle) {
            while (power > 0) {
                if (power & 1) {
                    CHECK_STATUS(gemmRowMajorSquare(handle, cfg, _dtype, matrix_dim, result, base_matrix, scratch));
                    std::swap(result, scratch);
                }
                power >>= 1;
                if (power == 0) {
                    break;
                }
                CHECK_STATUS(gemmRowMajorSquare(handle, cfg, _dtype, matrix_dim, base_matrix, base_matrix, scratch));
                std::swap(base_matrix, scratch);
            }
            return INFINI_STATUS_SUCCESS;
        }));

    if (y_contiguous) {
        if (result != y) {
            CHECK_CUDA(cudaMemcpyAsync(y, result, matrix_bytes, cudaMemcpyDeviceToDevice, cuda_stream));
        }
    } else {
        CHECK_STATUS(scatterContiguousToOutput(
            y, result, _dtype, matrix_size, y_stride_0, y_stride_1, cuda_stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matrix_power::nvidia
