
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#ifdef ENABLE_CUTLASS_API
#include "int8_gemm_kernel.cuh"
#endif
#include "../cuda/per_channel_dequant_int8.cuh"
#include "int8_gemm_nvidia.cuh"

template <typename Tdata>
INFINIOP_CUDA_KERNEL postSym(
    Tdata *y, int32_t *y_packed, const Tdata *bias, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N) {
    postSymKernel<Tdata>(y, y_packed, bias, x_packed, x_scale, w_packed, w_scale, M, K, N);
}
template <typename Tdata>
INFINIOP_CUDA_KERNEL postSym(
    Tdata *y, int32_t *y_packed, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N) {
    postSymKernel<Tdata>(y, y_packed, x_packed, x_scale, w_packed, w_scale, M, K, N);
}

namespace op::i8gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

#ifdef ENABLE_NVIDIA_API
inline int getSMVersion() {
    int device{-1};
    CHECK_CUDA(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}
#endif

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t a_scale_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scale_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    auto result = I8GemmInfo::create(out_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);
    size_t workspace_size = out_desc->dim(0) * out_desc->dim(1) * sizeof(int32_t);
    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        result.take(), workspace_size, dtype,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

#ifdef ENABLE_QY_API
template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t Descriptor::launchKernel(const I8GemmInfo &info, Tdata *y, const Tdata *bias, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, void *stream_, void *workspace) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    int M = (int)info.m;
    int K = (int)info.k;
    int N = (int)info.n;

    char *workspace_ptr = reinterpret_cast<char *>(workspace);
    int32_t *y_packed = reinterpret_cast<int32_t *>(workspace_ptr);
    const int32_t alpha_I = 1;
    const int32_t beta_I = 0;
    int lda = K; // w_packed is column-major [K, N]
    int ldb = K; // x_packed is row-major [M, K]
    int ldc = N; // y_packed is row-major [M, N]
    CHECK_STATUS(this->_opaque->internal->useCublas(
        stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_T, // A = w_packed^T : [N, K]
                CUBLAS_OP_N, // B = x_packed^T viewed column-major : [K, M]
                N,           // m
                M,           // n
                K,           // k
                &alpha_I,
                w_packed, CUDA_R_8I, lda,
                x_packed, CUDA_R_8I, ldb,
                &beta_I,
                y_packed, CUDA_R_32I, ldc,
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT));
            return INFINI_STATUS_SUCCESS;
        }));
    constexpr unsigned int BLOCK_SIZE_x = 32;
    constexpr unsigned int BLOCK_SIZE_y = 32;

    int num_block_x = (N + BLOCK_SIZE_x - 1) / BLOCK_SIZE_x;
    int num_block_y = (M + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
    dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    if (bias == nullptr) {
        postSym<Tdata><<<grid_dim, block_dim, 0, stream>>>(y, y_packed, x_packed, x_scale, w_packed, w_scale, M, K, N);
    } else {
        postSym<Tdata><<<grid_dim, block_dim, 0, stream>>>(y, y_packed, bias, x_packed, x_scale, w_packed, w_scale, M, K, N);
    }

    return INFINI_STATUS_SUCCESS;
}
#endif

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *bias,
    const void *a,
    const void *a_scale,
    const void *b,
    const void *b_scale,
    void *stream) const {
#if defined(ENABLE_NVIDIA_API) && defined(ENABLE_CUTLASS_API)
    auto sm_version = getSMVersion();
    if (sm_version >= 75 && sm_version < 80) {
        CHECK_DTYPE(this->_out_dtype, INFINI_DTYPE_F16);
        sm75_dispatch_shape<cutlass::half_t, cutlass::arch::Sm75, cutlass::gemm::GemmShape<8, 8, 16>>(
            out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
    } else if (sm_version >= 80 && sm_version < 90) {
        // sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
        if (sm_version == 86 || sm_version == 89) {
            if (this->_out_dtype == INFINI_DTYPE_BF16) {
                sm89_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                    out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
            } else {
                sm89_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                    out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
            }
        } else {
            if (this->_out_dtype == INFINI_DTYPE_BF16) {
                sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                    out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
            } else {
                sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                    out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
            }
        }
    } else if (sm_version == 90) {
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
        // cutlass 3.x
        if (this->_out_dtype == INFINI_DTYPE_BF16) {
            sm90_dispatch_shape<cutlass::bfloat16_t>(
                out, a, b, a_scale, b_scale, bias,
                _info.m, _info.n, _info.k,
                _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(),
                stream);
        } else {
            sm90_dispatch_shape<cutlass::half_t>(
                out, a, b, a_scale, b_scale, bias,
                _info.m, _info.n, _info.k,
                _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(),
                stream);
        }
#else
        // // fallback to cutlass 2.x
        if (this->_out_dtype == INFINI_DTYPE_BF16) {
            sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
        } else {
            sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
        }
#endif
    } else {
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }
#elif defined ENABLE_QY_API
#define CALCULATE_LINEAR(BLOCK_SIZE, TDATA) \
    launchKernel<BLOCK_SIZE, TDATA>(_info, (TDATA *)out, (const TDATA *)bias, (const int8_t *)a, (const float *)a_scale, (const int8_t *)b, (const float *)b_scale, stream, workspace)
#define CALCULATE_LINEAR_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                           \
        if (this->_out_dtype == INFINI_DTYPE_F16)               \
            return CALCULATE_LINEAR(BLOCK_SIZE, half);          \
        else if (this->_out_dtype == INFINI_DTYPE_F32)          \
            return CALCULATE_LINEAR(BLOCK_SIZE, float);         \
        else if (this->_out_dtype == INFINI_DTYPE_BF16)         \
            return CALCULATE_LINEAR(BLOCK_SIZE, __nv_bfloat16); \
        else                                                    \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;              \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
#endif
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::i8gemm::nvidia
