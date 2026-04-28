#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "gemm_qy.cuh"

namespace op::gemm::qy {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    cudaDataType a_type, b_type, c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type;
#else
    cublasComputeType_t compute_type;
#endif

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = CUDA_R_16BF;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    CHECK_STATUS(_opaque->internal->useCublas(
        (cudaStream_t)stream,
        [&](cublasHandle_t handle) {
            // 1. 获取batch数和各矩阵的元素大小（字节）
            const int batch = static_cast<int>(_info.batch);
            const size_t a_elem_size = 2; // getCudaDataTypeSize(a_type);
            const size_t b_elem_size = 2; // getCudaDataTypeSize(b_type);
            const size_t c_elem_size = 2; // getCudaDataTypeSize(c_type);

            // 2. 循环处理每个batch
            for (int i = 0; i < batch; ++i) {
                // 计算当前batch的A/B/C指针（stride是元素步长，转字节步长）
                const void *a_batch = (const char *)a + i * _info.a_matrix.stride * a_elem_size;
                const void *b_batch = (const char *)b + i * _info.b_matrix.stride * b_elem_size;
                void *c_batch = (char *)c + i * _info.c_matrix.stride * c_elem_size;

                // 3. 调用单batch的cublasGemmEx（参数与原接口完全对齐）
                CHECK_CUBLAS(
                    cublasGemmEx(
                        handle,
                        op_a,                                  // 原op_a
                        op_b,                                  // 原op_b
                        static_cast<int>(_info.m),             // 原m
                        static_cast<int>(_info.n),             // 原n
                        static_cast<int>(_info.k),             // 原k
                        &alpha,                                // 原alpha
                        a_batch,                               // 当前batch的A指针
                        a_type,                                // 原a_type
                        static_cast<int>(_info.a_matrix.ld()), // 原a的ld
                        b_batch,                               // 当前batch的B指针
                        b_type,                                // 原b_type
                        static_cast<int>(_info.b_matrix.ld()), // 原b的ld
                        &beta,                                 // 原beta
                        c_batch,                               // 当前batch的C指针
                        c_type,                                // 原c_type
                        static_cast<int>(_info.c_matrix.ld()), // 原c的ld
                        compute_type,                          // 原compute_type
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP          // 原算法选择
                        ));
            }
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::qy
