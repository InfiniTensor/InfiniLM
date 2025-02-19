#include "matmul_xdnn.h"

template <typename T>
infiniopStatus_t matmulKunlunCommon(infiniopMatmulKunlunDescriptor_t desc,
                                    void *c,
                                    float beta,
                                    void const *a,
                                    void const *b,
                                    float alpha,
                                    void *stream) {
    auto info = desc->info;

    if (info.is_transed) {
        std::swap(a, b);
    }

    auto transA = info.a_matrix.col_stride == 1 ? false : true;
    auto transB = info.b_matrix.col_stride == 1 ? false : true;

    auto ret = use_xdnn(desc->xdnn_handle_pool,
                        (XPUStream)stream,
                        [&](xdnnHandle_t handle) {
                            for (size_t i = 0; i < info.batch; i++) {
                                CHECK_KUNLUN((
                                    xdnn::fc_fusion<T, T, T, int16_t>(
                                        handle,
                                        (T *)((char *)a + i * info.a_matrix.stride * infiniSizeof(desc->dtype)),
                                        (T *)((char *)b + i * info.b_matrix.stride * infiniSizeof(desc->dtype)),
                                        (T *)((char *)c + i * info.c_matrix.stride * infiniSizeof(desc->dtype)),
                                        info.m,
                                        info.n,
                                        info.k,
                                        transA,
                                        transB,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        info.a_matrix.ld(),
                                        info.b_matrix.ld(),
                                        info.c_matrix.ld(),
                                        alpha,
                                        beta,
                                        nullptr,
                                        xdnn::Activation_t::LINEAR,
                                        nullptr)));
                            }
                            return INFINIOP_STATUS_SUCCESS;
                        });
    return ret;
}

infiniopStatus_t kunlunCreateMatmulDescriptor(infiniopKunlunHandle_t handle,
                                              infiniopMatmulKunlunDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t c_desc,
                                              infiniopTensorDescriptor_t a_desc,
                                              infiniopTensorDescriptor_t b_desc) {
    infiniDtype_t dtype = c_desc->dtype;

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status, false);
    if (status != INFINIOP_STATUS_SUCCESS) {
        return status;
    }

    *desc_ptr = new InfiniopMatmulKunlunDescriptor{
        INFINI_DEVICE_KUNLUN,
        dtype,
        handle->device_id,
        info,
        handle->xdnn_handle_pool};
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t kunlunGetMatmulWorkspaceSize(infiniopMatmulKunlunDescriptor_t desc,
                                              size_t *size) {
    *size = 0;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t kunlunMatmul(infiniopMatmulKunlunDescriptor_t desc,
                              void *workspace,
                              size_t workspace_size,
                              void *c,
                              void const *a,
                              void const *b,
                              float alpha,
                              float beta,
                              void *stream) {
    if (desc->dtype == INFINI_DTYPE_F16) {
        return matmulKunlunCommon<float16>(desc, c, beta, a, b, alpha, stream);
    }
    if (desc->dtype == INFINI_DTYPE_F32) {
        return matmulKunlunCommon<float>(desc, c, beta, a, b, alpha, stream);
    }
    return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t kunlunDestroyMatmulDescriptor(infiniopMatmulKunlunDescriptor_t desc) {
    desc->xdnn_handle_pool = nullptr;
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
