#ifndef __GPTQ_QYBLAS_GEMM_INFO_H__
#define __GPTQ_QYBLAS_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <optional>
#include <vector>

inline void prepare_matrix_for_cublas(
    infiniopTensorDescriptor_t tensor,
    bool &transpose_tensor) {

    auto strides = tensor->strides();
    auto sizes = tensor->shape();

    if ((strides[0] == 1) && (strides[1] >= std::max<int64_t>(1, sizes[0]))) {

        transpose_tensor = false;
        return;
    }
    if ((strides[1] == 1) && (strides[0] >= std::max<int64_t>(1, sizes[1]))) {

        transpose_tensor = true;
        return;
    }
    transpose_tensor = true;
}

namespace op::gptq_qyblas_gemm {

class GptqQyblasGemmInfo {
    GptqQyblasGemmInfo() = default;

public:
    infiniDtype_t dtype, weight_dtype, scales_dtype, zeros_dtype, out_dtype;
    size_t M, K, N, scales_size_0, scales_size_1;
    ptrdiff_t lda, ldb, result_ld;
    bool transpose_result;
    char transa, transb;

    static utils::Result<GptqQyblasGemmInfo> createGptqQyblasGemmInfo(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t b_scales_desc,
        infiniopTensorDescriptor_t b_zeros_desc) {

        auto dtype = a_desc->dtype();

        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
        auto out_dtype = out_desc->dtype();
        CHECK_DTYPE(dtype, out_dtype);

        const infiniDtype_t weight_dtype = b_desc->dtype();
        // CHECK_DTYPE(weight_dtype, INFINI_DTYPE_F8, INFINI_DTYPE_U8, INFINI_DTYPE_I8);

        const infiniDtype_t scales_dtype = b_scales_desc->dtype();
        const infiniDtype_t zeros_dtype = b_zeros_desc->dtype();

        bool transpose_result = false;
        bool transpose_mat_1 = false;
        bool transpose_mat_2 = false;

        prepare_matrix_for_cublas(out_desc, transpose_result);

        auto mata = (transpose_result ? b_desc : a_desc);
        prepare_matrix_for_cublas(transpose_result ? b_desc : a_desc, transpose_mat_1);
        auto matb = (transpose_result ? a_desc : b_desc);
        prepare_matrix_for_cublas(transpose_result ? a_desc : b_desc, transpose_mat_2);

        auto mat1_sizes = a_desc->shape();
        auto mat2_sizes = b_desc->shape();
        if (transpose_result) {
            transpose_mat_1 = !transpose_mat_1;
            transpose_mat_2 = !transpose_mat_2;
            mat1_sizes = mata->shape();
            mat2_sizes = matb->shape();
        }

        size_t M = mat1_sizes[transpose_result ? 1 : 0];
        size_t K = mat1_sizes[transpose_result ? 0 : 1];
        size_t N = mat2_sizes[transpose_result ? 0 : 1];

        size_t scales_size_0 = b_scales_desc->shape()[0];
        size_t scales_size_1 = b_scales_desc->shape()[1];

        auto ndim = out_desc->ndim();
        CHECK_OR_RETURN(ndim == 2
                            && a_desc->ndim() == ndim
                            && b_desc->ndim() == ndim
                            && b_scales_desc->ndim() == ndim
                            && b_zeros_desc->ndim() == ndim,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        ptrdiff_t lda = mata->strides()[(transpose_mat_1 == transpose_result)
                                            ? 1
                                            : 0];
        ptrdiff_t ldb = matb->strides()[(transpose_mat_2 == transpose_result)
                                            ? 1
                                            : 0];
        ptrdiff_t result_ld = out_desc->strides()[transpose_result ? 0 : 1];

        char transa = transpose_mat_1 ? 't' : 'n';
        char transb = transpose_mat_2 ? 't' : 'n';

        return utils::Result<GptqQyblasGemmInfo>(GptqQyblasGemmInfo{
            dtype, weight_dtype, scales_dtype, zeros_dtype, out_dtype,
            M, K, N, scales_size_0, scales_size_1,
            lda, ldb, result_ld,
            transpose_result,
            transa, transb});
    }
};

} // namespace op::gptq_qyblas_gemm

#endif // __GPTQ_QYBLAS_GEMM_INFO_H__
