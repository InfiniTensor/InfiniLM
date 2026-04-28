#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_handle.h"
#include "int8_gemm_moore.h"

namespace op::i8gemm::moore {

static void moore_i8gemm_launch(
    const I8GemmInfo &info,
    std::shared_ptr<device::moore::Handle::Internal> &internal,
    void* out,
    const int8_t* A,
    const int8_t* B,
    const float* A_scale,
    const float* B_scale,
    const void* bias,
    infiniDtype_t out_dtype,
    musaStream_t stream)
{
    internal->useMudnn(stream,
        [&](::musa::dnn::Handle &mudnn_handle) -> infiniStatus_t {

        // 1. Operator
        auto matmul = std::make_unique<::musa::dnn::BatchMatMul>();
        matmul->SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR);

        // 2. Tensors
        ::musa::dnn::Tensor out_t, a_t, b_t, bias_t;
        ::musa::dnn::Tensor scale_a_t, scale_b_t;

        // 3. Output dtype
        if (out_dtype == INFINI_DTYPE_F16) {
            out_t.SetType(::musa::dnn::Tensor::Type::HALF);
            bias_t.SetType(::musa::dnn::Tensor::Type::HALF);
        } else {
            out_t.SetType(::musa::dnn::Tensor::Type::BFLOAT16);
            bias_t.SetType(::musa::dnn::Tensor::Type::BFLOAT16);
        }

        // 4. Input INT8
        a_t.SetType(::musa::dnn::Tensor::Type::INT8);
        b_t.SetType(::musa::dnn::Tensor::Type::INT8);

        // 5. Scale (per-tensor)
        scale_a_t.SetType(::musa::dnn::Tensor::Type::FLOAT);
        scale_b_t.SetType(::musa::dnn::Tensor::Type::FLOAT);

        // 6. Bind memory
        out_t.SetAddr(out);
        a_t.SetAddr(const_cast<int8_t*>(A));
        b_t.SetAddr(const_cast<int8_t*>(B));
        scale_a_t.SetAddr(const_cast<float*>(A_scale));
        scale_b_t.SetAddr(const_cast<float*>(B_scale));

        if (bias)
            bias_t.SetAddr(const_cast<void*>(bias));

        // 7. A NdInfo
        {
            std::array<int64_t,3> dims;
            std::array<int64_t,3> strides;

            if (info.a_matrix.col_stride != 1) {
                dims = {info.batch, info.k, info.m};
            } else {
                dims = {info.batch, info.m, info.k};
            }
            strides = {
                info.a_matrix.stride,
                info.a_matrix.ld(),
                1
            };
            a_t.SetNdInfo(3, dims.data(), strides.data());
        }

        // 8. B NdInfo
        {
            std::array<int64_t,3> dims;
            std::array<int64_t,3> strides;

            if (info.b_matrix.col_stride != 1) {
                dims = {info.batch, info.n, info.k};
            } else {
                dims = {info.batch, info.k, info.n};
            }
            strides = {
                info.b_matrix.stride,
                info.b_matrix.ld(),
                1
            };
            b_t.SetNdInfo(3, dims.data(), strides.data());
        }

        // 9. out NdInfo
        {
            std::array<int64_t, 3> dims = {
                info.batch,
                info.m,
                info.n
            };

            std::array<int64_t, 3> strides = {
                info.m * info.n,
                info.n,
                1
            };

            out_t.SetNdInfo(3, dims.data(), strides.data());
        }


        // 10. Bias & scale NdInfo
        if (bias) {
            std::array<int64_t,1> dims = { info.n };
            std::array<int64_t,1> strides = { 1 };
            bias_t.SetNdInfo(1, dims.data(), strides.data());
        }

        {
            std::array<int64_t,3> a_scale_dims  = { info.batch, info.m, 1 };
            std::array<int64_t,3> a_scale_strides = { info.m, 1, 1 };
            scale_a_t.SetNdInfo(3, a_scale_dims.data(), a_scale_strides.data());
            
            std::array<int64_t,3> b_scale_dims  = { info.batch, 1, info.n };
            std::array<int64_t,3> b_scale_strides = { info.n, 1, 1 };
            scale_b_t.SetNdInfo(3, b_scale_dims.data(), b_scale_strides.data());
            
        }

        // 11. Transpose
        matmul->SetTranspose(
            info.a_matrix.col_stride != 1,
            info.b_matrix.col_stride != 1);

        // 12. Lt param (no epilogue enum)
        ::musa::dnn::MatMulLtParam lt_param;
        lt_param.SetScale(
            scale_a_t,
            scale_b_t,
            ::musa::dnn::Tensor(),
            ::musa::dnn::Tensor());

        // 13. Alpha / Beta
        matmul->SetAlpha(1.0);
        matmul->SetBeta(0.0);
        matmul->SetGamma(1.0);

        // 14. Workspace
        ::musa::dnn::MemoryMaintainer maintainer =
            [](size_t size) {
                void* ptr = nullptr;
                musaMalloc(&ptr, size);
                return ::musa::dnn::MemoryHandler(
                    ptr,
                    [](void* p) { if (p) musaFree(p); });
            };

        // 15. Run
        matmul->RunLt(
            mudnn_handle,
            out_t,
            a_t,
            b_t,
            ::musa::dnn::Tensor(),
            bias ? bias_t : ::musa::dnn::Tensor(),
            lt_param,
            maintainer);

        return INFINI_STATUS_SUCCESS;
    });
}

/* ================= Descriptor ================= */

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t a_scale_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scale_desc)
{
    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    auto result = I8GemmInfo::create(
        out_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        result.take(),
        0,
        dtype,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *bias,
    const void *a,
    const void *a_scale,
    const void *b,
    const void *b_scale,
    void *stream_) const
{
    moore_i8gemm_launch(
        _info,
        _opaque->internal,
        out,
        static_cast<const int8_t*>(a),
        static_cast<const int8_t*>(b),
        static_cast<const float*>(a_scale),
        static_cast<const float*>(b_scale),
        bias,
        _out_dtype,
        reinterpret_cast<musaStream_t>(stream_));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::i8gemm::moore
