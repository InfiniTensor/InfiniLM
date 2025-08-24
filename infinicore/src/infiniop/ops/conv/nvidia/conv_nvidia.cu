#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "conv_nvidia.cuh"

#define DESTROY_CUDNN_DESCRIPTOR(desc_ptr, destroy_func) \
    do {                                                 \
        if (desc_ptr) {                                  \
            destroy_func(desc_ptr);                      \
            desc_ptr = nullptr;                          \
        }                                                \
    } while (0)

#define CLEANUP_CUDNN_DESCRIPTORS()                                             \
    do {                                                                        \
        DESTROY_CUDNN_DESCRIPTOR(x_desc, cudnnDestroyTensorDescriptor);         \
        DESTROY_CUDNN_DESCRIPTOR(y_desc, cudnnDestroyTensorDescriptor);         \
        DESTROY_CUDNN_DESCRIPTOR(w_desc, cudnnDestroyFilterDescriptor);         \
        DESTROY_CUDNN_DESCRIPTOR(b_desc, cudnnDestroyTensorDescriptor);         \
        DESTROY_CUDNN_DESCRIPTOR(act_desc, cudnnDestroyActivationDescriptor);   \
        DESTROY_CUDNN_DESCRIPTOR(conv_desc, cudnnDestroyConvolutionDescriptor); \
    } while (0)

namespace op::conv::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    size_t workspace_size = 0;

#ifdef ENABLE_CUDNN_API
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    cudnnActivationDescriptor_t act_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#endif

private:
    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_ptr)
        : internal(internal_ptr) {}

#ifdef ENABLE_CUDNN_API
    void initializeDimensionArrays(const ConvInfo &info,
                                   std::vector<int> &input_dims,
                                   std::vector<int> &output_dims,
                                   std::vector<int> &filter_dims,
                                   std::vector<int> &input_strides,
                                   std::vector<int> &output_strides) const {
        bool is_1d_conv = (info.ndim() == 1);
        int actual_tensor_ndim = is_1d_conv ? 4 : static_cast<int>(info.ndim() + 2);

        input_dims[0] = static_cast<int>(info.batch());
        input_dims[1] = static_cast<int>(info.in_channels());
        output_dims[0] = static_cast<int>(info.batch());
        output_dims[1] = static_cast<int>(info.out_channels());
        filter_dims[0] = static_cast<int>(info.out_channels());
        filter_dims[1] = static_cast<int>(info.in_channels());

        if (is_1d_conv) {
            input_dims[2] = 1;
            input_dims[3] = static_cast<int>(info.input_dim(0));
            output_dims[2] = 1;
            output_dims[3] = static_cast<int>(info.output_dim(0));
            filter_dims[2] = 1;
            filter_dims[3] = static_cast<int>(info.kernel_dim(0));
        } else {
            for (size_t i = 0; i < info.ndim(); ++i) {
                input_dims[i + 2] = static_cast<int>(info.input_dim(i));
                output_dims[i + 2] = static_cast<int>(info.output_dim(i));
                filter_dims[i + 2] = static_cast<int>(info.kernel_dim(i));
            }
        }
        calculateStrides(input_dims, input_strides, actual_tensor_ndim);
        calculateStrides(output_dims, output_strides, actual_tensor_ndim);
    }

    void initializeConvolutionParams(const ConvInfo &info,
                                     std::vector<int> &pads,
                                     std::vector<int> &strides,
                                     std::vector<int> &dilations) const {
        bool is_1d_conv = (info.ndim() == 1);

        if (is_1d_conv) {
            pads[0] = 0;
            pads[1] = static_cast<int>(info.pad_info(0));
            strides[0] = 1;
            strides[1] = static_cast<int>(info.stride_info(0));
            dilations[0] = 1;
            dilations[1] = static_cast<int>(info.dilation_info(0));
        } else {
            for (size_t i = 0; i < info.ndim(); ++i) {
                pads[i] = static_cast<int>(info.pad_info(i));
                strides[i] = static_cast<int>(info.stride_info(i));
                dilations[i] = static_cast<int>(info.dilation_info(i));
            }
        }
    }

    void calculateStrides(const std::vector<int> &dims,
                          std::vector<int> &strides,
                          int ndim) const {
        strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * dims[d + 1];
        }
    }

    infiniStatus_t getCudnnDataType(infiniDtype_t data_type,
                                    cudnnDataType_t &cudnn_data_type) const {
        if (data_type == INFINI_DTYPE_F16) {
            cudnn_data_type = device::nvidia::getCudnnDtype(data_type);
        } else if (data_type == INFINI_DTYPE_F32) {
            cudnn_data_type = device::nvidia::getCudnnDtype(data_type);
        } else if (data_type == INFINI_DTYPE_BF16) {
            cudnn_data_type = device::nvidia::getCudnnDtype(data_type);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t createBasicDescriptors(const std::vector<int> &input_dims,
                                          const std::vector<int> &output_dims,
                                          const std::vector<int> &filter_dims,
                                          cudnnDataType_t cudnn_data_type,
                                          int actual_tensor_ndim) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

        CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(
            x_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
            actual_tensor_ndim, input_dims.data()));
        CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(
            y_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
            actual_tensor_ndim, output_dims.data()));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(
            w_desc, cudnn_data_type, CUDNN_TENSOR_NCHW,
            actual_tensor_ndim, filter_dims.data()));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t createBiasDescriptors(const ConvInfo &info,
                                         cudnnDataType_t cudnn_data_type,
                                         int actual_tensor_ndim) {
        if (info.bias_dims_size() == 0) {
            b_desc = nullptr;
            act_desc = nullptr;
            return INFINI_STATUS_SUCCESS;
        }

        std::vector<int> bias_dims_arr(actual_tensor_ndim);
        std::vector<int> bias_strides_arr(actual_tensor_ndim);

        bias_dims_arr[0] = 1;
        bias_dims_arr[1] = static_cast<int>(info.out_channels());
        for (int i = 2; i < actual_tensor_ndim; ++i) {
            bias_dims_arr[i] = 1;
        }

        if (actual_tensor_ndim == 4) {
            bias_strides_arr[0] = static_cast<int>(info.out_channels());
            bias_strides_arr[1] = 1;
            bias_strides_arr[2] = 1;
            bias_strides_arr[3] = 1;
        } else {
            calculateStrides(bias_dims_arr, bias_strides_arr, actual_tensor_ndim);
        }

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&b_desc));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(
            b_desc, cudnn_data_type, static_cast<int>(bias_dims_arr.size()),
            bias_dims_arr.data(), bias_strides_arr.data()));
        CHECK_CUDNN(cudnnCreateActivationDescriptor(&act_desc));
        CHECK_CUDNN(cudnnSetActivationDescriptor(
            act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t setupConvolutionDescriptor(const std::vector<int> &pads,
                                              const std::vector<int> &strides,
                                              const std::vector<int> &dilations,
                                              int spatial_ndim,
                                              cudnnDataType_t compute_type) {
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
            conv_desc,
            spatial_ndim,
            pads.data(),
            strides.data(),
            dilations.data(),
            CUDNN_CROSS_CORRELATION,
            compute_type));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t setupAlgorithmWithoutBias() {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        CHECK_STATUS(internal->useCudnn(
            nullptr,
            [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                    handle, x_desc, w_desc, conv_desc, y_desc,
                    algo, &workspace_size));
                return INFINI_STATUS_SUCCESS;
            }));
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t setupAlgorithmWithBias() {
        int maxAlgoCount = 0;
        CHECK_STATUS(internal->useCudnn(
            nullptr,
            [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maxAlgoCount));
                return INFINI_STATUS_SUCCESS;
            }));

        if (maxAlgoCount <= 0) {
            maxAlgoCount = 8;
        }

        std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(maxAlgoCount);
        int algoCounts = 0;

        CHECK_STATUS(internal->useCudnn(
            nullptr, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
                    handle, x_desc, w_desc, conv_desc, y_desc,
                    maxAlgoCount, &algoCounts, perf_results.data()));
                return INFINI_STATUS_SUCCESS;
            }));

        if (algoCounts < 1) {
            return INFINI_STATUS_BAD_PARAM;
        }

        for (int i = 0; i < algoCounts; ++i) {
            CHECK_STATUS(internal->useCudnn(
                nullptr,
                [&](cudnnHandle_t handle) {
                    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                        handle, x_desc, w_desc, conv_desc, y_desc,
                        perf_results[i].algo, &workspace_size));
                    return INFINI_STATUS_SUCCESS;
                }));
            algo = perf_results[i].algo;
            break;
        }

        return INFINI_STATUS_SUCCESS;
    }
#endif

public:
    Opaque(Opaque &&other) noexcept
        : internal(std::move(other.internal)),
          workspace_size(other.workspace_size)
    // clang-format off
#ifdef ENABLE_CUDNN_API
          , x_desc(other.x_desc)
          , y_desc(other.y_desc)
          , w_desc(other.w_desc)
          , b_desc(other.b_desc)
          , act_desc(other.act_desc)
          , conv_desc(other.conv_desc)
          , algo(other.algo)
#endif
    // clang-format on
    {
#ifdef ENABLE_CUDNN_API
        other.x_desc = nullptr;
        other.y_desc = nullptr;
        other.w_desc = nullptr;
        other.b_desc = nullptr;
        other.act_desc = nullptr;
        other.conv_desc = nullptr;
#endif
        other.workspace_size = 0;
    }

    ~Opaque() {
#ifdef ENABLE_CUDNN_API
        CLEANUP_CUDNN_DESCRIPTORS();
#endif
    }

#ifdef ENABLE_CUDNN_API
    infiniStatus_t initializeCudnnContext(ConvInfo &info,
                                          infiniDtype_t data_type,
                                          cudnnDataType_t compute_type) {
        bool is_1d_conv = (info.ndim() == 1);
        int actual_tensor_ndim = is_1d_conv ? 4 : static_cast<int>(info.ndim() + 2);
        int spatial_ndim_for_conv_desc = is_1d_conv ? 2 : static_cast<int>(info.ndim());

        std::vector<int> input_dims_arr(actual_tensor_ndim);
        std::vector<int> output_dims_arr(actual_tensor_ndim);
        std::vector<int> filter_dims_arr(actual_tensor_ndim);
        std::vector<int> input_strides_arr(actual_tensor_ndim);
        std::vector<int> output_strides_arr(actual_tensor_ndim);

        initializeDimensionArrays(info, input_dims_arr, output_dims_arr,
                                  filter_dims_arr, input_strides_arr, output_strides_arr);

        std::vector<int> pads_arr(spatial_ndim_for_conv_desc);
        std::vector<int> strides_arr(spatial_ndim_for_conv_desc);
        std::vector<int> dilations_arr(spatial_ndim_for_conv_desc);

        initializeConvolutionParams(info, pads_arr, strides_arr, dilations_arr);

        cudnnDataType_t cudnn_data_type;
        CHECK_STATUS(getCudnnDataType(data_type, cudnn_data_type));

        CHECK_STATUS(createBasicDescriptors(input_dims_arr, output_dims_arr,
                                            filter_dims_arr, cudnn_data_type, actual_tensor_ndim));

        CHECK_STATUS(createBiasDescriptors(info, cudnn_data_type, actual_tensor_ndim));

        CHECK_STATUS(setupConvolutionDescriptor(pads_arr, strides_arr, dilations_arr,
                                                spatial_ndim_for_conv_desc, compute_type));

        if (info.bias_dims_size() == 0) {
            CHECK_STATUS(setupAlgorithmWithoutBias());
        } else {
            CHECK_STATUS(setupAlgorithmWithBias());
        }

        return INFINI_STATUS_SUCCESS;
    }
#endif

    static inline utils::Result<Opaque> create(
        std::shared_ptr<device::nvidia::Handle::Internal> internal_ptr,
        ConvInfo &info,
        infiniDtype_t data_type) {
#ifdef ENABLE_CUDNN_API
        Opaque opaque(internal_ptr);
        auto status = opaque.initializeCudnnContext(info, data_type, CUDNN_DATA_FLOAT);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
        return utils::Result<Opaque>(std::move(opaque));
#else
        return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
    }
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n) {
#ifdef ENABLE_CUDNN_API
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc,
                                   pads, strides, dilations, n);

    CHECK_RESULT(result);
    auto conv_info = result.take();
    auto opaque_result = Opaque::create(handle->internal(), conv_info, dtype);
    CHECK_RESULT(opaque_result);
    auto opaque = new Opaque(opaque_result.take());

    *desc_ptr = new Descriptor(
        dtype,
        std::move(conv_info),
        opaque->workspace_size,
        opaque,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *bias,
    void *stream) const {
#ifdef ENABLE_CUDNN_API
    const float alpha = 1.0f, beta = 0.0f;
    if (bias != nullptr) {
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
                    handle,
                    &alpha,
                    _opaque->x_desc,
                    x,
                    _opaque->w_desc,
                    w,
                    _opaque->conv_desc,
                    _opaque->algo,
                    workspace, workspace_size,
                    &beta,
                    _opaque->y_desc,
                    y,
                    _opaque->b_desc,
                    bias,
                    _opaque->act_desc,
                    _opaque->y_desc,
                    y));
                return INFINI_STATUS_SUCCESS;
            }));
    } else {
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnConvolutionForward(
                    handle,
                    &alpha,
                    _opaque->x_desc,
                    x,
                    _opaque->w_desc,
                    w,
                    _opaque->conv_desc,
                    _opaque->algo,
                    workspace, workspace_size,
                    &beta,
                    _opaque->y_desc,
                    y));
                return INFINI_STATUS_SUCCESS;
            }));
    }

    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}
} // namespace op::conv::nvidia
