#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/swiglu.h"
#include "../../../ninetoothed/utils.h"

namespace op::swiglu::ninetoothed {
class Descriptor final : public InfiniopDescriptor {

public:
    Descriptor(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) : InfiniopDescriptor{handle->device, handle->device_id},
                                                                  out_shape_{out_desc->shape()},
                                                                  out_strides_{out_desc->strides()},
                                                                  up_shape_{input_desc_vec[0]->shape()},
                                                                  up_strides_{input_desc_vec[0]->strides()},
                                                                  gate_shape_{input_desc_vec[1]->shape()},
                                                                  gate_strides_{input_desc_vec[1]->strides()},
                                                                  dtype_{out_desc->dtype()} {}

    ~Descriptor() = default;

    size_t workspaceSize() const {
        return 0;
    }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) {
        *desc_ptr = new Descriptor(handle, out_desc, input_desc_vec);
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const {
        auto out_nt{::ninetoothed::Tensor(output, out_shape_, out_strides_)};
        auto up_nt{::ninetoothed::Tensor(inputs[0], up_shape_, up_strides_)};
        auto gate_nt{::ninetoothed::Tensor(inputs[1], gate_shape_, gate_strides_)};

        if (launch_swiglu(stream,
                          out_nt,
                          up_nt,
                          gate_nt,
                          out_shape_.size(),
                          dtype_,
                          1024)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }

        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Size> out_shape_;
    std::vector<Stride> out_strides_;

    std::vector<Size> up_shape_;
    std::vector<Stride> up_strides_;

    std::vector<Size> gate_shape_;
    std::vector<Stride> gate_strides_;

    infiniDtype_t dtype_;
};
} // namespace op::swiglu::ninetoothed

#endif // SWIGLU_H
