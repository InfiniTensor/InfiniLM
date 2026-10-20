#ifndef INFINI_OPS_CAMBRICON_SIGMOID_CNNL_H_
#define INFINI_OPS_CAMBRICON_SIGMOID_CNNL_H_

#include <algorithm>
#include <cassert>

#include "base/sigmoid.h"
#include "native/cambricon/cnnl_utils.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kCambricon> : public Sigmoid {
 public:
  Operator(const Tensor input, Tensor out) : Sigmoid{input, out} {
    assert(input_shape_ == out_shape_ &&
           "`CambriconSigmoid` requires matching input and output shapes.");
    assert(input_type_ == out_type_ &&
           "`CambriconSigmoid` requires matching input and output dtypes.");
    assert(input.device() == out.device() &&
           "`CambriconSigmoid` requires input and output on the same device.");
    assert((input_type_ == DataType::kFloat16 ||
            input_type_ == DataType::kBFloat16 ||
            input_type_ == DataType::kFloat32) &&
           "`CambriconSigmoid` supports float16, bfloat16, and float32 only.");
    assert(!out.HasBroadcastDim() &&
           "`CambriconSigmoid` output must not have broadcast dimensions.");
    assert(std::all_of(input_strides_.begin(), input_strides_.end(),
                       [](auto stride) { return stride >= 0; }) &&
           "`CambriconSigmoid` does not support negative input strides.");
    assert(std::all_of(out_strides_.begin(), out_strides_.end(),
                       [](auto stride) { return stride >= 0; }) &&
           "`CambriconSigmoid` does not support negative output strides.");

    if (output_size_ == 0) {
      return;
    }

    cnnl_handle_ = cnnl_utils::CreateHandle();
    input_desc_ = cnnl_utils::MakeTensorDescriptor(input_type_, input_shape_,
                                                   input_strides_);
    out_desc_ =
        cnnl_utils::MakeTensorDescriptor(out_type_, out_shape_, out_strides_);

    INFINI_OPS_CNNL_CHECK(cnnlCreateActivationDescriptor(&activation_desc_));
    const cnnlActivationMode_t mode = CNNL_ACTIVATION_SIGMOID;
    const cnnlComputationPreference_t preference =
        CNNL_COMPUTATION_HIGH_PRECISION;
    const cnnlNanPropagation_t nan_propagation = CNNL_PROPAGATE_NAN;
    INFINI_OPS_CNNL_CHECK(cnnlSetActivationDescAttr(
        activation_desc_, CNNL_ACTIVATION_MODE, &mode, sizeof(mode)));
    INFINI_OPS_CNNL_CHECK(
        cnnlSetActivationDescAttr(activation_desc_, CNNL_ACTIVATION_PREFERENCE,
                                  &preference, sizeof(preference)));
    INFINI_OPS_CNNL_CHECK(
        cnnlSetActivationDescAttr(activation_desc_, CNNL_ACTIVATION_NAN_PROP,
                                  &nan_propagation, sizeof(nan_propagation)));
  }

  ~Operator() {
    if (activation_desc_) {
      (void)cnnlDestroyActivationDescriptor(activation_desc_);
    }
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(
        cnnl_handle_.get(), static_cast<cnrtQueue_t>(stream_ ? stream_ : 0)));
    INFINI_OPS_CNNL_CHECK(cnnlActivationForward(
        cnnl_handle_.get(), activation_desc_, nullptr, input_desc_.get(),
        input.data(), nullptr, out_desc_.get(), out.data()));
  }

 private:
  cnnl_utils::Handle cnnl_handle_{};
  cnnl_utils::TensorDescriptor input_desc_{};
  cnnl_utils::TensorDescriptor out_desc_{};
  cnnlActivationDescriptor_t activation_desc_{nullptr};
};

}  // namespace infini::ops

#endif
