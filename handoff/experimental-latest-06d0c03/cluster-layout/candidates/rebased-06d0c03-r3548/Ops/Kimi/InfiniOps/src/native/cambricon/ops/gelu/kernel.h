#ifndef INFINI_OPS_CAMBRICON_GELU_CNNL_H_
#define INFINI_OPS_CAMBRICON_GELU_CNNL_H_

#include <algorithm>
#include <cassert>

#include "base/gelu.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T>
void GeluUnion(void* workspace, int core_per_cluster, int cluster_count,
               cnrtQueue_t queue, const void* input, void* out,
               const size_t* input_shape, const ptrdiff_t* input_strides,
               const size_t* out_shape, const ptrdiff_t* out_strides,
               size_t output_size, int ndim, bool approximate,
               bool input_contiguous, bool out_contiguous);

template <>
class Operator<Gelu, Device::Type::kCambricon> : public Gelu {
 public:
  Operator(const Tensor input, const std::string approximate, Tensor out)
      : Gelu{input, approximate, out} {
    assert(input_shape_ == out_shape_ &&
           "`CambriconGelu` requires matching input and output shapes.");
    assert(input_type_ == out_type_ &&
           "`CambriconGelu` requires matching input and output dtypes.");
    assert(input.device() == out.device() &&
           "`CambriconGelu` requires input and output on the same device.");
    assert((input_type_ == DataType::kFloat16 ||
            input_type_ == DataType::kBFloat16 ||
            input_type_ == DataType::kFloat32) &&
           "`CambriconGelu` supports float16, bfloat16, and float32 only.");
    assert(!out.HasBroadcastDim() &&
           "`CambriconGelu` output must not have broadcast dimensions.");
    assert(std::all_of(input_strides_.begin(), input_strides_.end(),
                       [](auto stride) { return stride >= 0; }) &&
           "`CambriconGelu` does not support negative input strides.");
    assert(std::all_of(out_strides_.begin(), out_strides_.end(),
                       [](auto stride) { return stride >= 0; }) &&
           "`CambriconGelu` does not support negative output strides.");

    if (output_size_ == 0) {
      return;
    }

    if (approximate_ == "tanh" || input_type_ == DataType::kBFloat16) {
      cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster_,
                                  &cluster_count_);
      const auto workspace_size = workspace_size_in_bytes();
      if (workspace_size != 0) {
        CNRT_CHECK(cnrtMalloc(&default_workspace_, workspace_size));
      }
      return;
    }

    cnnl_handle_ = cnnl_utils::CreateHandle();
    input_desc_ = cnnl_utils::MakeTensorDescriptor(input_type_, input_shape_,
                                                   input_strides_);
    out_desc_ =
        cnnl_utils::MakeTensorDescriptor(out_type_, out_shape_, out_strides_);

    INFINI_OPS_CNNL_CHECK(cnnlCreateActivationDescriptor(&activation_desc_));
    const cnnlActivationMode_t mode = CNNL_ACTIVATION_GELU;
    const cnnlComputationPreference_t preference =
        CNNL_COMPUTATION_HIGH_PRECISION;
    const cnnlNanPropagation_t nan_propagation = CNNL_PROPAGATE_NAN;
    const bool use_approximation = false;
    INFINI_OPS_CNNL_CHECK(cnnlSetActivationDescAttr(
        activation_desc_, CNNL_ACTIVATION_MODE, &mode, sizeof(mode)));
    INFINI_OPS_CNNL_CHECK(
        cnnlSetActivationDescAttr(activation_desc_, CNNL_ACTIVATION_PREFERENCE,
                                  &preference, sizeof(preference)));
    INFINI_OPS_CNNL_CHECK(
        cnnlSetActivationDescAttr(activation_desc_, CNNL_ACTIVATION_NAN_PROP,
                                  &nan_propagation, sizeof(nan_propagation)));
    INFINI_OPS_CNNL_CHECK(cnnlSetActivationDescAttr(
        activation_desc_, CNNL_ACTIVATION_APPROXIMATE, &use_approximation,
        sizeof(use_approximation)));
  }

  ~Operator() {
    if (default_workspace_) {
      (void)cnrtFree(default_workspace_);
    }
    if (activation_desc_) {
      (void)cnnlDestroyActivationDescriptor(activation_desc_);
    }
  }

  void operator()(const Tensor input, const std::string approximate,
                  Tensor out) const override {
    assert(approximate == approximate_ &&
           "`CambriconGelu` attributes changed after descriptor creation.");
    if (output_size_ == 0) {
      return;
    }

    if (approximate_ == "tanh" || input_type_ == DataType::kBFloat16) {
      auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
      auto* workspace = workspace_ ? workspace_ : default_workspace_;
      DispatchFunc<
          Device::Type::kCambricon,
          List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
          {out_type_},
          [&](auto tag) {
            using T = typename decltype(tag)::type;
            GeluUnion<T>(workspace, core_per_cluster_, cluster_count_, queue,
                         input.data(), out.data(), input_shape_.data(),
                         input_strides_.data(), out_shape_.data(),
                         out_strides_.data(), output_size_,
                         static_cast<int>(ndim_), approximate_ == "tanh",
                         is_input_contiguous_, is_out_contiguous_);
          },
          "CambriconGelu::operator() - output dispatch");
      return;
    }

    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(
        cnnl_handle_.get(), static_cast<cnrtQueue_t>(stream_ ? stream_ : 0)));
    INFINI_OPS_CNNL_CHECK(cnnlActivationForward(
        cnnl_handle_.get(), activation_desc_, nullptr, input_desc_.get(),
        input.data(), nullptr, out_desc_.get(), out.data()));
  }

  std::size_t workspace_size_in_bytes() const override {
    return (approximate_ == "tanh" || input_type_ == DataType::kBFloat16)
               ? ndim_ * (2 * sizeof(size_t) + 2 * sizeof(ptrdiff_t))
               : 0;
  }

 private:
  cnnl_utils::Handle cnnl_handle_{};
  cnnl_utils::TensorDescriptor input_desc_{};
  cnnl_utils::TensorDescriptor out_desc_{};
  cnnlActivationDescriptor_t activation_desc_{nullptr};
  void* default_workspace_{nullptr};
  int core_per_cluster_{0};
  int cluster_count_{0};
};

}  // namespace infini::ops

#endif
