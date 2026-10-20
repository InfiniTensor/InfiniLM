#ifndef INFINI_OPS_CAMBRICON_SOFTMAX_CNNL_H_
#define INFINI_OPS_CAMBRICON_SOFTMAX_CNNL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "base/softmax.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kCambricon> : public Softmax {
 public:
  Operator(const Tensor input, const int64_t dim,
           const std::optional<DataType> dtype, Tensor out)
      : Softmax{input, dim, dtype, out} {
    assert(input_shape_ == out_shape_ &&
           "`CambriconSoftmax` requires matching input and output shapes.");
    assert(ndim_ != 0 && dim_ >= 0 && dim_ < static_cast<int64_t>(ndim_) &&
           "`CambriconSoftmax` dim is out of range.");
    assert(!dtype_.has_value() || dtype_.value() == out_type_);
    assert(input_type_ == out_type_ &&
           "`CambriconSoftmax` requires matching input and output dtypes.");
    assert(input.device() == out.device() &&
           "`CambriconSoftmax` requires input and output on the same device.");
    assert((input_type_ == DataType::kFloat16 ||
            input_type_ == DataType::kBFloat16 ||
            input_type_ == DataType::kFloat32) &&
           "`CambriconSoftmax` supports float16, bfloat16, and float32 only.");
    assert(!out.HasBroadcastDim() &&
           "`CambriconSoftmax` output must not have broadcast dimensions.");
    assert(std::all_of(input_strides_.begin(), input_strides_.end(),
                       [](auto stride) { return stride >= 0; }) &&
           "`CambriconSoftmax` does not support negative input strides.");
    assert(std::all_of(out_strides_.begin(), out_strides_.end(),
                       [](auto stride) { return stride >= 0; }) &&
           "`CambriconSoftmax` does not support negative output strides.");

    if (row_count_ == 0 || dim_size_ == 0) {
      return;
    }

    const auto softmax_dim = static_cast<std::size_t>(dim_);
    assert(CanCollapse(input_shape_, input_strides_, 0, softmax_dim) &&
           CanCollapse(input_shape_, input_strides_, softmax_dim + 1, ndim_) &&
           "`CambriconSoftmax` input dimensions around `dim` must be "
           "collapsible.");
    assert(CanCollapse(out_shape_, out_strides_, 0, softmax_dim) &&
           CanCollapse(out_shape_, out_strides_, softmax_dim + 1, ndim_) &&
           "`CambriconSoftmax` output dimensions around `dim` must be "
           "collapsible.");

    const Tensor::Shape cnnl_shape{
        Product(input_shape_, 0, softmax_dim), input_shape_[softmax_dim],
        Product(input_shape_, softmax_dim + 1, ndim_)};
    const Tensor::Strides cnnl_input_strides{
        softmax_dim == 0
            ? input_strides_[0] * static_cast<Tensor::Stride>(input_shape_[0])
            : input_strides_[softmax_dim - 1],
        input_strides_[softmax_dim],
        softmax_dim + 1 == ndim_ ? 1 : input_strides_.back()};
    const Tensor::Strides cnnl_out_strides{
        softmax_dim == 0
            ? out_strides_[0] * static_cast<Tensor::Stride>(out_shape_[0])
            : out_strides_[softmax_dim - 1],
        out_strides_[softmax_dim],
        softmax_dim + 1 == ndim_ ? 1 : out_strides_.back()};

    cnnl_handle_ = cnnl_utils::CreateHandle();
    input_desc_ = cnnl_utils::MakeTensorDescriptor(input_type_, cnnl_shape,
                                                   cnnl_input_strides);
    out_desc_ = cnnl_utils::MakeTensorDescriptor(out_type_, cnnl_shape,
                                                 cnnl_out_strides);

    needs_staging_ = !input.IsContiguous() || !out.IsContiguous();
    if (needs_staging_) {
      const Tensor::Strides contiguous_strides{
          static_cast<Tensor::Stride>(cnnl_shape[1] * cnnl_shape[2]),
          static_cast<Tensor::Stride>(cnnl_shape[2]), 1};
      contiguous_desc_ = cnnl_utils::MakeTensorDescriptor(
          input_type_, cnnl_shape, contiguous_strides);

      std::size_t input_copy_workspace_size = 0;
      std::size_t output_copy_workspace_size = 0;
      INFINI_OPS_CNNL_CHECK(cnnlGetCopyWorkspaceSize(
          cnnl_handle_.get(), input_desc_.get(), contiguous_desc_.get(),
          &input_copy_workspace_size));
      INFINI_OPS_CNNL_CHECK(cnnlGetCopyWorkspaceSize(
          cnnl_handle_.get(), contiguous_desc_.get(), out_desc_.get(),
          &output_copy_workspace_size));
      copy_workspace_size_ =
          std::max(input_copy_workspace_size, output_copy_workspace_size);
      tensor_buffer_size_ = AlignUp(out.numel() * out.element_size(), 128);
      workspace_size_ = 2 * tensor_buffer_size_ + copy_workspace_size_;
      default_workspace_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    }
  }

  void operator()(const Tensor input, const int64_t dim,
                  const std::optional<DataType> dtype,
                  Tensor out) const override {
    const auto normalized_dim =
        dim < 0 ? dim + static_cast<int64_t>(ndim_) : dim;
    assert(normalized_dim == dim_ &&
           "`CambriconSoftmax` dim changed after descriptor creation.");
    assert(dtype == dtype_ &&
           "`CambriconSoftmax` dtype changed after descriptor creation.");
    if (row_count_ == 0 || dim_size_ == 0) {
      return;
    }

    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(
        cnnl_handle_.get(), static_cast<cnrtQueue_t>(stream_ ? stream_ : 0)));

    if (needs_staging_) {
      auto* workspace = static_cast<char*>(
          workspace_ ? workspace_ : default_workspace_.get());
      const auto available_workspace_size =
          workspace_ ? workspace_size_in_bytes_ : workspace_size_;
      assert(available_workspace_size >= workspace_size_ &&
             "`CambriconSoftmax` requires a sufficiently large workspace.");

      void* contiguous_input = workspace;
      void* contiguous_out = workspace + tensor_buffer_size_;
      void* copy_workspace = workspace + 2 * tensor_buffer_size_;
      INFINI_OPS_CNNL_CHECK(cnnlCopy_v2(cnnl_handle_.get(), input_desc_.get(),
                                        input.data(), contiguous_desc_.get(),
                                        contiguous_input, copy_workspace,
                                        copy_workspace_size_));
      INFINI_OPS_CNNL_CHECK(cnnlSoftmaxForward(
          cnnl_handle_.get(), CNNL_SOFTMAX_ACCURATE,
          CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION, nullptr, contiguous_desc_.get(),
          contiguous_input, nullptr, contiguous_desc_.get(), contiguous_out));
      INFINI_OPS_CNNL_CHECK(cnnlCopy_v2(
          cnnl_handle_.get(), contiguous_desc_.get(), contiguous_out,
          out_desc_.get(), out.data(), copy_workspace, copy_workspace_size_));
      return;
    }

    INFINI_OPS_CNNL_CHECK(cnnlSoftmaxForward(
        cnnl_handle_.get(), CNNL_SOFTMAX_ACCURATE,
        CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION, nullptr, input_desc_.get(),
        input.data(), nullptr, out_desc_.get(), out.data()));
  }

  std::size_t workspace_size_in_bytes() const override {
    return workspace_size_;
  }

 private:
  static Tensor::Size Product(const Tensor::Shape& shape, std::size_t begin,
                              std::size_t end) {
    Tensor::Size product = 1;
    for (auto dim = begin; dim < end; ++dim) {
      product *= shape[dim];
    }
    return product;
  }

  static bool CanCollapse(const Tensor::Shape& shape,
                          const Tensor::Strides& strides, std::size_t begin,
                          std::size_t end) {
    if (end <= begin + 1) {
      return true;
    }
    for (auto dim = begin; dim + 1 < end; ++dim) {
      if (strides[dim] !=
          strides[dim + 1] * static_cast<Tensor::Stride>(shape[dim + 1])) {
        return false;
      }
    }
    return true;
  }

  static std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
  }

  cnnl_utils::Handle cnnl_handle_{};
  cnnl_utils::TensorDescriptor input_desc_{};
  cnnl_utils::TensorDescriptor out_desc_{};
  cnnl_utils::TensorDescriptor contiguous_desc_{};
  cnrt_utils::DeviceBuffer default_workspace_{};
  bool needs_staging_{false};
  std::size_t tensor_buffer_size_{0};
  std::size_t copy_workspace_size_{0};
  std::size_t workspace_size_{0};
};

}  // namespace infini::ops

#endif
