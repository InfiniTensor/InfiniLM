#ifndef INFINI_OPS_CAMBRICON_EMBEDDING_KERNEL_H_
#define INFINI_OPS_CAMBRICON_EMBEDDING_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

// clang-format off
#include <cnnl.h>
#include <cnrt.h>
// clang-format on

#include "base/embedding.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/common.h"

namespace infini::ops {

namespace embedding_detail {

constexpr std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

constexpr std::size_t MetadataSize(std::size_t input_ndim) {
  return input_ndim * sizeof(std::size_t) +
         input_ndim * sizeof(std::ptrdiff_t) +
         (input_ndim + 1) * sizeof(std::ptrdiff_t);
}

constexpr std::size_t VisitedOffset(std::size_t input_ndim) {
  return AlignUp(MetadataSize(input_ndim), alignof(std::int32_t));
}

}  // namespace embedding_detail

inline std::size_t EmbeddingWorkspaceSize(std::size_t input_ndim,
                                          std::size_t vocab_size,
                                          bool apply_max_norm,
                                          bool launch_custom_forward) {
  if (!apply_max_norm && !launch_custom_forward) {
    return 0;
  }

  const auto visited_size =
      apply_max_norm ? vocab_size * sizeof(std::int32_t) : 0;

  return embedding_detail::VisitedOffset(input_ndim) + visited_size;
}

void EmbeddingKernelLaunch(
    void* workspace, DataType input_dtype, DataType weight_dtype,
    int core_per_cluster, int cluster_count, cnrtQueue_t queue, void* output,
    const void* input, void* weight, std::size_t num_indices,
    std::size_t input_ndim, const std::size_t* input_shape,
    const std::ptrdiff_t* input_strides, const std::ptrdiff_t* output_strides,
    std::ptrdiff_t weight_row_stride, std::ptrdiff_t weight_col_stride,
    std::size_t embedding_dim, std::size_t vocab_size, bool apply_max_norm,
    float max_norm, float norm_type, bool launch_custom_forward);

template <>
class Operator<Embedding, Device::Type::kCambricon> : public Embedding {
 public:
  Operator(const Tensor input, const Tensor weight,
           const std::optional<int64_t> padding_idx,
           const std::optional<double> max_norm, const double norm_type,
           const bool scale_grad_by_freq, const bool sparse, Tensor out)
      : Embedding{input,    weight,    padding_idx,
                  max_norm, norm_type, scale_grad_by_freq,
                  sparse,   out},
        input_ndim_{input.ndim()},
        weight_row_stride_{weight.stride(0)},
        weight_col_stride_{weight.stride(1)},
        use_cnnl_forward_{input.ndim() > 0 && weight.size(0) > 0 &&
                          input.IsContiguous() && weight.IsContiguous() &&
                          out.IsContiguous()} {
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster_,
                                &cluster_count_);

    if (use_cnnl_forward_) {
      cnnl_handle_ = cnnl_utils::CreateHandle();
      input_desc_ = cnnl_utils::MakeTensorDescriptor(input_dtype_, input_shape_,
                                                     input_strides_);
      weight_desc_ = cnnl_utils::MakeTensorDescriptor(
          weight_dtype_, weight_shape_, weight_strides_);
      out_desc_ = cnnl_utils::MakeTensorDescriptor(out_dtype_, out_shape_,
                                                   out_strides_);
    }

    workspace_size_ = EmbeddingWorkspaceSize(
        input_ndim_, vocab_size_, max_norm.has_value(), !use_cnnl_forward_);
    default_workspace_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
  }

  Operator(const Tensor input, const Tensor weight, Tensor out)
      : Operator(input, weight, std::nullopt, std::nullopt, 2.0, false, false,
                 out) {}

  /// \deprecated Use the overload that also accepts `max_norm` and
  /// `norm_type` instead.
  [[deprecated("Use the PyTorch-compatible overload instead.")]]
  Operator(const Tensor input, const Tensor weight, const int64_t padding_idx,
           const bool scale_grad_by_freq, const bool sparse, Tensor out)
      : Operator(input, weight, padding_idx, std::nullopt, 2.0,
                 scale_grad_by_freq, sparse, out) {}

  std::size_t workspace_size_in_bytes() const override {
    return workspace_size_;
  }

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<int64_t> /*padding_idx*/,
                  const std::optional<double> max_norm, const double norm_type,
                  const bool /*scale_grad_by_freq*/, const bool /*sparse*/,
                  Tensor out) const override {
    if (num_indices_ == 0 || embedding_dim_ == 0) {
      return;
    }

    assert(max_norm.has_value() == max_norm_.has_value() &&
           "`CambriconEmbedding` max_norm presence changed after creation");

    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    const bool launch_custom_forward = !use_cnnl_forward_;

    if (max_norm.has_value() || launch_custom_forward) {
      void* workspace = workspace_ ? workspace_ : default_workspace_.get();
      [[maybe_unused]] const auto workspace_size =
          workspace_ ? workspace_size_in_bytes_ : workspace_size_;
      assert(workspace && workspace_size >= workspace_size_ &&
             "`CambriconEmbedding` requires a sufficiently large workspace.");

      EmbeddingKernelLaunch(
          workspace, input.dtype(), weight.dtype(), core_per_cluster_,
          cluster_count_, queue, out.data(), input.data(),
          const_cast<void*>(weight.data()), num_indices_, input_ndim_,
          input_shape_.data(), input_strides_.data(), out_strides_.data(),
          weight_row_stride_, weight_col_stride_, embedding_dim_, vocab_size_,
          max_norm.has_value(), static_cast<float>(max_norm.value_or(0.0)),
          static_cast<float>(norm_type), launch_custom_forward);
    }

    if (launch_custom_forward) {
      return;
    }

    [[maybe_unused]] const auto set_queue_status =
        cnnlSetQueue(cnnl_handle_.get(), queue);
    assert(set_queue_status == CNNL_STATUS_SUCCESS && "`cnnlSetQueue` failed.");

    // A non-negative CNNL padding_idx zeroes the corresponding weight row.
    // InfiniOps forward semantics only use padding_idx for backward behavior.
    [[maybe_unused]] const auto embedding_status = cnnlEmbeddingForward_v2(
        cnnl_handle_.get(), weight_desc_.get(), weight.data(),
        input_desc_.get(), input.data(), -1, nullptr, nullptr, out_desc_.get(),
        out.data());
    assert(embedding_status == CNNL_STATUS_SUCCESS &&
           "`cnnlEmbeddingForward_v2` failed.");
  }

 private:
  std::size_t input_ndim_{0};

  std::ptrdiff_t weight_row_stride_{0};

  std::ptrdiff_t weight_col_stride_{0};

  int core_per_cluster_{0};

  int cluster_count_{0};

  std::size_t workspace_size_{0};

  cnrt_utils::DeviceBuffer default_workspace_{};

  bool use_cnnl_forward_{false};

  cnnl_utils::Handle cnnl_handle_{};

  cnnl_utils::TensorDescriptor input_desc_{};

  cnnl_utils::TensorDescriptor weight_desc_{};

  cnnl_utils::TensorDescriptor out_desc_{};
};

}  // namespace infini::ops

#endif
