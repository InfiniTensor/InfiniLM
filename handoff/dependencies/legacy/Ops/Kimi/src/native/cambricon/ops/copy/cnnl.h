#ifndef INFINI_OPS_CAMBRICON_COPY_CNNL_H_
#define INFINI_OPS_CAMBRICON_COPY_CNNL_H_

#include <algorithm>
#include <cassert>

#include "base/copy.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kCambricon> : public Copy {
 public:
  Operator(const Tensor src, const bool non_blocking, Tensor out)
      : Copy{src, non_blocking, out} {
    if (output_size_ == 0) {
      return;
    }

    assert(std::all_of(input_strides_.begin(), input_strides_.end(),
                       [](auto stride) { return stride >= 0; }) &&
           "`CambriconCopy` does not support negative input strides.");
    assert(std::all_of(out_strides_.begin(), out_strides_.end(),
                       [](auto stride) { return stride >= 0; }) &&
           "`CambriconCopy` does not support negative output strides.");

    cnnl_handle_ = cnnl_utils::CreateHandle();
    input_desc_ = cnnl_utils::MakeTensorDescriptor(input_type_, input_shape_,
                                                   input_strides_);
    out_desc_ =
        cnnl_utils::MakeTensorDescriptor(out_type_, out_shape_, out_strides_);

    INFINI_OPS_CNNL_CHECK(
        cnnlGetCopyWorkspaceSize(cnnl_handle_.get(), input_desc_.get(),
                                 out_desc_.get(), &workspace_size_));

    default_workspace_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
  }

  void operator()(const Tensor src, const bool /*non_blocking*/,
                  Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(
        cnnl_handle_.get(), static_cast<cnrtQueue_t>(stream_ ? stream_ : 0)));

    void* workspace = workspace_ ? workspace_ : default_workspace_.get();
    const auto workspace_size =
        workspace_ ? workspace_size_in_bytes_ : workspace_size_;
    assert(workspace_size >= workspace_size_ &&
           "`CambriconCopy` requires a sufficiently large workspace.");

    INFINI_OPS_CNNL_CHECK(cnnlCopy_v2(cnnl_handle_.get(), input_desc_.get(),
                                      src.data(), out_desc_.get(), out.data(),
                                      workspace, workspace_size_));
  }

  std::size_t workspace_size_in_bytes() const override {
    return workspace_size_;
  }

 private:
  std::size_t workspace_size_{0};

  cnrt_utils::DeviceBuffer default_workspace_{};

  cnnl_utils::Handle cnnl_handle_{};

  cnnl_utils::TensorDescriptor input_desc_{};

  cnnl_utils::TensorDescriptor out_desc_{};
};

}  // namespace infini::ops

#endif
