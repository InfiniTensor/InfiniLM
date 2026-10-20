#ifndef INFINI_OPS_CUDA_ARGMAX_KERNEL_H_
#define INFINI_OPS_CUDA_ARGMAX_KERNEL_H_

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "base/argmax.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/argmax/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaArgmax : public Argmax {
 public:
  CudaArgmax(const Tensor input, const std::optional<int64_t> dim,
             const bool keepdim, Tensor out)
      : Argmax{input, dim, keepdim, out},
        numel_{input.numel()},
        workspace_size_{DispatchWorkspaceSize(input.dtype(), input.numel())} {
    assert(!dim.has_value() && !keepdim &&
           "`CudaArgmax` currently supports only flattened reduction");
    assert(input.IsContiguous() && "`CudaArgmax` requires contiguous input");
    assert(input.numel() > 0 && input.numel() <= INT_MAX &&
           "`CudaArgmax` input size must fit in a positive `int`");
    assert(out.numel() == 1 && out.dtype() == DataType::kInt64 &&
           "`CudaArgmax` requires one `int64` output value");
    assert(input.device() == out.device() &&
           "`CudaArgmax` input and output must be on the same device");

    auto error = Backend::Malloc(&default_workspace_, workspace_size_);
    assert(error == 0 && "`CudaArgmax` failed to allocate workspace");
  }

  ~CudaArgmax() override {
    if (default_workspace_ != nullptr) {
      auto error = Backend::Free(default_workspace_);
      assert(error == 0 && "`CudaArgmax` failed to free workspace");
    }
  }

  std::size_t workspace_size_in_bytes() const override {
    return workspace_size_;
  }

  void operator()(const Tensor input, const std::optional<int64_t> dim,
                  const bool keepdim, Tensor out) const override {
    assert(!dim.has_value() && !keepdim);
    assert(input.shape() == input_shape_ && input.strides() == input_strides_ &&
           input.dtype() == input_type_);
    assert(out.shape() == out_shape_ && out.strides() == out_strides_ &&
           out.dtype() == out_type_);

    void* workspace = workspace_ ? workspace_ : default_workspace_;
    auto workspace_size =
        workspace_ ? workspace_size_in_bytes_ : workspace_size_;
    assert(workspace != nullptr && workspace_size >= workspace_size_ &&
           "`CudaArgmax` received insufficient workspace");
    auto stream = static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    DispatchFunc<Backend::kDeviceType,
                 ConcatType<FloatTypes, ReducedFloatTypes>>(
        input.dtype(),
        [&](auto dtype_tag) {
          using T = typename decltype(dtype_tag)::type;
          cuda_argmax_detail::Launch(
              workspace, workspace_size, static_cast<const T*>(input.data()),
              numel_, static_cast<int64_t*>(out.data()), stream);
        },
        "CudaArgmax::operator()");
  }

 private:
  static std::size_t DispatchWorkspaceSize(DataType dtype, std::size_t numel) {
    std::size_t workspace_size = 0;
    DispatchFunc<Backend::kDeviceType,
                 ConcatType<FloatTypes, ReducedFloatTypes>>(
        dtype,
        [&](auto dtype_tag) {
          using T = typename decltype(dtype_tag)::type;
          workspace_size = cuda_argmax_detail::WorkspaceSize<T>(numel);
        },
        "CudaArgmax::DispatchWorkspaceSize");

    return workspace_size;
  }

  std::size_t numel_{0};

  std::size_t workspace_size_{0};

  void* default_workspace_{nullptr};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_ARGMAX_KERNEL_H_
