#ifndef INFINI_OPS_NVIDIA_CAT_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAT_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "base/cat.h"
#include "base/copy.h"
#include "native/cuda/nvidia/ops/copy/kernel.h"

namespace infini::ops::nvidia_cat_detail {
void LaunchAxis0Bytes(const void* first, const void* second, void* out,
                      std::size_t first_bytes, std::size_t total_bytes,
                      cudaStream_t stream);
}  // namespace infini::ops::nvidia_cat_detail

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kNvidia> : public Cat {
 public:
  static bool supports(const std::vector<Tensor>& tensors, int64_t dim,
                       Tensor out) {
    if (out.device().type() != Device::Type::kNvidia ||
        !Cat::supports(tensors, dim, out)) return false;
    const auto normalized = dim < 0 ? dim + static_cast<int64_t>(out.ndim()) : dim;
    if (FastAxis0(tensors, normalized, out)) return true;
    return true;
  }

  Operator(const std::vector<Tensor> tensors, int64_t dim, Tensor out)
      : Cat(tensors, dim, out) {
    if (!supports(tensors, dim, out))
      throw std::invalid_argument("NVIDIA Cat metadata or Copy provider unsupported");
    fast_axis0_ = FastAxis0(tensors, dim_, out);
    if (fast_axis0_) {
      first_bytes_ = tensors[0].numel() * out.element_size();
      total_bytes_ = out.numel() * out.element_size();
      capture_safe_ = true;
      return;
    }
    const auto slices = Slices(tensors, dim_, out);
    copies_.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i)
      copies_.push_back(Copy::Make(Config::Automatic(), tensors[i], false, slices[i]));
    capture_safe_ = !copies_.empty();
    for (const auto& copy : copies_) capture_safe_ &= copy->capture_safe();
  }

  bool capture_safe() const override { return capture_safe_; }

  void operator()(const std::vector<Tensor> tensors, int64_t dim,
                  Tensor out) const override {
    if (fast_axis0_) {
      nvidia_cat_detail::LaunchAxis0Bytes(
          tensors[0].data(), tensors[1].data(), out.data(), first_bytes_,
          total_bytes_, static_cast<cudaStream_t>(stream_ ? stream_ : 0));
      return;
    }
    const auto slices = Slices(tensors, dim_, out);
    for (size_t i = 0; i < tensors.size(); ++i)
      (*copies_[i])(*handle_ptr_, tensors[i], false, slices[i]);
  }

 private:
  static bool FastAxis0(const std::vector<Tensor>& tensors, int64_t dim,
                        Tensor out) {
    return dim == 0 && tensors.size() == 2 && out.numel() != 0 &&
           tensors[0].numel() != 0 && tensors[1].numel() != 0 &&
           out.IsContiguous() && tensors[0].IsContiguous() &&
           tensors[1].IsContiguous();
  }

  static std::vector<Tensor> Slices(const std::vector<Tensor>& tensors,
                                    int64_t dim, Tensor out) {
    std::vector<Tensor> slices;
    slices.reserve(tensors.size());
    size_t offset = 0;
    for (const auto& tensor : tensors) {
      void* data = out.data();
      if (tensor.numel())
        data = static_cast<std::byte*>(data) +
               offset * static_cast<size_t>(out.stride(dim)) * out.element_size();
      slices.emplace_back(data, tensor.shape(), out.dtype(), out.device(),
                          out.strides());
      offset += tensor.size(dim);
    }
    return slices;
  }

  std::vector<std::unique_ptr<Operator<Copy>>> copies_;
  bool capture_safe_{false};
  bool fast_axis0_{false};
  size_t first_bytes_{0};
  size_t total_bytes_{0};
};

}  // namespace infini::ops

#endif
