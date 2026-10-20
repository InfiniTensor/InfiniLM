#ifndef INFINI_OPS_HYGON_ARGMAX_KERNEL_H_
#define INFINI_OPS_HYGON_ARGMAX_KERNEL_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <climits>
#include <limits>
#include <optional>
#include <stdexcept>

#include "base/argmax.h"
#include "native/cuda/hygon/ops/argmax/kernel.cuh"
#include "native/cuda/hygon/runtime_.h"

namespace infini::ops {

template <>
class Operator<Argmax, Device::Type::kHygon> : public Argmax {
 public:
  Operator(const Tensor input, const std::optional<int64_t> dim,
           const bool keepdim, Tensor out)
      : Argmax(input, dim, keepdim, out) {
    Validate(input, dim, keepdim, out);
    const auto rows = input.ndim() == 2 ? input.shape()[0] : 1;
    const auto width = input.shape().back();
    if (width > hygon_argmax_rows_r23_detail::kChunkWidth) {
      const auto chunks = (width - 1) / hygon_argmax_rows_r23_detail::kChunkWidth + 1;
      if (rows > std::numeric_limits<std::size_t>::max() / chunks /
                     sizeof(hygon_argmax_rows_r23_detail::Candidate)) {
        throw std::invalid_argument("HYGON Argmax workspace size overflows");
      }
      workspace_bytes_ = rows * chunks * sizeof(hygon_argmax_rows_r23_detail::Candidate);
    }
  }

  bool capture_safe() const override { return true; }

  std::size_t workspace_size_in_bytes() const override { return workspace_bytes_; }

  bool requires_workspace() const override { return workspace_bytes_ != 0; }

  void operator()(const Tensor input, const std::optional<int64_t> dim,
                  const bool keepdim, Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        input.dtype() != input_type_ || out.shape() != out_shape_ ||
        out.strides() != out_strides_ || out.dtype() != out_type_ ||
        dim != dim_ || keepdim != keepdim_ ||
        out.device().index() != device_index_) {
      throw std::invalid_argument("HYGON Argmax call metadata changed after Make");
    }
    Validate(input, dim, keepdim, out);
    const auto rows = static_cast<unsigned>(input.ndim() == 2 ? input.shape()[0] : 1);
    const auto width = static_cast<int>(input.shape().back());
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
    if (workspace_bytes_) CheckWorkspace(input, out);
    switch (input.dtype()) {
      case DataType::kFloat16:
        Launch(static_cast<const half*>(input.data()), out, rows, width, stream,
               workspace_);
        break;
      case DataType::kBFloat16:
        Launch(static_cast<const __nv_bfloat16*>(input.data()), out, rows,
               width, stream, workspace_);
        break;
      case DataType::kFloat32:
        Launch(static_cast<const float*>(input.data()), out, rows, width, stream,
               workspace_);
        break;
      default:
        throw std::invalid_argument("HYGON Argmax unsupported dtype");
    }
  }

 private:
  void CheckWorkspace(const Tensor input, const Tensor out) const {
    if (!workspace_ || workspace_size_in_bytes_ < workspace_bytes_ ||
        reinterpret_cast<std::uintptr_t>(workspace_) %
            alignof(hygon_argmax_rows_r23_detail::Candidate)) {
      throw std::invalid_argument("HYGON Argmax workspace is missing or too small");
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(workspace_);
    if (workspace_bytes_ > std::numeric_limits<std::uintptr_t>::max() - begin) {
      throw std::invalid_argument("HYGON Argmax workspace span overflows");
    }
    const auto end = begin + workspace_bytes_;
    const auto overlaps = [begin, end](const Tensor tensor) {
      const auto tensor_begin = reinterpret_cast<std::uintptr_t>(tensor.data());
      const auto tensor_end = tensor_begin + tensor.numel() * tensor.element_size();
      return begin < tensor_end && tensor_begin < end;
    };
    if (overlaps(input) || overlaps(out)) {
      throw std::invalid_argument("HYGON Argmax workspace overlaps a tensor");
    }
  }

  static void Validate(const Tensor input, const std::optional<int64_t> dim,
                       bool keepdim, const Tensor out) {
    if (input.device().type() != Device::Type::kHygon ||
        input.device() != out.device() ||
        (input.ndim() != 1 && input.ndim() != 2) || keepdim ||
        (input.ndim() == 1 ? dim.has_value() : dim != 1) ||
        !input.IsContiguous() || !out.IsContiguous() ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32) ||
        out.dtype() != DataType::kInt64) {
      throw std::invalid_argument("unsupported HYGON Argmax metadata");
    }
    const auto rows = input.ndim() == 2 ? input.shape()[0] : 1;
    const auto width = input.shape().back();
    if (!rows || !width || rows > static_cast<std::size_t>(INT_MAX) ||
        width > static_cast<std::size_t>(INT_MAX) ||
        (input.ndim() == 1 ? out.ndim() != 0
                           : out.ndim() != 1 || out.shape()[0] != rows) ||
        !input.data() || !out.data()) {
      throw std::invalid_argument("HYGON Argmax has empty or invalid storage");
    }
    if (rows > std::numeric_limits<std::size_t>::max() / width ||
        rows * width > std::numeric_limits<std::size_t>::max() /
                           input.element_size() ||
        rows > std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t)) {
      throw std::invalid_argument("HYGON Argmax storage size overflows");
    }
    const auto input_bytes = input.numel() * input.element_size();
    const auto output_bytes = out.numel() * out.element_size();
    const auto in_begin = reinterpret_cast<std::uintptr_t>(input.data());
    const auto out_begin = reinterpret_cast<std::uintptr_t>(out.data());
    if (input_bytes > std::numeric_limits<std::uintptr_t>::max() - in_begin ||
        output_bytes > std::numeric_limits<std::uintptr_t>::max() - out_begin ||
        (in_begin < out_begin + output_bytes &&
         out_begin < in_begin + input_bytes)) {
      throw std::invalid_argument("HYGON Argmax input/output storage overlaps");
    }
    int current_device = -1;
    const auto status = cudaGetDevice(&current_device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    if (current_device != input.device().index()) {
      throw std::invalid_argument("HYGON Argmax tensor device is not current device");
    }
  }

  template <typename T>
  static void Launch(const T* input, Tensor out, unsigned rows, int width,
                     cudaStream_t stream, void* workspace) {
    if (width <= hygon_argmax_rows_r23_detail::kChunkWidth) {
      hygon_argmax_rows_r23_detail::RowsKernel<T>
          <<<rows, hygon_argmax_rows_r23_detail::kThreads, 0, stream>>>(
              input, static_cast<std::int64_t*>(out.data()), width);
    } else {
      const int chunks = (width - 1) / hygon_argmax_rows_r23_detail::kChunkWidth + 1;
      const auto tasks = static_cast<std::size_t>(rows) * chunks;
      auto* partials = static_cast<hygon_argmax_rows_r23_detail::Candidate*>(workspace);
      const auto partial_grid = static_cast<unsigned>(
          std::min<std::size_t>(tasks, 65535));
      const auto final_grid = std::min<unsigned>(rows, 65535);
      hygon_argmax_rows_r23_detail::PartialsKernel<T>
          <<<partial_grid, hygon_argmax_rows_r23_detail::kThreads, 0, stream>>>(
              input, partials, static_cast<int>(rows), width, chunks);
      const auto partial_status = cudaPeekAtLastError();
      if (partial_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(partial_status));
      }
      hygon_argmax_rows_r23_detail::FinishKernel
          <<<final_grid, hygon_argmax_rows_r23_detail::kThreads, 0, stream>>>(
              partials, static_cast<std::int64_t*>(out.data()),
              static_cast<int>(rows), chunks);
    }
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }

  std::size_t workspace_bytes_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_ARGMAX_KERNEL_H_
