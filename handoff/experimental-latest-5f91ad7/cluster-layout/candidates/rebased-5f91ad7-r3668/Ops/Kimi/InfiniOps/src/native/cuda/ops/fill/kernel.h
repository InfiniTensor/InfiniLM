#ifndef INFINI_OPS_CUDA_FILL_KERNEL_H_
#define INFINI_OPS_CUDA_FILL_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "base/fill.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/device_buffer.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/fill/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaFill : public Fill {
 public:
  CudaFill(const Tensor input, const double value, Tensor out)
      : Fill{input, value, out},
        output_size_{out.numel()},
        ndim_{out.ndim()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(input.shape() == out.shape() &&
           "`CudaFill` requires `input` and `out` to have the same shape.");
    assert(input.dtype() == out.dtype() &&
           "`CudaFill` requires `input` and `out` to have the same dtype.");
    assert(input.device() == out.device() &&
           "`CudaFill` requires `input` and `out` on the same device.");
    assert(!out.HasBroadcastDim() &&
           "`CudaFill` output must not have broadcast dimensions.");
    if (output_size_ == 0 || is_out_contiguous_) return;

    size_t shape_size = ndim_ * sizeof(*d_out_shape_);
    size_t strides_size = ndim_ * sizeof(*d_out_strides_);
    const size_t metadata_size = shape_size + strides_size;
    std::vector<std::byte> metadata(metadata_size);

    auto* d_metadata_ = metadata_.Allocate(metadata_size);

    size_t offset = 0;
    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_shape_.data(), shape_size);
    offset += shape_size;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_strides_.data(), strides_size);

    if (Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                        Backend::kMemcpyHostToDevice) != Backend::kSuccess)
      throw std::runtime_error("CudaFill metadata copy failed");
  }

  void operator()(const Tensor input, const double value,
                  Tensor out) const override {
    (void)input;
    if (output_size_ == 0) {
      return;
    }

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    dim3 block(std::min(static_cast<Tensor::Size>(block_size_), output_size_));
    dim3 grid(utils::CeilDiv(output_size_, block.x));

    DispatchFunc<AllTypes, List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(out_type_), block_size_},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          FillKernel<Backend::kDeviceType, T, kBlockSize>
              <<<grid, block, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()), d_out_shape_,
                  d_out_strides_, output_size_, ndim_, is_out_contiguous_,
                  value);
        },
        "CudaFill::operator()");
  }

  void operator()(const Tensor input, const Tensor value,
                  Tensor out) const override {
    (void)input;
    (void)value;
    (void)out;
    assert(false && "The CUDA `Fill` provider does not support Tensor values.");
  }

 private:
  const int block_size_ =
      std::min(RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);
  Tensor::Size output_size_{0};

  Tensor::Size ndim_{0};

  bool is_out_contiguous_{false};

  CudaDeviceBuffer<Backend> metadata_;

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_FILL_KERNEL_H_
