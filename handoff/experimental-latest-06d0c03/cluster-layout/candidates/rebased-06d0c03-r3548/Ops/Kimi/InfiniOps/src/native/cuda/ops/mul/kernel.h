#ifndef INFINI_OPS_CUDA_MUL_KERNEL_H_
#define INFINI_OPS_CUDA_MUL_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "base/mul.h"
#include "common/generic_utils.h"
#include "native/cuda/device_buffer.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/mul/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaMul : public Mul {
 public:
  CudaMul(const Tensor input, const Tensor other, Tensor out)
      : Mul{input, other, out} {
    size_t shape_size = ndim_ * sizeof(*d_input_shape_);
    size_t strides_size = ndim_ * sizeof(*d_input_strides_);
    const size_t metadata_size = 3 * (shape_size + strides_size);
    if (metadata_size == 0) {
      return;
    }

    std::vector<std::byte> metadata(metadata_size);

    auto* d_metadata_ = metadata_.Allocate(metadata_size);

    size_t offset = 0;
    d_input_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_shape_.data(), shape_size);
    offset += shape_size;

    d_other_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, other_shape_.data(), shape_size);
    offset += shape_size;

    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_shape_.data(), shape_size);
    offset += shape_size;

    d_input_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_strides_.data(), strides_size);
    offset += strides_size;

    d_other_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, other_strides_.data(), strides_size);
    offset += strides_size;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_strides_.data(), strides_size);

    if (Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                        Backend::kMemcpyHostToDevice) != Backend::kSuccess)
      throw std::runtime_error("CudaMul metadata copy failed");
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    if (scalar_mode_)
      throw std::invalid_argument("Mul prepared for a scalar operand");
    if (output_size_ == 0) {
      return;
    }

    DispatchFunc<AllTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(out_type_), block_size_},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
          dim3 block_dims(
              std::min(static_cast<Tensor::Size>(block_size_), output_size_));
          dim3 grid_dims(utils::CeilDiv(output_size_, block_dims.x));

          T* d_out = reinterpret_cast<T*>(out.data());
          const T* d_input = reinterpret_cast<const T*>(input.data());
          const T* d_other = reinterpret_cast<const T*>(other.data());

          MulKernel<Backend::kDeviceType, T, kBlockSize>
              <<<grid_dims, block_dims, 0, cuda_stream>>>(
                  d_out, d_input, d_other, d_out_shape_, d_input_shape_,
                  d_other_shape_, d_out_strides_, d_input_strides_,
                  d_other_strides_, output_size_, ndim_, is_out_contiguous_,
                  is_input_contiguous_, is_other_contiguous_);
        },
        "CudaMul::operator()");
  }

 protected:
  CudaMul(const Tensor input, const double alpha, Tensor out)
      : Mul{input, alpha, out}, block_size_(0) {}

 private:
  const int block_size_ =
      RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();
  CudaDeviceBuffer<Backend> metadata_;

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_other_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_other_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
