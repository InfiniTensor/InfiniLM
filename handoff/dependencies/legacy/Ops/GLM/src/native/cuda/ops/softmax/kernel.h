#ifndef INFINI_OPS_CUDA_SOFTMAX_KERNEL_H_
#define INFINI_OPS_CUDA_SOFTMAX_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <vector>

#include "base/softmax.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/softmax_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaSoftmax : public Softmax {
 public:
  CudaSoftmax(const Tensor input, const int64_t dim,
              const std::optional<DataType> dtype, Tensor out)
      : Softmax{input, dim, dtype, out} {
    assert(input_shape_ == out_shape_ &&
           "`Softmax` input and output shapes must match");
    assert(dim_ >= 0 && dim_ < static_cast<int64_t>(ndim_) &&
           "`Softmax` dim out of range");
    assert(!dtype_.has_value() || dtype_.value() == out_type_);
    assert(input_type_ == out_type_ &&
           "`Softmax` native provider requires matching input and output "
           "dtypes");
    assert(input.device() == out.device() &&
           "`Softmax` input and output devices must match");
    assert(!out.HasBroadcastDim() &&
           "`Softmax` output must not have broadcasted dimensions");

    if (Backend::GetDevice(&owner_device_) != Backend::kSuccess)
      throw std::runtime_error("CudaSoftmax cannot query metadata owner device");
    block_size_ = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);
    if (row_count_ == 0 || dim_size_ == 0) return;

    const size_t shape_size = ndim_ * sizeof(*d_shape_);
    const size_t strides_size = ndim_ * sizeof(*d_input_strides_);
    const size_t metadata_size = shape_size + 2 * strides_size;
    if (metadata_size == 0) {
      return;
    }

    std::vector<std::byte> metadata(metadata_size);
    void* allocation = nullptr;
    const auto status = Backend::Malloc(&allocation, metadata_size);
    if (status != Backend::kSuccess || !allocation) {
      if (allocation && Backend::Free(allocation) != Backend::kSuccess)
        std::terminate();
      throw std::bad_alloc();
    }
    d_metadata_ = static_cast<std::byte*>(allocation);

    try {
      size_t offset = 0;
      d_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, out_shape_.data(), shape_size);
      offset += shape_size;

      d_input_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, input_strides_.data(), strides_size);
      offset += strides_size;

      d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, out_strides_.data(), strides_size);

      if (Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                          Backend::kMemcpyHostToDevice) != Backend::kSuccess)
        throw std::runtime_error("CudaSoftmax metadata copy failed");
    } catch (...) {
      if (Backend::Free(d_metadata_) != Backend::kSuccess) std::terminate();
      d_metadata_ = nullptr;
      throw;
    }
  }

  ~CudaSoftmax() {
    if (!d_metadata_) return;
    int previous = -1;
    if (Backend::GetDevice(&previous) != Backend::kSuccess) std::terminate();
    const bool switched = previous != owner_device_;
    if (switched && Backend::SetDevice(owner_device_) != Backend::kSuccess)
      std::terminate();
    const auto freed = Backend::Free(d_metadata_);
    if (switched && Backend::SetDevice(previous) != Backend::kSuccess)
      std::terminate();
    if (freed != Backend::kSuccess) std::terminate();
  }

  void operator()(const Tensor input, const int64_t,
                  const std::optional<DataType>, Tensor out) const override {
    if (row_count_ == 0 || dim_size_ == 0) {
      return;
    }

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    DispatchFunc<AllFloatTypes, List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(out_type_), block_size_},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          const unsigned grid_x =
              static_cast<unsigned>(std::min<Tensor::Size>(row_count_, 65535));
          const unsigned grid_y = static_cast<unsigned>(
              utils::CeilDiv(row_count_, static_cast<Tensor::Size>(grid_x)));

          SoftmaxInfinilmKernel<kBlockSize, Backend::kDeviceType, T>
              <<<dim3(grid_x, grid_y), kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(input.data()), d_shape_,
                  d_out_strides_, d_input_strides_, row_count_, dim_size_,
                  ndim_, dim_);
        },
        "CudaSoftmax::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};
  int owner_device_{-1};
  int block_size_{0};

  Tensor::Size* d_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
