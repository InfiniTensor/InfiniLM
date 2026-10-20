#ifndef INFINI_OPS_CUDA_COPY_KERNEL_H_
#define INFINI_OPS_CUDA_COPY_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <vector>

#include "base/copy.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/copy/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaCopy : public Copy {
 public:
  CudaCopy(const Tensor input, const bool non_blocking, Tensor out)
      : Copy{input, non_blocking, out} {
    if (Backend::GetDevice(&owner_device_) != Backend::kSuccess)
      throw std::runtime_error("CudaCopy cannot query metadata owner device");
    if (!output_size_ || (is_input_contiguous_ && is_out_contiguous_)) return;
    size_t shape_size = ndim_ * sizeof(*d_input_shape_);
    size_t strides_size = ndim_ * sizeof(*d_input_strides_);
    const size_t metadata_size = 2 * (shape_size + strides_size);
    std::vector<std::byte> metadata(metadata_size);

    void* allocation = nullptr;
    const auto allocation_status = Backend::Malloc(&allocation, metadata_size);
    if (allocation_status != Backend::kSuccess || !allocation) {
      if (allocation && Backend::Free(allocation) != Backend::kSuccess)
        std::terminate();
      throw std::bad_alloc();
    }
    d_metadata_ = static_cast<std::byte*>(allocation);

    try {
      size_t offset = 0;
      d_input_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, input_shape_.data(), shape_size);
      offset += shape_size;

      d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, out_shape_.data(), shape_size);
      offset += shape_size;

      d_input_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, input_strides_.data(), strides_size);
      offset += strides_size;

      d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, out_strides_.data(), strides_size);

      const bool copied =
          Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                          Backend::kMemcpyHostToDevice) == Backend::kSuccess;
      if (!copied)
        throw std::runtime_error("CudaCopy metadata copy failed");
    } catch (...) {
      if (Backend::Free(d_metadata_) != Backend::kSuccess) std::terminate();
      d_metadata_ = nullptr;
      throw;
    }
  }

  ~CudaCopy() {
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

  void operator()(const Tensor input, const bool /*non_blocking*/,
                  Tensor out) const override {
    if (output_size_ == 0 || SameView(input, out)) return;

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);
    dim3 block(std::min(static_cast<Tensor::Size>(block_size), output_size_));
    dim3 grid(utils::CeilDiv(output_size_, block.x));

    DispatchFunc<AllTypes, List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(out_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          CopyKernel<T, kBlockSize><<<grid, block, 0, cuda_stream>>>(
              reinterpret_cast<T*>(out.data()),
              reinterpret_cast<const T*>(input.data()), d_out_shape_,
              d_input_shape_, d_out_strides_, d_input_strides_, output_size_,
              ndim_, is_out_contiguous_, is_input_contiguous_);
        },
        "CudaCopy::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};
  int owner_device_{-1};

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
