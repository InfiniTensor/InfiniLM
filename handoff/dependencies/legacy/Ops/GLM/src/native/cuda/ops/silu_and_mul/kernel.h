#ifndef INFINI_OPS_CUDA_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_CUDA_SILU_AND_MUL_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <vector>

#include "base/silu_and_mul.h"
#include "common/generic_utils.h"
#include "native/cuda/ops/silu_and_mul/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaSiluAndMul : public SiluAndMul {
 public:
  CudaSiluAndMul(const Tensor input, Tensor out) : SiluAndMul{input, out} {
    if (Backend::GetDevice(&owner_device_) != Backend::kSuccess)
      throw std::runtime_error("CudaSiluAndMul cannot query metadata owner device");
    if (output_size_ == 0 || (is_input_contiguous_ && is_out_contiguous_))
      return;

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

      if (Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                          Backend::kMemcpyHostToDevice) != Backend::kSuccess)
        throw std::runtime_error("CudaSiluAndMul metadata copy failed");
    } catch (...) {
      if (Backend::Free(d_metadata_) != Backend::kSuccess) std::terminate();
      d_metadata_ = nullptr;
      throw;
    }
  }

  ~CudaSiluAndMul() {
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

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();
    DispatchFunc<AllFloatTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(out_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(block_size), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));

          T* d_out = reinterpret_cast<T*>(out.data());
          const T* d_input = reinterpret_cast<const T*>(input.data());
          SiluAndMulKernel<Backend::kDeviceType, T, kBlockSize>
              <<<gridDims, blockDims, 0, cuda_stream>>>(
                  d_out, d_input, d_out_shape_, d_input_shape_, d_out_strides_,
                  d_input_strides_, output_size_, ndim_, hidden_size_,
                  is_out_contiguous_, is_input_contiguous_);
        },
        "CudaSiluAndMul::operator()");
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
