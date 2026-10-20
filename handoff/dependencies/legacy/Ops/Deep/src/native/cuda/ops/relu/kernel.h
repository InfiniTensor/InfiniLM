#ifndef INFINI_OPS_CUDA_RELU_KERNEL_H_
#define INFINI_OPS_CUDA_RELU_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "base/relu.h"
#include "common/generic_utils.h"
#include "host_range_profiler.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/relu/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaRelu : public Relu {
 public:
  CudaRelu(const Tensor input, Tensor out) : Relu{input, out} {
    size_t shape_size = ndim_ * sizeof(*d_input_shape_);
    size_t strides_size = ndim_ * sizeof(*d_input_strides_);
    const size_t metadata_size = 2 * (shape_size + strides_size);

    if (metadata_size == 0) {
      return;
    }

    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc((void**)&d_metadata_, metadata_size);

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

    Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                    Backend::kMemcpyHostToDevice);
  }

  ~CudaRelu() {
    if (d_metadata_ != nullptr) {
      Backend::Free(d_metadata_);
    }
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    [[maybe_unused]] HostRangeScope host_range_backend_submit{
        HostRangeLayer::kBackendSubmit};

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);
    dim3 block(std::min(static_cast<Tensor::Size>(block_size), output_size_));
    dim3 grid(utils::CeilDiv(output_size_, block.x));

    DispatchFunc<ReluDataTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(out_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto* out_ptr = reinterpret_cast<T*>(out.data());
          const auto* input_ptr = reinterpret_cast<const T*>(input.data());
          T* input_copy = nullptr;
          bool input_contiguous = is_input_contiguous_;

          if (NeedsInputCopy(input, out)) {
            Backend::Malloc(reinterpret_cast<void**>(&input_copy),
                            output_size_ * sizeof(T));
            ReluKernel<Backend::kDeviceType, T, kBlockSize>
                <<<grid, block, 0, cuda_stream>>>(
                    input_copy, input_ptr, d_out_shape_, d_input_shape_,
                    d_out_strides_, d_input_strides_, output_size_, ndim_, true,
                    is_input_contiguous_);
            input_ptr = input_copy;
            input_contiguous = true;
          }

          ReluKernel<Backend::kDeviceType, T, kBlockSize>
              <<<grid, block, 0, cuda_stream>>>(
                  out_ptr, input_ptr, d_out_shape_, d_input_shape_,
                  d_out_strides_, d_input_strides_, output_size_, ndim_,
                  is_out_contiguous_, input_contiguous);

          if (input_copy != nullptr) {
            Backend::Free(input_copy);
          }
        },
        "CudaRelu::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
