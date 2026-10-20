#ifndef INFINI_OPS_CUDA_TOPKSOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_TOPKSOFTMAX_INFINILM_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "base/topksoftmax_infinilm.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/topksoftmax_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaTopksoftmaxInfinilm : public TopksoftmaxInfinilm {
 public:
  CudaTopksoftmaxInfinilm(const Tensor input, const int64_t topk,
                          const bool norm, Tensor values, Tensor indices)
      : TopksoftmaxInfinilm{input, topk, norm, values, indices} {
    constexpr size_t ndim = 2;
    size_t strides_size = ndim * sizeof(*d_input_strides_);
    const size_t metadata_size = 3 * strides_size;
    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc((void**)&d_metadata_, metadata_size);

    size_t offset = 0;
    d_input_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_strides_.data(), strides_size);
    offset += strides_size;

    d_values_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, values_strides_.data(), strides_size);
    offset += strides_size;

    d_indices_strides_ =
        reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, indices_strides_.data(),
                strides_size);

    Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                    Backend::kMemcpyHostToDevice);
  }

  ~CudaTopksoftmaxInfinilm() { Backend::Free(d_metadata_); }

  void operator()(const Tensor input, const int64_t topk, const bool norm,
                  Tensor values, Tensor indices) const override {
    (void)topk;
    (void)norm;
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);

    DispatchFunc<AllFloatTypes, List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(input_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          const unsigned grid_x =
              static_cast<unsigned>(std::min<Tensor::Size>(row_count_, 65535));
          const unsigned grid_y = static_cast<unsigned>(
              utils::CeilDiv(row_count_, static_cast<Tensor::Size>(grid_x)));

          TopksoftmaxInfinilmKernel<kBlockSize, Backend::kDeviceType, T>
              <<<dim3(grid_x, grid_y), kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<float*>(values.data()),
                  reinterpret_cast<int32_t*>(indices.data()),
                  reinterpret_cast<const T*>(input.data()), d_input_strides_,
                  d_values_strides_, d_indices_strides_, row_count_, width_,
                  static_cast<size_t>(topk_), norm_);
        },
        "CudaTopksoftmaxInfinilm::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_values_strides_{nullptr};

  Tensor::Stride* d_indices_strides_{nullptr};
};

}  // namespace infini::ops

#endif
