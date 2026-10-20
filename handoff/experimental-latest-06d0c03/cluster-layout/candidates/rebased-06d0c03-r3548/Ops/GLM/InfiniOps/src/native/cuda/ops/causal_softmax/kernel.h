#ifndef INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "base/causal_softmax.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/causal_softmax/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaCausalSoftmax : public CausalSoftmax {
 public:
  using CausalSoftmax::CausalSoftmax;

  void operator()(const Tensor input, Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    auto stride_input_batch = ndim_ == 3 ? input_strides_[0] : 0;
    auto stride_input_row = input_strides_[ndim_ - 2];
    auto stride_out_batch = ndim_ == 3 ? out_strides_[0] : 0;
    auto stride_out_row = out_strides_[ndim_ - 2];

    dim3 grid(static_cast<unsigned>(seq_len_),
              static_cast<unsigned>(batch_size_));

    assert(out.dtype() == input.dtype());

    constexpr int kMaxBlockSize = BackendMaxBlockSize<Backend>::value;
    int block_size =
        std::min(RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(),
                 kMaxBlockSize);

    DispatchFunc<
        ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
        SupportedCudaBlockSizesType<BackendMaxBlockSize<Backend>::value>>(
        // TODO: Output dtype should use the one passed in during construction.
        {static_cast<int64_t>(out.dtype()), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          CausalSoftmaxKernel<kBlockSize, Backend::kDeviceType, T, float>
              <<<grid, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(input.data()), batch_size_,
                  seq_len_, total_seq_len_, stride_out_batch, stride_out_row,
                  stride_input_batch, stride_input_row);
        },
        "CudaCausalSoftmax::operator()");
  }
};

}  // namespace infini::ops

#endif
