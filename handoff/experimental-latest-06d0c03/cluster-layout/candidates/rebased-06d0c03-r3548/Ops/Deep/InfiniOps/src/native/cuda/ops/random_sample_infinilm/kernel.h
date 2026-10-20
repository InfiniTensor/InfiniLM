#ifndef INFINI_OPS_CUDA_RANDOM_SAMPLE_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_RANDOM_SAMPLE_INFINILM_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "base/random_sample_infinilm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/random_sample_infinilm/kernel.cuh"
#include "native/cuda/runtime_.h"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaRandomSampleInfinilm : public RandomSampleInfinilm {
 public:
  CudaRandomSampleInfinilm(const Tensor logits, float random_val, float topp,
                           int64_t topk, float temperature, Tensor out)
      : RandomSampleInfinilm{logits, random_val, topp, topk, temperature, out},
        workspace_size_{DispatchWorkspaceSize(out_dtype_, dtype_, n_)} {
    if (workspace_size_ > 0) {
      auto err = Backend::Malloc(&default_workspace_, workspace_size_);
      assert(err == 0 &&
             "`CudaRandomSampleInfinilm` failed to allocate workspace");
    }
  }

  ~CudaRandomSampleInfinilm() override {
    if (default_workspace_ != nullptr) {
      auto err = Backend::Free(default_workspace_);
      assert(err == 0 && "`CudaRandomSampleInfinilm` failed to free workspace");
    }
  }

  std::size_t workspace_size_in_bytes() const override {
    return workspace_size_;
  }

  void operator()(const Tensor logits, float random_val, float topp,
                  int64_t topk, float temperature, Tensor out) const override {
    assert(logits.dtype() == dtype_ && out.dtype() == out_dtype_);
    assert(logits.size(0) == n_ && logits.stride(0) == logits_stride_);
    assert(topk > 0 && "`RandomSampleInfinilm` requires `topk > 0`");

    void* workspace = workspace_ ? workspace_ : default_workspace_;
    auto workspace_size =
        workspace_ ? workspace_size_in_bytes_ : workspace_size_;
    assert(workspace != nullptr && workspace_size >= workspace_size_);

    int block_size =
        std::min(RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(),
                 BackendMaxBlockSize<Backend>::value);

    DispatchFunc<
        AllIntTypes, ConcatType<FloatTypes, ReducedFloatTypes>,
        SupportedCudaBlockSizesType<BackendMaxBlockSize<Backend>::value>>(
        {static_cast<int64_t>(out.dtype()),
         static_cast<int64_t>(logits.dtype()), block_size},
        [&](auto list_tag) {
          using Tidx = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using Tval = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          constexpr int kBlockSize = ListGet<2>(list_tag);

          auto* result = reinterpret_cast<Tidx*>(out.data());
          auto* input = reinterpret_cast<const Tval*>(logits.data());

          if (random_val == 0.0f || topp == 0.0f || topk == 1 ||
              temperature == 0.0f) {
            RandomSampleInfinilmArgmax<kBlockSize, Backend::kDeviceType>(
                workspace, workspace_size, result, input, n_, stream_);
          } else {
            RandomSampleInfinilmTopP<kBlockSize, Backend::kDeviceType>(
                workspace, workspace_size, result, input, n_, random_val, topp,
                static_cast<int>(topk), temperature, stream_);
          }
        },
        "CudaRandomSampleInfinilm::operator()");
  }

 private:
  static std::size_t DispatchWorkspaceSize(DataType out_dtype, DataType dtype,
                                           std::size_t n) {
    std::size_t workspace_size = 0;

    DispatchFunc<AllIntTypes, ConcatType<FloatTypes, ReducedFloatTypes>>(
        {static_cast<int64_t>(out_dtype), static_cast<int64_t>(dtype)},
        [&](auto list_tag) {
          using Tidx = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using Tval = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          workspace_size =
              random_sample_infinilm_detail::WorkspaceSize<Backend::kDeviceType,
                                                           Tidx, Tval>(n);
        },
        "CudaRandomSampleInfinilm::DispatchWorkspaceSize");

    return workspace_size;
  }

  std::size_t workspace_size_{0};

  void* default_workspace_{nullptr};
};

}  // namespace infini::ops

#endif
