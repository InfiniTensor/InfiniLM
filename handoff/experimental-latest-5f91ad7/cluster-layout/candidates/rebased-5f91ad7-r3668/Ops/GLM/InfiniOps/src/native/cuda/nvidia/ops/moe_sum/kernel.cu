#include <cassert>
#include <cstdint>

#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/ops/moe_sum/kernel.h"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/moe_sum/kernel.cuh"

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`MoeSum` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`MoeSum` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`MoeSum` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

}  // namespace

void Operator<MoeSum, Device::Type::kNvidia, 0>::operator()(
    const Tensor input, std::optional<Tensor> topk_ids,
    std::optional<Tensor> expert_map, Tensor output) const {
  ValidateCallMetadata(input, topk_ids, expert_map, output);

  if (num_tokens_ == 0 || hidden_size_ == 0) {
    return;
  }

  DeviceGuard device_guard{device_index_};
  auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
  constexpr unsigned int kBlockSize = 256;
  using DataTypes = ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;
  using IndexTypes = List<DataType::kInt32, DataType::kInt64>;

  DispatchFunc<DataTypes, IndexTypes>(
      {static_cast<int64_t>(dtype_), static_cast<int64_t>(topk_ids_dtype_)},
      [&](auto list_tag) {
        using Data = TypeMapType<Device::Type::kNvidia, ListGet<0>(list_tag)>;
        using Index = TypeMapType<Device::Type::kNvidia, ListGet<1>(list_tag)>;

        MoeSumKernel<Device::Type::kNvidia, Data, Index>
            <<<static_cast<unsigned int>(num_tokens_), kBlockSize, 0, stream>>>(
                reinterpret_cast<const Data*>(input.data()),
                topk_ids ? reinterpret_cast<const Index*>(topk_ids->data())
                         : nullptr,
                expert_map
                    ? reinterpret_cast<const int32_t*>(expert_map->data())
                    : nullptr,
                reinterpret_cast<Data*>(output.data()),
                static_cast<int64_t>(topk_), static_cast<int64_t>(hidden_size_),
                input_strides_[0], input_strides_[1], input_strides_[2],
                topk_ids_token_stride_, topk_ids_slot_stride_,
                static_cast<int64_t>(expert_map_size_), expert_map_stride_);
      },
      "Operator<MoeSum, Device::Type::kNvidia>::operator()");

  const auto status = cudaGetLastError();
  assert(status == cudaSuccess && "`MoeSum` CUDA kernel launch failed");
}

}  // namespace infini::ops
