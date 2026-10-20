#ifndef INFINI_OPS_CUDA_SEPARATE_TABLE_ROPE_KERNEL_H_
#define INFINI_OPS_CUDA_SEPARATE_TABLE_ROPE_KERNEL_H_
#include "base/separate_table_rope.h"
#include "layout.h"
#include "native/cuda/ops/rotary_embedding_infinilm/kernel.h"

namespace infini::ops {
template <typename Backend>
class CudaSeparateTableRope : public SeparateTableRope {
 public:
  static bool supports(const Tensor x, const Tensor p, const Tensor s,
                       const Tensor c, bool neox, const Tensor y) {
    if (Backend::kDeviceType != Device::Type::kNvidia ||
        !SeparateTableRope::supports(x,p,s,c,neox,y)) return false;
    return separate_rope_nvidia::layout_supported(x,p,s,c,neox,y);
  }
  CudaSeparateTableRope(const Tensor x, const Tensor p, const Tensor s,
                       const Tensor c, bool neox, Tensor y)
      : SeparateTableRope(x,p,s,c,neox,y),
        block_size_{prepare_block_size()} {}
  void operator()(const Tensor x, const Tensor p, const Tensor s,
                  const Tensor c, bool neox, Tensor y) const override {
    // Positions must be in [0, s.size(0)).
    if (!separate_rope_nvidia::layout_supported(x,p,s,c,neox,y))
      throw std::invalid_argument("SeparateTableRope: NVIDIA unsupported rebind");
    if (x.numel() == 0) return;
    const auto rank = x.ndim();
    const bool batched = rank == 4;
    const bool batch_pos = p.ndim() == 2;
    const auto batch = batched ? x.size(0) : 1;
    dim3 grid(static_cast<unsigned>(x.size(rank-3)),
              static_cast<unsigned>(x.size(rank-2)),static_cast<unsigned>(batch));
    const auto stream = static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    using DataTypes = ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;
    using AngleTypes = ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;
    DispatchFunc<DataTypes, AngleTypes, SupportedCudaBlockSizesType<1024>>(
        {static_cast<int64_t>(x.dtype()),static_cast<int64_t>(s.dtype()),block_size_},
        [&](auto tag) {
          using Data = TypeMapType<Backend::kDeviceType,ListGet<0>(tag)>;
          using Angle = TypeMapType<Backend::kDeviceType,ListGet<1>(tag)>;
          constexpr int kBlock = ListGet<2>(tag);
          // Existing kernel true = interleaved pair (GPT-J), false = split
          // half (GPT-NeoX). Preserve Rs4's bool neox, not the old wrapper name.
          auto launch = [&](auto interleaved) {
            LaunchRopeKernel<decltype(interleaved)::value,Backend,Data,Angle>(
                grid,kBlock,stream,reinterpret_cast<Data*>(y.data()),
                reinterpret_cast<const Data*>(x.data()),p.data(),p.dtype(),
                reinterpret_cast<const Angle*>(s.data()),
                reinterpret_cast<const Angle*>(c.data()),s.size(1),
                batched ? y.stride(0) : 0,y.stride(rank-3),y.stride(rank-2),
                batched ? x.stride(0) : 0,x.stride(rank-3),x.stride(rank-2),
                batch_pos ? p.stride(0) : 0,batch_pos,batched);
          };
          if (neox) launch(std::false_type{});
          else launch(std::true_type{});
        },"SeparateTableRope NVIDIA");
  }
 private:
  static int prepare_block_size() {
    const int block = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();
    if (block != 128 && block != 256 && block != 512 && block != 1024)
      throw std::runtime_error("SeparateTableRope: unsupported NVIDIA block size");
    return block;
  }
  const int block_size_;
};
}  // namespace infini::ops
#endif
