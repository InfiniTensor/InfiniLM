#ifndef INFINI_OPS_CPU_SEPARATE_TABLE_ROPE_KERNEL_H_
#define INFINI_OPS_CPU_SEPARATE_TABLE_ROPE_KERNEL_H_
#include "base/separate_table_rope.h"
#include "math.h"
#include "native/cpu/caster_.h"
namespace infini::ops {
template <>
class Operator<SeparateTableRope, Device::Type::kCpu>
    : public SeparateTableRope {
 public:
  static bool supports(const Tensor x, const Tensor p, const Tensor s,
                       const Tensor c, bool neox, const Tensor y) {
    return x.device().type() == Device::Type::kCpu &&
           SeparateTableRope::supports(x, p, s, c, neox, y);
  }
  Operator(const Tensor x, const Tensor p, const Tensor s, const Tensor c,
           bool neox, Tensor y)
      : SeparateTableRope(x, p, s, c, neox, y) {
    if (!supports(x, p, s, c, neox, y))
      throw std::invalid_argument("SeparateTableRope: CPU only");
  }
  void operator()(const Tensor x, const Tensor p, const Tensor s,
                  const Tensor c, bool neox, Tensor y) const override {
    if (x.numel() == 0) return;
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        x.dtype(),
        [&](auto data_tag) {
          using Data = typename decltype(data_tag)::type;
          DispatchFunc<Device::Type::kCpu, AllIntTypes>(
              p.dtype(),
              [&](auto index_tag) {
                using Index = typename decltype(index_tag)::type;
                using Compute =
                    std::conditional_t<IsFP16<Device::Type::kCpu, Data> ||
                                           IsBFloat16<Device::Type::kCpu, Data>,
                                       float, Data>;
                const auto rank = x.ndim();
                const separate_rope_cpu::Layout m{
                    rank == 4 ? x.size(0) : 1,
                    x.size(rank - 3),
                    x.size(rank - 2),
                    x.size(rank - 1) / 2,
                    s.size(0),
                    rank == 4 ? static_cast<std::size_t>(x.stride(0)) : 0,
                    static_cast<std::size_t>(x.stride(rank - 3)),
                    static_cast<std::size_t>(x.stride(rank - 2)),
                    rank == 4 ? static_cast<std::size_t>(y.stride(0)) : 0,
                    static_cast<std::size_t>(y.stride(rank - 3)),
                    static_cast<std::size_t>(y.stride(rank - 2)),
                    p.ndim() == 2,
                    neox};
                auto load = [](Data v) -> Compute {
                  if constexpr (IsFP16<Device::Type::kCpu, Data> ||
                                IsBFloat16<Device::Type::kCpu, Data>)
                    return v.ToFloat();
                  else
                    return v;
                };
                auto store = [](Compute v) -> Data {
                  if constexpr (IsFP16<Device::Type::kCpu, Data>)
                    return Data{separate_rope_cpu::legacy_f16(v)};
                  else if constexpr (IsBFloat16<Device::Type::kCpu, Data>)
                    return Data{separate_rope_cpu::legacy_bf16(v)};
                  else
                    return v;
                };
                separate_rope_cpu::compute<Data, Index, Compute>(
                    m, static_cast<Data*>(y.data()),
                    static_cast<const Data*>(x.data()),
                    static_cast<const Index*>(p.data()),
                    static_cast<const Data*>(s.data()),
                    static_cast<const Data*>(c.data()), load, store);
              },
              "SeparateTableRope CPU positions");
        },
        "SeparateTableRope CPU data");
  }
  // workspace=0 and capture_safe=false inherited: CPU host path only.
};
}  // namespace infini::ops
#endif
