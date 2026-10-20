#ifndef INFINI_OPS_CUDA_MAMBA_SELECTIVE_SCAN_KERNEL_H_
#define INFINI_OPS_CUDA_MAMBA_SELECTIVE_SCAN_KERNEL_H_
#include <cmath>
#include <algorithm>
#include <type_traits>
#include "base/mamba_selective_scan.h"
#include "caster.h"
#include "dispatcher.h"

namespace infini::ops {
namespace mamba_scan_detail {
template<bool Contiguous>
__device__ __forceinline__ std::size_t Offset(const MambaScanLayout& p, int i,
    std::size_t batch, std::size_t t, std::size_t ch) {
  if constexpr (Contiguous) {
    if (i == 5) return batch * p.states + t;
    if (i == 6 || i == 8) return batch;
    if (i == 9) return (batch * p.channels + t) * p.states + ch;
    return (batch * p.length + t) * ((i == 3 || i == 4) ? p.states : p.channels) + ch;
  }
  return batch * p.strides[i][0] + t * p.strides[i][1] + ch * p.strides[i][2];
}
template<Device::Type Dev, typename T, typename P, bool Contiguous>
__global__ void Scan(T* out, const T* x, const T* dt, const T* b, const T* c,
    const P* a_log, const P* d, const T* gate, const T* bias, float* state,
    MambaScanLayout p, std::size_t first) {
  const std::size_t linear = first + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= p.batch * p.channels) return;
  const auto ch = linear % p.channels, batch = linear / p.channels;
  for (std::size_t t = 0; t < p.length; ++t) {
    const float xt = Caster<Dev>::template Cast<float>(x[Offset<Contiguous>(p, 1, batch, t, ch)]);
    const float z = Caster<Dev>::template Cast<float>(dt[Offset<Contiguous>(p, 2, batch, t, ch)]) +
                    Caster<Dev>::template Cast<float>(bias[Offset<Contiguous>(p, 8, ch, 0, 0)]);
    const float dtv = z > 20.0f ? z : log1pf(expf(z));
    float y = 0.0f;
    for (std::size_t n = 0; n < p.states; ++n) {
      const float a = -expf(Caster<Dev>::template Cast<float>(a_log[Offset<Contiguous>(p, 5, ch, n, 0)]));
      const float discrete_a = expf(a * dtv);
      const float bn = Caster<Dev>::template Cast<float>(b[Offset<Contiguous>(p, 3, batch, t, n)]);
      const float cn = Caster<Dev>::template Cast<float>(c[Offset<Contiguous>(p, 4, batch, t, n)]);
      const auto si = Offset<Contiguous>(p, 9, batch, ch, n);
      const float s = discrete_a * state[si] + dtv * bn * xt;
      state[si] = s;
      y += s * cn;
    }
    y += xt * Caster<Dev>::template Cast<float>(d[Offset<Contiguous>(p, 6, ch, 0, 0)]);
    const float g = Caster<Dev>::template Cast<float>(gate[Offset<Contiguous>(p, 7, batch, t, ch)]);
    y *= g / (1.0f + expf(-g));
    out[Offset<Contiguous>(p, 0, batch, t, ch)] = Caster<Dev>::template Cast<T>(y);
  }
}
} // namespace mamba_scan_detail

template<typename Backend>
class CudaMambaSelectiveScan : public MambaSelectiveScan {
 public:
  static bool supports(Tensor out, const Tensor x, const Tensor dt, const Tensor b,
      const Tensor c, const Tensor a_log, const Tensor d, const Tensor gate,
      const Tensor dt_bias, Tensor state) {
    try { ValidateMetadata({out,x,dt,b,c,a_log,d,gate,dt_bias,state}); return true; }
    catch (const std::invalid_argument&) { return false; }
    catch (const std::overflow_error&) { return false; }
  }
  using MambaSelectiveScan::MambaSelectiveScan;
  void operator()(Tensor out, const Tensor x, const Tensor dt, const Tensor b,
      const Tensor c, const Tensor a_log, const Tensor d, const Tensor gate,
      const Tensor dt_bias, Tensor state) const override {
    if (!layout_.batch || !layout_.length || !layout_.channels) return;
    DispatchFunc<List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
      dtype_, [&](auto tag) {
        using T = TypeMapType<Backend::kDeviceType, decltype(tag)::value>;
        auto launch = [&](auto* parameter_type) {
          using P = std::remove_pointer_t<decltype(parameter_type)>;
          const auto stream = static_cast<typename Backend::Stream>(stream_);
          constexpr std::size_t cap = 65535u * 256u;
          const auto total = layout_.batch * layout_.channels;
          for (std::size_t first = 0; first < total;) {
            const auto count = std::min(cap, total - first);
            const dim3 blocks(static_cast<unsigned>((count + 255) / 256));
            if (layout_.contiguous)
              mamba_scan_detail::Scan<Backend::kDeviceType,T,P,true><<<blocks,256,0,stream>>>(
                static_cast<T*>(out.data()), static_cast<const T*>(x.data()),
                static_cast<const T*>(dt.data()), static_cast<const T*>(b.data()),
                static_cast<const T*>(c.data()), static_cast<const P*>(a_log.data()),
                static_cast<const P*>(d.data()), static_cast<const T*>(gate.data()),
                static_cast<const T*>(dt_bias.data()), static_cast<float*>(state.data()), layout_,first);
            else
              mamba_scan_detail::Scan<Backend::kDeviceType,T,P,false><<<blocks,256,0,stream>>>(
                static_cast<T*>(out.data()), static_cast<const T*>(x.data()),
                static_cast<const T*>(dt.data()), static_cast<const T*>(b.data()),
                static_cast<const T*>(c.data()), static_cast<const P*>(a_log.data()),
                static_cast<const P*>(d.data()), static_cast<const T*>(gate.data()),
                static_cast<const T*>(dt_bias.data()), static_cast<float*>(state.data()), layout_,first);
            Backend::CheckLaunch();
            first += count;
          }
        };
        if (a_log.dtype() == DataType::kFloat32) launch(static_cast<float*>(nullptr));
        else launch(static_cast<T*>(nullptr));
      }, "CudaMambaSelectiveScan");
  }
};
} // namespace infini::ops
#endif
