#pragma once
#include "base/mrope.h"
#include "dispatcher.h"
#include "native/cuda/ops/mrope/kernel.cuh"
namespace infini::ops {
template<class Backend> class CudaMrope : public Mrope {
 public:
  using Mrope::Mrope;
  void operator()(const Tensor q, const Tensor k, const Tensor cos, const Tensor sin, const Tensor p,
                  int head, int rot, int st, int sh, int sw, bool interleaved, Tensor qo, Tensor ko) const override {
    if (q.size(0) == 0) return;
    auto stream = static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    DispatchFunc<ConcatType<List<DataType::kFloat32, DataType::kFloat64>, ReducedFloatTypes>, List<DataType::kInt32, DataType::kInt64>>(
      {static_cast<int64_t>(q.dtype()), static_cast<int64_t>(p.dtype())}, [&](auto tag) {
        using T = TypeMapType<Backend::kDeviceType, ListGet<0>(tag)>;
        using P = TypeMapType<Backend::kDeviceType, ListGet<1>(tag)>;
          MropeKernel<Backend::kDeviceType, T, P><<<static_cast<unsigned>(q.size(0)), 256, 0, stream>>>(
            reinterpret_cast<T*>(qo.data()), reinterpret_cast<T*>(ko.data()),
            reinterpret_cast<const T*>(q.data()), reinterpret_cast<const T*>(k.data()),
            reinterpret_cast<const T*>(cos.data()), reinterpret_cast<const T*>(sin.data()), reinterpret_cast<const P*>(p.data()),
            q.size(1)/head, k.size(1)/head, head, rot, qo.stride(0), ko.stride(0), q.stride(0), k.stride(0), cos.stride(0), sin.stride(0),
            p.ndim() == 2 ? p.stride(0) : 0, p.stride(-1), cos.size(0), st, sh, sw, p.ndim() == 2, interleaved);
      }, "CudaMrope::operator()");
  }
};
}
