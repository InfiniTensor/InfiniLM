#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "native/cuda/caster.cuh"
namespace infini::ops {
// Read both pair members before stores: exact input/output alias is supported.
template <Device::Type Dev, class T, class P>
__device__ void MropeRotate(T* out, const T* in, const T* cos, const T* sin,
                            const P* positions, size_t heads, size_t head,
                            size_t rot, ptrdiff_t os, ptrdiff_t is,
                            ptrdiff_t cs, ptrdiff_t ss, ptrdiff_t pa,
                            ptrdiff_t pt, size_t table_len, size_t st,
                            size_t sh, size_t sw, bool axes, bool interleaved) {
  using A = std::conditional_t<std::is_same_v<T, double>, double, float>;
  const size_t token = blockIdx.x, half = rot / 2;
  for (size_t pair = threadIdx.x; pair < heads * half; pair += blockDim.x) {
    const size_t i = pair % half, h = pair / half;
    const size_t axis = interleaved ? (i % 3 == 1 && i < sh * 3   ? 1
                                       : i % 3 == 2 && i < sw * 3 ? 2
                                                                  : 0)
                                    : (i < st        ? 0
                                       : i < st + sh ? 1
                                                     : 2);
    const int64_t raw =
        static_cast<int64_t>(positions[token * pt + (axes ? axis * pa : 0)]);
    const size_t position = raw >= 0 && static_cast<size_t>(raw) < table_len
                                ? static_cast<size_t>(raw)
                                : 0;
    const A c = Caster<Dev>::template Cast<A>(cos[position * cs + i]);
    const A s = Caster<Dev>::template Cast<A>(sin[position * ss + i]);
    const A x = Caster<Dev>::template Cast<A>(in[token * is + h * head + i]);
    const A y =
        Caster<Dev>::template Cast<A>(in[token * is + h * head + i + half]);
    out[token * os + h * head + i] =
        Caster<Dev>::template Cast<T>(x * c - y * s);
    out[token * os + h * head + i + half] =
        Caster<Dev>::template Cast<T>(y * c + x * s);
  }
  for (size_t i = threadIdx.x; i < heads * (head - rot); i += blockDim.x) {
    const size_t offset = i / (head - rot) * head + rot + i % (head - rot);
    out[token * os + offset] = in[token * is + offset];
  }
}
template <Device::Type Dev, class T, class P>
__global__ void MropeKernel(T* qo, T* ko, const T* q, const T* k, const T* cos,
                            const T* sin, const P* p, size_t qheads,
                            size_t kheads, size_t head, size_t rot,
                            ptrdiff_t qos, ptrdiff_t kos, ptrdiff_t qs,
                            ptrdiff_t ks, ptrdiff_t cs, ptrdiff_t ss,
                            ptrdiff_t pa, ptrdiff_t pt, size_t table_len,
                            size_t st, size_t sh, size_t sw, bool axes,
                            bool interleaved) {
  MropeRotate<Dev, T, P>(qo, q, cos, sin, p, qheads, head, rot, qos, qs, cs, ss,
                         pa, pt, table_len, st, sh, sw, axes, interleaved);
  MropeRotate<Dev, T, P>(ko, k, cos, sin, p, kheads, head, rot, kos, ks, cs, ss,
                         pa, pt, table_len, st, sh, sw, axes, interleaved);
}
}  // namespace infini::ops
