#ifndef INFINI_OPS_CPU_SEPARATE_TABLE_ROPE_MATH_H_
#define INFINI_OPS_CPU_SEPARATE_TABLE_ROPE_MATH_H_
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
namespace infini::ops::separate_rope_cpu {
struct Layout {
  std::size_t batch, seq, heads, half, table_rows;
  std::size_t xb, xs, xh, yb, ys, yh;
  bool per_batch, neox;
};
// Preserve Core0.2.9 CPU output conversion, not RT's newer half RNE.
inline std::uint16_t legacy_f16(float value) {
  std::uint32_t bits; std::memcpy(&bits, &value, 4);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
  const int exponent = static_cast<int>((bits >> 23) & 255) - 127;
  auto mantissa = bits & 0x7fffff;
  if (exponent >= 16)
    return static_cast<std::uint16_t>(sign | ((exponent == 128 && mantissa) ? 0x7e00 : 0x7c00));
  if (exponent >= -14)
    return static_cast<std::uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13));
  if (exponent >= -24) {
    mantissa |= 0x800000; mantissa >>= (-14 - exponent);
    return static_cast<std::uint16_t>(sign | (mantissa >> 13));
  }
  return sign;
}
inline std::uint16_t legacy_bf16(float value) {
  std::uint32_t bits; std::memcpy(&bits, &value, 4);
  return static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
template<class Data, class Index, class Compute, class Load, class Store>
void compute(const Layout& m, Data* y, const Data* x, const Index* positions,
             const Data* sin_table, const Data* cos_table, Load load, Store store) {
  if (m.batch == 0 || m.seq == 0 || m.heads == 0 || m.half == 0) return;
  // Validate all position values before any store; old negative/OOB values were
  // unchecked memory access, not supported semantics. No partial mutation.
  for (std::size_t i = 0; i < (m.per_batch ? m.batch * m.seq : m.seq); ++i) {
    if constexpr (std::is_signed_v<Index>)
      if (positions[i] < 0) throw std::invalid_argument("SeparateTableRope: negative position");
    if (static_cast<std::uint64_t>(positions[i]) >= m.table_rows)
      throw std::invalid_argument("SeparateTableRope: position outside tables");
  }
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t b = 0; b < m.batch; ++b)
    for (std::size_t h = 0; h < m.heads; ++h)
      for (std::size_t t = 0; t < m.seq; ++t) {
        const auto xo = b * m.xb + t * m.xs + h * m.xh;
        const auto yo = b * m.yb + t * m.ys + h * m.yh;
        const auto table = static_cast<std::size_t>(positions[(m.per_batch ? b * m.seq : 0) + t]) * m.half;
        for (std::size_t i = 0; i < m.half; ++i) {
          const auto a = m.neox ? i : 2 * i, z = m.neox ? i + m.half : 2 * i + 1;
          // Both pair members and table entries read before either inplace write.
          const Compute x0 = load(x[xo + a]), x1 = load(x[xo + z]);
          const Compute s = load(sin_table[table + i]), c = load(cos_table[table + i]);
          y[yo + a] = store(x0 * c - x1 * s);
          y[yo + z] = store(x0 * s + x1 * c);
        }
      }
}
}  // namespace infini::ops::separate_rope_cpu
#endif
