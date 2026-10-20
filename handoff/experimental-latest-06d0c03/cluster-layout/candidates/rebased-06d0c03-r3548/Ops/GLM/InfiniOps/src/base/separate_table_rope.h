#ifndef INFINI_OPS_BASE_SEPARATE_TABLE_ROPE_H_
#define INFINI_OPS_BASE_SEPARATE_TABLE_ROPE_H_
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {
// Separate tables, full head rotation, distinct output: NOT packed
// RotaryEmbedding. bool is_neox=false is GPT-J, true is GPT-NeoX. No implicit
// algorithm fallback.
class SeparateTableRope : public Operator<SeparateTableRope> {
 public:
  static constexpr bool kAutomaticDefault = true;
  static bool supports(const Tensor input, const Tensor positions,
                       const Tensor sin_table, const Tensor cos_table,
                       bool /*is_neox*/, const Tensor out) {
    const auto rank = input.ndim();
    // Preserve the CPU table dtype domain. NVIDIA additionally retains LM's
    // independently typed tables and padded per-batch position rows.
    const bool nvidia = input.device().type() == Device::Type::kNvidia;
    if ((rank != 3 && rank != 4) || out.shape() != input.shape() ||
        out.dtype() != input.dtype() ||
        cos_table.dtype() != sin_table.dtype() ||
        (!nvidia && sin_table.dtype() != input.dtype()))
      return false;
    if (input.dtype() != DataType::kFloat16 &&
        input.dtype() != DataType::kBFloat16 &&
        input.dtype() != DataType::kFloat32 &&
        input.dtype() != DataType::kFloat64)
      return false;
    if (nvidia && (input.dtype() == DataType::kFloat64 ||
                   (sin_table.dtype() != DataType::kFloat16 &&
                    sin_table.dtype() != DataType::kBFloat16 &&
                    sin_table.dtype() != DataType::kFloat32)))
      return false;
    if (positions.dtype() != DataType::kInt8 &&
        positions.dtype() != DataType::kInt16 &&
        positions.dtype() != DataType::kInt32 &&
        positions.dtype() != DataType::kInt64 &&
        positions.dtype() != DataType::kUInt8 &&
        positions.dtype() != DataType::kUInt16 &&
        positions.dtype() != DataType::kUInt32 &&
        positions.dtype() != DataType::kUInt64)
      return false;
    if (nvidia && positions.dtype() != DataType::kInt32 &&
        positions.dtype() != DataType::kInt64)
      return false;
    if (input.device() != out.device() ||
        input.device() != positions.device() ||
        input.device() != sin_table.device() ||
        input.device() != cos_table.device())
      return false;
    const auto batch = rank == 4 ? input.size(0) : 1;
    const auto seq = input.size(rank - 3);
    if (positions.ndim() == 1) {
      if (positions.size(0) != seq || positions.stride(0) != 1) return false;
    } else if (positions.ndim() == 2) {
      if (positions.size(0) != batch || positions.size(1) != seq ||
          positions.stride(1) != 1 ||
          (!nvidia && positions.stride(0) != static_cast<Tensor::Stride>(seq)))
        return false;
    } else
      return false;
    if (sin_table.ndim() != 2 || cos_table.shape() != sin_table.shape() ||
        sin_table.size(1) > std::numeric_limits<std::size_t>::max() / 2 ||
        input.size(rank - 1) != 2 * sin_table.size(1) ||
        sin_table.stride(1) != 1 || cos_table.stride(1) != 1 ||
        sin_table.stride(0) != static_cast<Tensor::Stride>(sin_table.size(1)) ||
        cos_table.stride(0) != sin_table.stride(0) ||
        input.stride(rank - 1) != 1 || out.stride(rank - 1) != 1)
      return false;
    return true;
  }
  SeparateTableRope(const Tensor input, const Tensor positions,
                    const Tensor sin_table, const Tensor cos_table,
                    bool is_neox, Tensor out) {
    if (!supports(input, positions, sin_table, cos_table, is_neox, out))
      throw std::invalid_argument(
          "SeparateTableRope: unsupported metadata/alias");
  }
  virtual void operator()(const Tensor input, const Tensor positions,
                          const Tensor sin_table, const Tensor cos_table,
                          bool is_neox, Tensor out) const = 0;
};
}  // namespace infini::ops
#endif
