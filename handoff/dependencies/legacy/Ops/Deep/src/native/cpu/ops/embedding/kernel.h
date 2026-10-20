#ifndef INFINI_OPS_CPU_EMBEDDING_KERNEL_H_
#define INFINI_OPS_CPU_EMBEDDING_KERNEL_H_

#include <cstring>
#include <cstdint>
#include <type_traits>
#include <vector>
#include "base/embedding.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kCpu> : public Embedding {
 public:
  Operator(const Tensor input, const Tensor weight,
           const std::optional<int64_t> padding_idx,
           const std::optional<double> max_norm, const double norm_type,
           const bool scale_grad_by_freq, const bool sparse, Tensor out)
      : Embedding(input, weight, padding_idx, max_norm, norm_type,
                  scale_grad_by_freq, sparse, out) {
    Validate(input, weight, out, max_norm);
  }
  Operator(const Tensor input, const Tensor weight, Tensor out)
      : Operator(input, weight, std::nullopt, std::nullopt, 2.0, false, false, out) {}
  Operator(const Tensor input, const Tensor weight, const int64_t padding_idx,
           const bool scale_grad_by_freq, const bool sparse, Tensor out)
      : Operator(input, weight, padding_idx, std::nullopt, 2.0,
                 scale_grad_by_freq, sparse, out) {}

  using Embedding::operator();

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<int64_t> padding_idx,
                  const std::optional<double> max_norm, const double norm_type,
                  const bool scale_grad_by_freq, const bool sparse,
                  Tensor out) const override {
    if (input.shape() != input_shape_ || input.dtype() != input_dtype_ ||
        weight.shape() != weight_shape_ || weight.dtype() != weight_dtype_ ||
        out.shape() != out_shape_ || out.dtype() != out_dtype_)
      throw std::invalid_argument("CPU Embedding call metadata differs from Make");
    Validate(input, weight, out, max_norm);
    if (out.numel() == 0) return;
    std::vector<std::size_t> rows(num_indices_);
    for (std::size_t i = 0; i < num_indices_; ++i) {
      const auto index = ReadIndex(input, i);
      if (index >= vocab_size_)
        throw std::out_of_range("CPU Embedding index exceeds weight rows");
      rows[i] = static_cast<std::size_t>(index);
    }
    const auto row_bytes = embedding_dim_ * weight.element_size();
    const auto* src = static_cast<const char*>(weight.data());
    auto* dst = static_cast<char*>(out.data());
    for (std::size_t i = 0; i < num_indices_; ++i)
      std::memcpy(dst + i * row_bytes, src + rows[i] * row_bytes, row_bytes);
    (void)padding_idx;
    (void)norm_type;
    (void)scale_grad_by_freq;
    (void)sparse;
  }

 private:
  static void Validate(const Tensor input, const Tensor weight, const Tensor out,
                       const std::optional<double> max_norm) {
    if (input.device().type() != Device::Type::kCpu || input.device() != weight.device() ||
        input.device() != out.device() || max_norm || !input.IsContiguous() ||
        !weight.IsContiguous() || !out.IsContiguous() ||
        (out.numel() && (!input.data() || !weight.data() || !out.data())))
      throw std::invalid_argument("CPU Embedding requires contiguous same-device plain lookup");
    const auto overlaps = [](const Tensor a, const Tensor b) {
      const auto left = reinterpret_cast<std::uintptr_t>(a.data());
      const auto right = reinterpret_cast<std::uintptr_t>(b.data());
      const auto a_bytes = a.numel() * a.element_size();
      const auto b_bytes = b.numel() * b.element_size();
      return a_bytes && b_bytes &&
             (left <= right ? right - left < a_bytes : left - right < b_bytes);
    };
    if (overlaps(input, out) || overlaps(weight, out))
      throw std::invalid_argument("CPU Embedding output overlaps an input");
  }
  template <typename T>
  static uint64_t LoadIndex(const Tensor input, std::size_t offset) {
    T value;
    std::memcpy(&value, static_cast<const char*>(input.data()) + offset * sizeof(T), sizeof(T));
    if constexpr (std::is_signed_v<T>)
      if (value < 0) throw std::out_of_range("CPU Embedding index is negative");
    return static_cast<uint64_t>(value);
  }
  static uint64_t ReadIndex(const Tensor input, std::size_t offset) {
    switch (input.dtype()) {
      case DataType::kInt32: return LoadIndex<int32_t>(input, offset);
      case DataType::kInt64: return LoadIndex<int64_t>(input, offset);
      case DataType::kUInt32: return LoadIndex<uint32_t>(input, offset);
      case DataType::kUInt64: return LoadIndex<uint64_t>(input, offset);
      default: throw std::invalid_argument("CPU Embedding indices must be 32/64-bit integers");
    }
  }
};

}  // namespace infini::ops
#endif
