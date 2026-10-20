#ifndef INFINI_OPS_TUNING_H_
#define INFINI_OPS_TUNING_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "data_type.h"
#include "device.h"
#include "tensor.h"

namespace infini::ops {

namespace detail {

template <typename T>
void CombineTuningHash(std::size_t& hash, const T& value) {
  hash ^= std::hash<T>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

}  // namespace detail

struct TuningSignature {
  struct TensorSig {
    std::vector<int64_t> shape;
    DataType dtype;

    bool operator==(const TensorSig& other) const {
      return shape == other.shape && dtype == other.dtype;
    }
  };

  std::vector<TensorSig> tensors;
  std::vector<double> scalars;

  template <typename... Args>
  static TuningSignature Build(const Args&... args) {
    TuningSignature sig;
    (sig.Absorb(args), ...);
    return sig;
  }

  bool operator==(const TuningSignature& other) const {
    return tensors == other.tensors && scalars == other.scalars;
  }

  std::size_t Hash() const {
    std::size_t hash = 0;
    for (const auto& t : tensors) {
      for (auto dimension : t.shape) {
        detail::CombineTuningHash(hash, dimension);
      }
      detail::CombineTuningHash(hash, static_cast<int>(t.dtype));
    }
    for (auto s : scalars) {
      detail::CombineTuningHash(hash, s);
    }
    return hash;
  }

 private:
  void Absorb(const Tensor& t) {
    std::vector<int64_t> shape_vec;
    for (std::size_t i = 0; i < t.shape().size(); ++i) {
      shape_vec.push_back(static_cast<int64_t>(t.shape()[i]));
    }
    tensors.push_back({shape_vec, t.dtype()});
  }

  void Absorb(const std::optional<Tensor>& t) {
    if (t.has_value()) {
      Absorb(*t);
    }
  }

  void Absorb(const std::vector<Tensor>& ts) {
    for (const auto& t : ts) {
      Absorb(t);
    }
  }

  template <typename T>
  void Absorb(const T& v) {
    if constexpr (std::is_arithmetic_v<T>) {
      scalars.push_back(static_cast<double>(v));
    } else if constexpr (std::is_enum_v<T>) {
      scalars.push_back(static_cast<double>(static_cast<int64_t>(v)));
    }
  }

  template <typename T>
  void Absorb(const std::optional<T>& v) {
    if (v.has_value()) {
      Absorb(*v);
    }
  }
};

class TuningManager {
 public:
  static TuningManager& Instance();

  void InitializeFromEnvironment();

  void LoadTuningCache(const std::string& json_path);

  std::optional<std::size_t> Lookup(std::string_view operator_name,
                                    Device::Type device,
                                    const TuningSignature& signature) const;

  void Record(std::string_view operator_name, Device::Type device,
              const TuningSignature& signature, std::size_t best_index);

  bool IsEnabled() const { return enabled_; }

  int warmup_count() const { return warmup_count_; }

  int repeat_count() const { return repeat_count_; }

 private:
  TuningManager() = default;

  TuningManager(const TuningManager&) = delete;

  TuningManager& operator=(const TuningManager&) = delete;

  static constexpr int kDefaultWarmupCount = 1;

  static constexpr int kDefaultRepeatCount = 5;

  struct CacheKey {
    std::string operator_name;
    Device::Type device;
    TuningSignature signature;

    bool operator==(const CacheKey& other) const {
      return operator_name == other.operator_name && device == other.device &&
             signature == other.signature;
    }
  };

  struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const {
      std::size_t hash = 0;
      detail::CombineTuningHash(hash, key.operator_name);
      detail::CombineTuningHash(hash, static_cast<int>(key.device));
      detail::CombineTuningHash(hash, key.signature.Hash());
      return hash;
    }
  };

  void FlushToDiskLocked() const;

  std::unordered_map<CacheKey, std::size_t, CacheKeyHash> cache_;

  bool enabled_{false};

  std::string json_path_;

  int warmup_count_{kDefaultWarmupCount};

  int repeat_count_{kDefaultRepeatCount};

  mutable std::mutex mutex_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TUNING_H_
