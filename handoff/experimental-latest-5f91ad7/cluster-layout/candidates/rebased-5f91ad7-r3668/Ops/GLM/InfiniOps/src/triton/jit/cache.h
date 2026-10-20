#ifndef INFINI_OPS_TRITON_JIT_CACHE_H_
#define INFINI_OPS_TRITON_JIT_CACHE_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "triton/jit/backend.h"
#include "triton/jit/config_.h"

namespace infini::ops::triton::jit {

namespace detail {

struct KernelArtifact {
  KernelMetadata metadata;

  std::string binary;
};

class KernelCacheKey {
 public:
  static KernelCacheKey Build(const Target& target, int device_id,
                              const std::string& compilation_fingerprint,
                              const std::string& operator_name,
                              const std::string& signature,
                              const Config& config);

  const std::string& identity() const { return identity_; }

  const std::string& memory_identity() const { return memory_identity_; }

  std::string ArtifactPrefix() const;

 private:
  KernelCacheKey(std::string identity, std::string memory_identity)
      : identity_(std::move(identity)),
        memory_identity_(std::move(memory_identity)) {}

  std::string identity_;

  std::string memory_identity_;
};

class AutoTuningCacheKey {
 public:
  static AutoTuningCacheKey Build(
      const Target& target, const std::string& compilation_fingerprint,
      const std::string& operator_name, const std::string& signature,
      const std::vector<std::string>& key_names,
      const std::vector<std::uint64_t>& key_values,
      const std::vector<Config>& candidates, const std::vector<Grid>& grids,
      int warmup_milliseconds, int repetition_milliseconds);

  const std::string& identity() const { return identity_; }

 private:
  explicit AutoTuningCacheKey(std::string identity)
      : identity_(std::move(identity)) {}

  std::string identity_;
};

std::optional<KernelArtifact> ReadKernelArtifact(
    const std::string& output_prefix, const std::string& expected_identity);

template <Device::Type kDev>
class KernelCache {
 public:
  using Entry = std::unique_ptr<Kernel<kDev>>;

  static KernelCache& Instance() {
    // Backend contexts may already be gone during static destruction.
    static auto* cache_ptr = new KernelCache;
    return *cache_ptr;
  }

  const Kernel<kDev>* Find(const KernelCacheKey& key) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(key.memory_identity());
    return it == entries_.end() ? nullptr : it->second.get();
  }

  const Kernel<kDev>* InsertOrGet(const KernelCacheKey& key, Entry candidate) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto result =
        entries_.try_emplace(key.memory_identity(), std::move(candidate));
    return result.first->second.get();
  }

 private:
  mutable std::mutex mutex_;

  // There is no erase path, so returned pointee addresses remain stable.
  std::unordered_map<std::string, Entry> entries_;
};

class AutoTuningCache {
 public:
  static AutoTuningCache& Instance();

  std::optional<Config> Find(const AutoTuningCacheKey& key);

  void Insert(const AutoTuningCacheKey& key, const Config& config);

 private:
  std::mutex mutex_;

  std::unordered_map<std::string, Config> entries_;
};

}  // namespace detail

}  // namespace infini::ops::triton::jit

#endif
