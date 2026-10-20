#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infini::ops {

enum class HostRangeLayer {
  kBindingBody,
  kTensorConversion,
  kDeviceConversion,
  kDispatchCall,
  kOperatorCall,
  kCacheKey,
  kCacheLookup,
  kCacheConstruct,
  kOperatorInvoke,
  kBackendSubmit,
  kCalibrationDepth1,
  kCalibrationDepth2,
  kCalibrationDepth3,
  kCount,
};

const char* HostRangeLayerName(HostRangeLayer layer);

struct HostRangeSummary {
  HostRangeLayer layer;
  std::size_t count;
  double inclusive_mean;
  double inclusive_median;
  double self_mean;
  double self_median;
};

class HostRangeProfiler {
 public:
  static bool IsCompiled();
  static void Start();
  static std::vector<HostRangeSummary> Stop();
  static std::vector<HostRangeSummary> Calibrate(std::size_t iterations);
};

#if defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)

class HostRangeScope {
 public:
  explicit HostRangeScope(HostRangeLayer layer);
  ~HostRangeScope() noexcept;

  HostRangeScope(const HostRangeScope&) = delete;
  HostRangeScope& operator=(const HostRangeScope&) = delete;
  HostRangeScope(HostRangeScope&&) = delete;
  HostRangeScope& operator=(HostRangeScope&&) = delete;

 private:
  bool active_{false};
  const void* owner_{nullptr};
  std::uint64_t session_id_{0};
  HostRangeLayer layer_{HostRangeLayer::kCount};
  std::size_t depth_{0};
};

#else

class HostRangeScope {
 public:
  explicit constexpr HostRangeScope(HostRangeLayer) noexcept {}

  HostRangeScope(const HostRangeScope&) = delete;
  HostRangeScope& operator=(const HostRangeScope&) = delete;
  HostRangeScope(HostRangeScope&&) = delete;
  HostRangeScope& operator=(HostRangeScope&&) = delete;
};

#endif

}  // namespace infini::ops
