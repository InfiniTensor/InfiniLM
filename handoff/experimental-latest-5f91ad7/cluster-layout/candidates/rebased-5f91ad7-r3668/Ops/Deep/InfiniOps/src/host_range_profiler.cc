#include "host_range_profiler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace infini::ops {
namespace {

constexpr std::array<const char*,
                     static_cast<std::size_t>(HostRangeLayer::kCount)>
    kLayerNames{
        "binding.body",
        "binding.tensor_conversion",
        "binding.device_conversion",
        "dispatch.call",
        "operator.call",
        "cache.key",
        "cache.lookup",
        "cache.construct",
        "operator.invoke",
        "backend.submit",
        "calibration.depth1",
        "calibration.depth2",
        "calibration.depth3",
    };

std::size_t LayerIndex(HostRangeLayer layer) {
  const auto index = static_cast<std::size_t>(layer);
  if (index >= kLayerNames.size()) {
    throw std::runtime_error("invalid host range layer");
  }
  return index;
}

}  // namespace

const char* HostRangeLayerName(HostRangeLayer layer) {
  return kLayerNames[LayerIndex(layer)];
}

#if defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)

namespace {

using Clock = std::chrono::steady_clock;
using DurationNs = std::uint64_t;

constexpr std::size_t kInitialStackCapacity = 16;
constexpr std::size_t kInitialSampleCapacity = 256;

struct Frame {
  HostRangeLayer layer;
  Clock::time_point start;
  DurationNs child_duration{0};
};

struct LayerSamples {
  std::vector<DurationNs> inclusive;
  std::vector<DurationNs> self;
};

struct CollectorState {
  bool active{false};
  bool recording_failed{false};
  std::uint64_t session_id{0};
  std::vector<Frame> stack;
  std::array<LayerSamples, kLayerNames.size()> samples;
};

thread_local CollectorState collector_state;
thread_local std::uint64_t next_session_id{0};

double Mean(const std::vector<DurationNs>& samples) {
  long double total = 0;
  for (const auto sample : samples) {
    total += sample;
  }
  return static_cast<double>(total / samples.size());
}

double Median(std::vector<DurationNs> samples) {
  std::sort(samples.begin(), samples.end());
  const auto middle = samples.size() / 2;
  if (samples.size() % 2 != 0) {
    return static_cast<double>(samples[middle]);
  }
  return static_cast<double>(
      (static_cast<long double>(samples[middle - 1]) + samples[middle]) / 2.0L);
}

void ReserveCollectorStorage(CollectorState& state) {
  state.stack.reserve(kInitialStackCapacity);
  for (auto& samples : state.samples) {
    samples.inclusive.reserve(kInitialSampleCapacity);
    samples.self.reserve(kInitialSampleCapacity);
  }
}

}  // namespace

bool HostRangeProfiler::IsCompiled() { return true; }

void HostRangeProfiler::Start() {
  if (collector_state.active) {
    throw std::runtime_error("host range profiler is already active");
  }

  CollectorState state;
  ReserveCollectorStorage(state);
  if (next_session_id == std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error("host range profiler session IDs are exhausted");
  }
  state.session_id = ++next_session_id;
  state.active = true;
  collector_state = std::move(state);
}

std::vector<HostRangeSummary> HostRangeProfiler::Stop() {
  if (!collector_state.active) {
    throw std::runtime_error("host range profiler is not active");
  }

  CollectorState completed_state = std::move(collector_state);
  collector_state = CollectorState{};

  if (!completed_state.stack.empty()) {
    throw std::runtime_error("host range profiler has unclosed ranges");
  }
  if (completed_state.recording_failed) {
    throw std::runtime_error("host range profiler failed to record a sample");
  }

  std::vector<HostRangeSummary> summaries;
  summaries.reserve(kLayerNames.size());
  for (std::size_t index = 0; index < completed_state.samples.size(); ++index) {
    const auto& samples = completed_state.samples[index];
    if (samples.inclusive.empty()) {
      continue;
    }
    if (samples.inclusive.size() != samples.self.size()) {
      throw std::runtime_error("host range profiler sample counts differ");
    }

    summaries.push_back(HostRangeSummary{
        static_cast<HostRangeLayer>(index),
        samples.inclusive.size(),
        Mean(samples.inclusive),
        Median(samples.inclusive),
        Mean(samples.self),
        Median(samples.self),
    });
  }

  return summaries;
}

std::vector<HostRangeSummary> HostRangeProfiler::Calibrate(
    std::size_t iterations) {
  if (collector_state.active) {
    throw std::runtime_error(
        "cannot calibrate while the host range profiler is active");
  }
  if (iterations == 0) {
    throw std::runtime_error("host range calibration requires iterations");
  }

  Start();
  try {
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      HostRangeScope depth1{HostRangeLayer::kCalibrationDepth1};
      HostRangeScope depth2{HostRangeLayer::kCalibrationDepth2};
      HostRangeScope depth3{HostRangeLayer::kCalibrationDepth3};
    }
    return Stop();
  } catch (...) {
    collector_state = CollectorState{};
    throw;
  }
}

HostRangeScope::HostRangeScope(HostRangeLayer layer) {
  if (!collector_state.active) {
    return;
  }

  LayerIndex(layer);
  collector_state.stack.push_back(Frame{layer, Clock::now(), 0});
  owner_ = &collector_state;
  session_id_ = collector_state.session_id;
  layer_ = layer;
  depth_ = collector_state.stack.size();
  active_ = true;
}

HostRangeScope::~HostRangeScope() noexcept {
  if (!active_) {
    return;
  }
  if (owner_ != &collector_state) {
    return;
  }
  if (!collector_state.active || collector_state.session_id != session_id_) {
    return;
  }
  if (collector_state.stack.empty()) {
    collector_state.recording_failed = true;
    return;
  }
  if (collector_state.stack.size() != depth_ ||
      collector_state.stack.back().layer != layer_) {
    collector_state.recording_failed = true;
    return;
  }

  const auto end = Clock::now();
  const auto frame = collector_state.stack.back();
  collector_state.stack.pop_back();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - frame.start)
          .count();
  const auto inclusive =
      elapsed > 0 ? static_cast<DurationNs>(elapsed) : DurationNs{0};
  const auto self = inclusive >= frame.child_duration
                        ? inclusive - frame.child_duration
                        : DurationNs{0};

  if (!collector_state.stack.empty()) {
    auto& parent_child_duration = collector_state.stack.back().child_duration;
    if (inclusive >
        std::numeric_limits<DurationNs>::max() - parent_child_duration) {
      parent_child_duration = std::numeric_limits<DurationNs>::max();
    } else {
      parent_child_duration += inclusive;
    }
  }

  auto& samples =
      collector_state.samples[static_cast<std::size_t>(frame.layer)];
  try {
    samples.inclusive.push_back(inclusive);
    try {
      samples.self.push_back(self);
    } catch (...) {
      samples.inclusive.pop_back();
      throw;
    }
  } catch (...) {
    collector_state.recording_failed = true;
  }
}

#else

namespace {

[[noreturn]] void ThrowNotCompiled() {
  throw std::runtime_error("host range profiling is not compiled");
}

}  // namespace

bool HostRangeProfiler::IsCompiled() { return false; }

void HostRangeProfiler::Start() { ThrowNotCompiled(); }

std::vector<HostRangeSummary> HostRangeProfiler::Stop() { ThrowNotCompiled(); }

std::vector<HostRangeSummary> HostRangeProfiler::Calibrate(
    std::size_t iterations) {
  (void)iterations;
  ThrowNotCompiled();
}

#endif

}  // namespace infini::ops
