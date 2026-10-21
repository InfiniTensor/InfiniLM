#pragma once

#include "../device.hpp"
#include "../memory.hpp"

#include "../graph/graph.hpp"

#include <infini/rt.h>

#include <cstdint>
#include <memory>

namespace infinicore {

namespace context {
void setDevice(Device device);
Device getDevice();
size_t getDeviceCount(Device::Type type);

infini::rt::runtime::Stream getStream();

void syncStream();
void syncDevice();
void trimMemory();

std::shared_ptr<Memory> allocateMemory(size_t size);
std::shared_ptr<Memory> allocateHostMemory(size_t size);
std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);

void memcpyH2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyD2H(void *dst, const void *src, size_t size);
void memcpyD2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyH2H(void *dst, const void *src, size_t size);

void setDeviceMemory(void *ptr, int value, size_t count);
void setDeviceMemoryAsync(void *ptr, int value, size_t count, infini::rt::runtime::Stream stream);

// Timing APIs for performance measurement
infini::rt::runtime::Event createEvent();
infini::rt::runtime::Event createEventWithFlags(uint32_t flags);
void recordEvent(infini::rt::runtime::Event event, infini::rt::runtime::Stream stream = nullptr);
bool queryEvent(infini::rt::runtime::Event event);
void synchronizeEvent(infini::rt::runtime::Event event);
void destroyEvent(infini::rt::runtime::Event event);
float elapsedTime(infini::rt::runtime::Event start, infini::rt::runtime::Event end);
void streamWaitEvent(infini::rt::runtime::Stream stream, infini::rt::runtime::Event event);

// Graph recording APIs
bool isGraphRecording();
void startGraphRecording();
void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);
std::shared_ptr<graph::Graph> stopGraphRecording();
void cancelGraphRecording() noexcept;

/// ``INFINI_CUDAGRAPH_POLICY``: empty (legacy), ``eager``, or ``full_and_piecewise``.
/// Unknown values are treated as empty / legacy.
const char *cudagraphPolicy();

/// True when device-graph segment capture/replay should run.
/// ``eager`` → false; ``full_and_piecewise`` → true unless
/// ``INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS`` is set; empty/legacy honors
/// that kill-switch when present, otherwise enables segments.
bool deviceGraphSegmentsEnabled();

/// Engine TLS inference phase for phase-scoped MoE / FA capture policy.
/// FA uses ``INFINI_FA_INGRAPH``; MoE uses ``INFINI_MOE_INGRAPH``.
enum class InferencePhase : uint8_t {
    Unknown = 0,
    Prefill = 1,
    Decode = 2,
};

void setInferencePhase(InferencePhase phase);
InferencePhase getInferencePhase();

/// FA may enter a device segment — phase allow-list via ``INFINI_FA_INGRAPH``
/// (``off`` | ``decode`` | ``both``; truthy ``1`` → ``both``).
/// Product ``full_and_piecewise`` defaults to ``decode`` when unset
/// (FULL decode segs≈1; prefill FA stays host-break).
/// Legacy ``INFINI_FA_FORCE_CAPTURE`` → ``both``. Kill-switch: ``0`` / ``off``.
bool faInGraphAllowed();

/// Triton MoE under stream capture — phase allow-list via ``INFINI_MOE_INGRAPH``
/// (``off`` | ``decode`` | ``prefill`` | ``both``; default ``off``).
/// ``INFINI_MOE_FORCE_HOST_BREAK=1`` → always host-break.
/// ``eager`` policy → host-break.
bool moeTritonCaptureAllowed();

/// RAII restore of ``InferencePhase`` for graph capture / forward scopes.
class InferencePhaseGuard {
public:
    explicit InferencePhaseGuard(InferencePhase phase)
        : prev_(getInferencePhase()) {
        setInferencePhase(phase);
    }
    ~InferencePhaseGuard() { setInferencePhase(prev_); }
    InferencePhaseGuard(const InferencePhaseGuard &) = delete;
    InferencePhaseGuard &operator=(const InferencePhaseGuard &) = delete;

private:
    InferencePhase prev_;
};

} // namespace context

} // namespace infinicore
