#include "context_impl.hpp"
#include "internal.hpp"

#include "../utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinicore {
namespace {

constexpr std::array<Device::Type, static_cast<size_t>(Device::Type::kCount)> kDefaultDevicePriority{
    Device::Type::kNvidia,
    Device::Type::kCambricon,
    Device::Type::kAscend,
    Device::Type::kMetax,
    Device::Type::kMoore,
    Device::Type::kIluvatar,
    Device::Type::kHygon,
    Device::Type::kThead,
    Device::Type::kQy,
    Device::Type::kCpu,
};

void warn_context_cleanup_failure(const char *operation, const char *detail) noexcept {
    try {
        spdlog::warn("{} failed during context cleanup: {}", operation, detail);
    } catch (...) {
    }
}

} // namespace

thread_local std::shared_ptr<Runtime> ContextImpl::current_runtime_;
thread_local std::shared_ptr<Runtime> ContextImpl::graph_runtime_;

std::shared_ptr<Runtime> ContextImpl::getOrCreateRuntimeLocked(Device device, const std::thread::id &thread_id) {
    const auto type_index = static_cast<size_t>(device.type());
    const auto device_index = static_cast<size_t>(device.index());
    INFINICORE_ASSERT(type_index < runtime_table_.size());
    INFINICORE_ASSERT(device_index < runtime_table_[type_index].size());

    auto &thread_runtimes = runtime_table_[type_index][device_index];
    if (const auto found = thread_runtimes.find(thread_id); found != thread_runtimes.end()) {
        if (auto runtime = found->second.lock()) {
            return runtime;
        }
        thread_runtimes.erase(found);
    }

    std::shared_ptr<Runtime> runtime;
    try {
        runtime = std::shared_ptr<Runtime>(new Runtime(device));
    } catch (...) {
        const auto original_error = std::current_exception();
        if (current_runtime_ != nullptr) {
            try {
                current_runtime_->activate();
            } catch (const std::exception &error) {
                warn_context_cleanup_failure(
                    "restoring the current runtime after construction failure",
                    error.what());
            } catch (...) {
                warn_context_cleanup_failure(
                    "restoring the current runtime after construction failure",
                    "unknown error");
            }
        }
        std::rethrow_exception(original_error);
    }
    thread_runtimes.emplace(thread_id, runtime);
    return runtime;
}

template <Device::Type device_type>
void ContextImpl::initializeDeviceType() {
    if constexpr (infini::rt::DeviceEnabled<device_type>::value) {
        int device_count = 0;
        infini::rt::set_runtime_device_type(device_type);
        INFINICORE_CHECK_ERROR(infini::rt::runtime::GetDeviceCount(&device_count));
        INFINICORE_ASSERT(device_count >= 0);
        runtime_table_[static_cast<size_t>(device_type)].resize(static_cast<size_t>(device_count));
    }
}

Runtime *ContextImpl::getCurrentRuntime() {
    if (current_runtime_ != nullptr) {
        spdlog::debug("getCurrentRuntime() returning {} (ptr={})", current_runtime_->device().ToString(), static_cast<void *>(current_runtime_.get()));
        return current_runtime_.get();
    }

    spdlog::debug("current_runtime_ is null, performing lazy initialization");
    const auto thread_id = std::this_thread::get_id();
    {
        std::lock_guard<std::mutex> lock{runtime_table_mutex_};
        for (const auto device_type : kDefaultDevicePriority) {
            const auto type_index = static_cast<size_t>(device_type);
            if (!runtime_table_[type_index].empty()) {
                current_runtime_ = getOrCreateRuntimeLocked(
                    Device{device_type, 0}, thread_id);
                break;
            }
        }
    }

    INFINICORE_ASSERT(current_runtime_ != nullptr);
    current_runtime_->activate();
    spdlog::debug("Lazy init: Set current_runtime_ to {} (ptr={})", current_runtime_->device().ToString(), static_cast<void *>(current_runtime_.get()));
    return current_runtime_.get();
}

void ContextImpl::setDevice(Device device) {
    if (device == getCurrentRuntime()->device()) {
        current_runtime_->activate();
        return;
    }

    std::shared_ptr<Runtime> runtime;
    {
        std::lock_guard<std::mutex> lock{runtime_table_mutex_};
        runtime = getOrCreateRuntimeLocked(device, std::this_thread::get_id());
    }
    current_runtime_ = std::move(runtime);
    current_runtime_->activate();
}

size_t ContextImpl::getDeviceCount(Device::Type type) {
    const auto type_index = static_cast<size_t>(type);
    if (type_index >= runtime_table_.size()) {
        throw std::invalid_argument("invalid device type");
    }
    std::lock_guard<std::mutex> lock{runtime_table_mutex_};
    return runtime_table_[type_index].size();
}

bool ContextImpl::isGraphRecording() {
    getCurrentRuntime();
    return graph_runtime_ != nullptr
        && graph_runtime_ == current_runtime_
        && graph_runtime_->isGraphRecording();
}

void ContextImpl::startGraphRecording() {
    getCurrentRuntime();
    if (graph_runtime_ != nullptr) {
        throw std::runtime_error("graph recording is already active on this thread");
    }
    current_runtime_->startGraphRecording();
    graph_runtime_ = current_runtime_;
}

void ContextImpl::addGraphOperator(std::shared_ptr<graph::GraphOperator> op) {
    getCurrentRuntime();
    if (graph_runtime_ == nullptr || graph_runtime_ != current_runtime_) {
        throw std::runtime_error("cannot record a graph operator on a non-capture device");
    }
    graph_runtime_->addGraphOperator(std::move(op));
}

std::shared_ptr<graph::Graph> ContextImpl::stopGraphRecording() {
    getCurrentRuntime();
    if (graph_runtime_ == nullptr) {
        return current_runtime_->stopGraphRecording();
    }

    auto owner = graph_runtime_;
    auto previous = current_runtime_;
    current_runtime_ = owner;
    bool stop_attempted = false;
    try {
        current_runtime_->activate();
        stop_attempted = true;
        auto graph = owner->stopGraphRecording();
        graph_runtime_.reset();
        current_runtime_ = previous;
        current_runtime_->activate();
        return graph;
    } catch (...) {
        const auto original_error = std::current_exception();
        if (stop_attempted) {
            graph_runtime_.reset();
        }
        current_runtime_ = previous;
        try {
            current_runtime_->activate();
        } catch (const std::exception &error) {
            warn_context_cleanup_failure("restoring the previous runtime", error.what());
        } catch (...) {
            warn_context_cleanup_failure("restoring the previous runtime", "unknown error");
        }
        std::rethrow_exception(original_error);
    }
}

void ContextImpl::cancelGraphRecording() noexcept {
    auto owner = std::exchange(graph_runtime_, nullptr);
    if (owner == nullptr) {
        return;
    }

    auto previous = current_runtime_;
    current_runtime_ = owner;
    try {
        owner->activate();
    } catch (const std::exception &error) {
        warn_context_cleanup_failure("activating the graph runtime", error.what());
    } catch (...) {
        warn_context_cleanup_failure("activating the graph runtime", "unknown error");
    }
    owner->cancelGraphRecording();

    current_runtime_ = previous;
    if (current_runtime_ != nullptr) {
        try {
            current_runtime_->activate();
        } catch (const std::exception &error) {
            warn_context_cleanup_failure("restoring the previous runtime", error.what());
        } catch (...) {
            warn_context_cleanup_failure("restoring the previous runtime", "unknown error");
        }
    }
}

ContextImpl &ContextImpl::singleton() {
    static ContextImpl instance;
    return instance;
}

ContextImpl::ContextImpl() {
    initializeDeviceType<Device::Type::kCpu>();
    initializeDeviceType<Device::Type::kNvidia>();
    initializeDeviceType<Device::Type::kCambricon>();
    initializeDeviceType<Device::Type::kAscend>();
    initializeDeviceType<Device::Type::kMetax>();
    initializeDeviceType<Device::Type::kMoore>();
    initializeDeviceType<Device::Type::kIluvatar>();
    initializeDeviceType<Device::Type::kHygon>();
    initializeDeviceType<Device::Type::kThead>();
    initializeDeviceType<Device::Type::kQy>();

    const bool has_runtime = std::any_of(
        runtime_table_.begin(), runtime_table_.end(),
        [](const auto &devices) { return !devices.empty(); });
    INFINICORE_ASSERT(has_runtime);
}

namespace context {

void setDevice(Device device) {
    ContextImpl::singleton().setDevice(device);
}

Device getDevice() {
    return ContextImpl::singleton().getCurrentRuntime()->device();
}

size_t getDeviceCount(Device::Type type) {
    return ContextImpl::singleton().getDeviceCount(type);
}

infini::rt::runtime::Stream getStream() {
    return ContextImpl::singleton().getCurrentRuntime()->stream();
}

void syncStream() {
    return ContextImpl::singleton().getCurrentRuntime()->syncStream();
}

void syncDevice() {
    return ContextImpl::singleton().getCurrentRuntime()->syncDevice();
}

void trimMemory() {
    return ContextImpl::singleton().getCurrentRuntime()->trimMemory();
}

std::shared_ptr<Memory> allocateMemory(size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->allocateMemory(size);
}

std::shared_ptr<Memory> allocateHostMemory(size_t size) {
    setDevice(Device{Device::Type::kCpu});
    return allocateMemory(size);
}

std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->allocatePinnedHostMemory(size);
}

void memcpyH2D(void *dst, const void *src, size_t size, bool async) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyH2D(dst, src, size, async);
}

void memcpyD2H(void *dst, const void *src, size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyD2H(dst, src, size);
}

void memcpyD2D(void *dst, const void *src, size_t size, bool async) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyD2D(dst, src, size, async);
}

void memcpyH2H(void *dst, const void *src, size_t size) {
    setDevice(Device{Device::Type::kCpu});
    return ContextImpl::singleton().getCurrentRuntime()->memcpyD2D(dst, src, size);
}

void setDeviceMemory(void *ptr, int value, size_t count) {
    return ContextImpl::singleton().getCurrentRuntime()->setDeviceMemory(ptr, value, count);
}

void setDeviceMemoryAsync(void *ptr, int value, size_t count, infini::rt::runtime::Stream stream) {
    return ContextImpl::singleton().getCurrentRuntime()->setDeviceMemoryAsync(ptr, value, count, stream);
}

// Timing API implementations
infini::rt::runtime::Event createEvent() {
    return ContextImpl::singleton().getCurrentRuntime()->createEvent();
}

infini::rt::runtime::Event createEventWithFlags(uint32_t flags) {
    return ContextImpl::singleton().getCurrentRuntime()->createEventWithFlags(flags);
}

void recordEvent(infini::rt::runtime::Event event, infini::rt::runtime::Stream stream) {
    ContextImpl::singleton().getCurrentRuntime()->recordEvent(event, stream);
}

bool queryEvent(infini::rt::runtime::Event event) {
    return ContextImpl::singleton().getCurrentRuntime()->queryEvent(event);
}

void synchronizeEvent(infini::rt::runtime::Event event) {
    ContextImpl::singleton().getCurrentRuntime()->synchronizeEvent(event);
}

void destroyEvent(infini::rt::runtime::Event event) {
    ContextImpl::singleton().getCurrentRuntime()->destroyEvent(event);
}

float elapsedTime(infini::rt::runtime::Event start, infini::rt::runtime::Event end) {
    return ContextImpl::singleton().getCurrentRuntime()->elapsedTime(start, end);
}

void streamWaitEvent(infini::rt::runtime::Stream stream, infini::rt::runtime::Event event) {
    ContextImpl::singleton().getCurrentRuntime()->streamWaitEvent(stream, event);
}

bool isGraphRecording() {
    return ContextImpl::singleton().isGraphRecording();
}

void startGraphRecording() {
    ContextImpl::singleton().startGraphRecording();
}

void addGraphOperator(std::shared_ptr<graph::GraphOperator> op) {
    ContextImpl::singleton().addGraphOperator(std::move(op));
}

std::shared_ptr<graph::Graph> stopGraphRecording() {
    return ContextImpl::singleton().stopGraphRecording();
}

void cancelGraphRecording() noexcept {
    ContextImpl::singleton().cancelGraphRecording();
}

const char *cudagraphPolicy() {
    const char *v = std::getenv("INFINI_CUDAGRAPH_POLICY");
    if (v == nullptr || v[0] == '\0') {
        return "";
    }
    const std::string s(v);
    if (s == "eager") {
        return "eager";
    }
    if (s == "full_and_piecewise") {
        return "full_and_piecewise";
    }
    // Reject unknown values — fall back to legacy (empty).
    return "";
}

bool deviceGraphSegmentsEnabled() {
    // Kill-switch always wins (diagnostic escape hatch).
    if (std::getenv("INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS") != nullptr) {
        return false;
    }
    const char *policy = cudagraphPolicy();
    if (std::strcmp(policy, "eager") == 0) {
        return false;
    }
    // full_and_piecewise and empty/legacy → enable segmented device graphs.
    return true;
}

namespace {
thread_local InferencePhase g_inference_phase = InferencePhase::Unknown;

bool env_truthy_(const char *name) {
    const char *v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::string(v) != "0"
        && std::strcmp(v, "false") != 0 && std::strcmp(v, "no") != 0
        && std::strcmp(v, "off") != 0;
}

void warn_once_(bool &flag, const char *msg) {
    if (flag) {
        return;
    }
    flag = true;
    spdlog::warn("{}", msg);
}

enum class MoeIngraphMode : uint8_t { Off = 0, Decode, Prefill, Both };

MoeIngraphMode moeIngraphMode_() {
    const char *raw = std::getenv("INFINI_MOE_INGRAPH");
    if (raw != nullptr && raw[0] != '\0') {
        std::string s(raw);
        for (char &c : s) {
            if (c >= 'A' && c <= 'Z') {
                c = static_cast<char>(c - 'A' + 'a');
            }
        }
        if (s == "off" || s == "0" || s == "false" || s == "no") {
            return MoeIngraphMode::Off;
        }
        if (s == "decode") {
            return MoeIngraphMode::Decode;
        }
        if (s == "prefill") {
            return MoeIngraphMode::Prefill;
        }
        if (s == "both") {
            return MoeIngraphMode::Both;
        }
        static bool unknown_warned = false;
        warn_once_(unknown_warned,
                   "INFINI_MOE_INGRAPH has unknown value (expected "
                   "off|decode|prefill|both); treating as off");
        return MoeIngraphMode::Off;
    }
    if (env_truthy_("INFINI_MOE_FORCE_CAPTURE")
        || env_truthy_("INFINI_MOE_TRITON_CAPTURE")) {
        static bool legacy_warned = false;
        warn_once_(
            legacy_warned,
            "INFINI_MOE_FORCE_CAPTURE / INFINI_MOE_TRITON_CAPTURE are deprecated; "
            "shim to INFINI_MOE_INGRAPH=decode");
        return MoeIngraphMode::Decode;
    }
    return MoeIngraphMode::Off;
}

enum class FaIngraphMode : uint8_t { Off = 0, Decode, Both };

FaIngraphMode faIngraphMode_() {
    const char *raw = std::getenv("INFINI_FA_INGRAPH");
    if (raw != nullptr && raw[0] != '\0') {
        std::string s(raw);
        for (char &c : s) {
            if (c >= 'A' && c <= 'Z') {
                c = static_cast<char>(c - 'A' + 'a');
            }
        }
        if (s == "off" || s == "0" || s == "false" || s == "no") {
            return FaIngraphMode::Off;
        }
        if (s == "decode") {
            return FaIngraphMode::Decode;
        }
        if (s == "both" || s == "1" || s == "true" || s == "yes" || s == "on") {
            return FaIngraphMode::Both;
        }
        static bool unknown_warned = false;
        warn_once_(unknown_warned,
                   "INFINI_FA_INGRAPH has unknown value (expected "
                   "off|decode|both); treating as off");
        return FaIngraphMode::Off;
    }
    if (env_truthy_("INFINI_FA_FORCE_CAPTURE")) {
        static bool warned = false;
        warn_once_(
            warned,
            "INFINI_FA_FORCE_CAPTURE is deprecated; use "
            "INFINI_FA_INGRAPH=decode (product) or both/1 (Prefill+Decode)");
        return FaIngraphMode::Both;
    }
    // Product full_and_piecewise defaults to decode (FULL segs≈1; prefill FA HB).
    if (std::strcmp(cudagraphPolicy(), "full_and_piecewise") == 0) {
        return FaIngraphMode::Decode;
    }
    return FaIngraphMode::Off;
}
} // namespace

void setInferencePhase(InferencePhase phase) {
    g_inference_phase = phase;
}

InferencePhase getInferencePhase() {
    return g_inference_phase;
}

bool faInGraphAllowed() {
    const FaIngraphMode mode = faIngraphMode_();
    const InferencePhase phase = getInferencePhase();
    switch (mode) {
    case FaIngraphMode::Off:
        return false;
    case FaIngraphMode::Decode:
        return phase == InferencePhase::Decode;
    case FaIngraphMode::Both:
        return phase == InferencePhase::Prefill || phase == InferencePhase::Decode;
    }
    return false;
}

bool moeTritonCaptureAllowed() {
    if (env_truthy_("INFINI_MOE_FORCE_HOST_BREAK")) {
        return false;
    }
    if (std::strcmp(cudagraphPolicy(), "eager") == 0) {
        return false;
    }
    const MoeIngraphMode mode = moeIngraphMode_();
    const InferencePhase phase = getInferencePhase();
    switch (mode) {
    case MoeIngraphMode::Off:
        return false;
    case MoeIngraphMode::Decode:
        return phase == InferencePhase::Decode;
    case MoeIngraphMode::Prefill:
        return phase == InferencePhase::Prefill;
    case MoeIngraphMode::Both:
        return phase == InferencePhase::Prefill || phase == InferencePhase::Decode;
    }
    return false;
}

std::shared_ptr<Memory> reinstantiateBlob(std::shared_ptr<Memory> blob) {
    setDevice(blob->device());
    return ContextImpl::singleton().getCurrentRuntime()->reinstantiateBlob(blob);
}

void retainGraphMemory(const std::shared_ptr<Memory> &memory) {
    ContextImpl::singleton().getCurrentRuntime()->retainGraphMemory(memory);
}

} // namespace context

} // namespace infinicore
