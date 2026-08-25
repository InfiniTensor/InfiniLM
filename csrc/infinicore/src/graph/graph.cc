#include "graph_manager.hpp"

#include "../context/runtime/runtime.hpp"
#include "../utils.hpp"
#include "infinicore/context/context.hpp"

#ifdef USE_INFINIRT_GRAPH
#include <infini/rt.h>
#endif

#include <cstdlib>
#include <utility>

namespace infinicore::graph {

#ifdef USE_INFINIRT_GRAPH
namespace rt_runtime = ::infini::rt::runtime;

namespace {

void warn_runtime_failure(const char *operation, rt_runtime::Error status) noexcept {
    if (status == rt_runtime::kSuccess) {
        return;
    }
    try {
        spdlog::warn("{} failed during graph cleanup with error code {}",
                     operation,
                     static_cast<long long>(status));
    } catch (...) {
    }
}

void warn_runtime_failure(const char *operation, const char *detail) noexcept {
    try {
        spdlog::warn("{} failed during graph cleanup: {}", operation, detail);
    } catch (...) {
    }
}

class RuntimeDeviceGuard {
public:
    RuntimeDeviceGuard(::infini::rt::Device::Type target_type, int target_index) noexcept
        : previous_type_(::infini::rt::runtime_device_type()) {
        ::infini::rt::set_runtime_device_type(previous_type_);
        const auto get_status = rt_runtime::GetDevice(&previous_index_);
        restore_index_ = get_status == rt_runtime::kSuccess;
        warn_runtime_failure("reading the previous device", get_status);

        if (target_type == ::infini::rt::Device::Type::kCount) {
            warn_runtime_failure("selecting the graph device", "invalid device type");
            restore();
            return;
        }
        ::infini::rt::set_runtime_device_type(target_type);
        const auto set_status = rt_runtime::SetDevice(target_index);
        active_ = set_status == rt_runtime::kSuccess;
        warn_runtime_failure("selecting the graph device", set_status);
        if (!active_) {
            restore();
        }
    }

    RuntimeDeviceGuard(const RuntimeDeviceGuard &) = delete;
    RuntimeDeviceGuard &operator=(const RuntimeDeviceGuard &) = delete;

    ~RuntimeDeviceGuard() noexcept { restore(); }

    bool active() const { return active_; }

private:
    void restore() noexcept {
        if (restored_) {
            return;
        }
        restored_ = true;
        ::infini::rt::set_runtime_device_type(previous_type_);
        if (restore_index_) {
            warn_runtime_failure("restoring the previous device", rt_runtime::SetDevice(previous_index_));
        }
    }

    ::infini::rt::Device::Type previous_type_;
    int previous_index_ = 0;
    bool restore_index_ = false;
    bool active_ = false;
    bool restored_ = false;
};

class StreamCaptureGuard {
public:
    explicit StreamCaptureGuard(rt_runtime::Stream stream) : stream_(stream) {}

    StreamCaptureGuard(const StreamCaptureGuard &) = delete;
    StreamCaptureGuard &operator=(const StreamCaptureGuard &) = delete;

    ~StreamCaptureGuard() noexcept { abort(); }

    rt_runtime::Error begin() {
        const auto status = rt_runtime::StreamBeginCapture(
            stream_, rt_runtime::StreamCaptureMode::kStreamCaptureModeRelaxed);
        active_ = status == rt_runtime::kSuccess;
        return status;
    }

    rt_runtime::Error end(rt_runtime::Graph *graph) {
        active_ = false;
        return rt_runtime::StreamEndCapture(stream_, graph);
    }

private:
    void abort() noexcept {
        if (!active_) {
            return;
        }
        active_ = false;
        rt_runtime::Graph abandoned_graph = nullptr;
        const auto end_status = rt_runtime::StreamEndCapture(stream_, &abandoned_graph);
        warn_runtime_failure("ending an abandoned stream capture", end_status);
        if (end_status == rt_runtime::kSuccess && abandoned_graph != nullptr) {
            warn_runtime_failure("destroying an abandoned graph", rt_runtime::GraphDestroy(abandoned_graph));
        }
    }

    rt_runtime::Stream stream_;
    bool active_ = false;
};

} // namespace
#endif

/* =========================
 * GraphTensor
 * ========================= */

GraphTensor::GraphTensor(const Tensor &tensor) : Tensor(tensor->to_blob_()) {
}

/* =========================
 * GraphOperator
 * ========================= */

void DispatchableGraphOperator::run() const {
    runner_(planned_meta_);
}

DispatchableGraphOperator::~DispatchableGraphOperator() {
    if (deleter_) {
        deleter_(&planned_meta_);
    }
}

/* =========================
 * Graph
 * ========================= */

#ifdef USE_INFINIRT_GRAPH
struct Graph::DeviceGraph {
    rt_runtime::Graph graph = nullptr;
    rt_runtime::GraphExec exec = nullptr;
    rt_runtime::Stream stream = nullptr;
    ::infini::rt::Device::Type device_type = ::infini::rt::Device::Type::kCount;
    int device_index = 0;

    ~DeviceGraph() noexcept {
        if (exec == nullptr && graph == nullptr) {
            return;
        }
        RuntimeDeviceGuard guard{device_type, device_index};
        if (!guard.active()) {
            warn_runtime_failure("activating the graph device for cleanup", "device selection failed");
            return;
        }
        if (exec) {
            warn_runtime_failure("destroying the graph executable", rt_runtime::GraphExecDestroy(exec));
            exec = nullptr;
        }
        if (graph) {
            warn_runtime_failure("destroying the graph", rt_runtime::GraphDestroy(graph));
            graph = nullptr;
        }
    }

    void launch() {
        RuntimeDeviceGuard guard{device_type, device_index};
        INFINICORE_ASSERT(guard.active());
        INFINICORE_CHECK_ERROR(rt_runtime::GraphLaunch(exec, stream));
    }
};
#else
struct Graph::DeviceGraph {};
#endif

Graph::Segment::Segment(bool capture_safe_) : capture_safe(capture_safe_) {
}

Graph::Segment::~Segment() noexcept = default;

void Graph::Segment::run() const {
#ifdef USE_INFINIRT_GRAPH
    if (device_graph_ != nullptr && device_graph_->exec != nullptr) {
        device_graph_->launch();
        return;
    }
#endif
    for (const auto &op : ops) {
        op->run();
    }
}

Graph::Graph() {
}

void Graph::retain_runtime(std::shared_ptr<::infinicore::Runtime> runtime,
                           std::shared_ptr<void> allocation_lease) {
    runtime_lease_ = std::move(runtime);
    allocation_lease_ = std::move(allocation_lease);
}

void Graph::run() const {
    (void)context::isGraphRecording();
    if (segments_.empty()) {
        for (const auto &op : op_list_) {
            op->run();
        }
        return;
    }
    for (const auto &segment : segments_) {
        segment->run();
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate() {
#ifdef USE_INFINIRT_GRAPH
    segments_.clear();

    // Warm the complete op list before splitting it into replay segments.
    for (size_t iter = 0; iter < 5; ++iter) {
        this->run();
    }
    infinicore::context::syncStream();

    // Keep recorded-operator replay available as a diagnostic escape hatch.
    if (std::getenv("INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS") != nullptr) {
        spdlog::info("device graph segments disabled; replaying recorded operators");
        return;
    }

    for (const auto &op : op_list_) {
        const bool capture_safe = op->is_device_graph_capture_safe();
        if (segments_.empty()
            || segments_.back()->capture_safe != capture_safe) {
            segments_.push_back(std::make_unique<Segment>(capture_safe));
        }
        segments_.back()->ops.push_back(op);
    }

    if (segments_.empty()) {
        return;
    }

    auto current_device = context::getDevice();
    RuntimeDeviceGuard device_guard{current_device.type(), current_device.index()};
    if (!device_guard.active()) {
        spdlog::warn("InfiniRT graph runtime failed to select the current device. Falling back to eager execution.");
        return;
    }

    bool capture_failed = false;
    for (auto &segment : segments_) {
        if (capture_failed) {
            segment->run();
            continue;
        }
        if (!segment->capture_safe) {
            // Execute host segments once between captured segments so later
            // capture observes the same stream-ordered dependencies.
            segment->run();
            continue;
        }

        segment->device_graph_ = std::make_unique<DeviceGraph>();
        segment->device_graph_->device_type = current_device.type();
        segment->device_graph_->device_index = current_device.index();
        segment->device_graph_->stream = context::getStream();

        StreamCaptureGuard capture_guard{segment->device_graph_->stream};
        auto begin_status = capture_guard.begin();
        if (begin_status != rt_runtime::kSuccess) {
            spdlog::warn("Fail to begin device graph segment capture. Falling back to eager execution.");
            segment->device_graph_.reset();
            segment->run();
            capture_failed = true;
            continue;
        }

        // Running the segment records only capture-safe operators.
        segment->run();

        auto end_status = capture_guard.end(&segment->device_graph_->graph);
        if (end_status != rt_runtime::kSuccess) {
            spdlog::warn("Fail to end device graph segment capture. Falling back to eager execution.");
            segment->device_graph_.reset();
            capture_failed = true;
            continue;
        }

        auto instantiate_status = rt_runtime::GraphInstantiate(
            &segment->device_graph_->exec,
            segment->device_graph_->graph);
        if (instantiate_status != rt_runtime::kSuccess) {
            static bool warned_once = false;
            if (!warned_once) {
                warned_once = true;
                spdlog::warn("Fail to instantiate device graph segment. Falling back to eager execution.");
            }
            segment->device_graph_.reset();
            capture_failed = true;
        }
    }

    if (capture_failed) {
        // The assembly pass has still run every operator exactly once. Drop
        // any successfully captured prefix so future runs replay wholly eager.
        for (auto &segment : segments_) {
            segment->device_graph_.reset();
        }
        return;
    }

    if (std::getenv("INFINICORE_GRAPH_DEBUG") != nullptr) {
        size_t host_segments = 0;
        for (const auto &segment : segments_) {
            host_segments += segment->capture_safe ? 0 : 1;
        }
        spdlog::info(
            "segmented graph: operators={}, segments={}, host_segments={}",
            op_list_.size(), segments_.size(), host_segments);
    }

    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        spdlog::info("Using InfiniRT C++ segmented graph runtime API for graph capture and replay.");
    }
#endif
}

Graph::~Graph() noexcept {
    if (runtime_lease_ != nullptr) {
        runtime_lease_->syncStreamForCleanup();
    }
}

/* =========================
 * GraphManager
 * ========================= */

bool GraphManager::is_recording() const {
    std::lock_guard<std::mutex> lock{mutex_};
    if (!recording_) {
        return false;
    }
    const auto state = capture_owner_ == std::this_thread::get_id()
                         ? CaptureState::kActiveOwner
                         : CaptureState::kActiveNonOwner;
    if (state == CaptureState::kActiveNonOwner) {
        throw std::runtime_error("cannot access the shared runtime stream: another thread owns the graph capture");
    }
    return true;
}

GraphManager::CaptureState GraphManager::capture_state() const {
    std::lock_guard<std::mutex> lock{mutex_};
    if (!recording_) {
        return CaptureState::kInactive;
    }
    return capture_owner_ == std::this_thread::get_id()
             ? CaptureState::kActiveOwner
             : CaptureState::kActiveNonOwner;
}

void GraphManager::start_recording() {
    auto graph = std::make_shared<Graph>();
    const auto caller = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock{mutex_};
    if (recording_ && capture_owner_ != caller) {
        throw std::runtime_error("cannot start graph recording: another thread owns the capture");
    }
    if (recording_) {
        spdlog::warn("Graph is already recording. Previous recording will be dropped.");
    }
    recording_ = true;
    capture_owner_ = caller;
    graph_ = std::move(graph);
}

void GraphManager::add_operator(std::shared_ptr<GraphOperator> op) {
    std::lock_guard<std::mutex> lock{mutex_};
    INFINICORE_ASSERT(recording_ && capture_owner_ == std::this_thread::get_id());
    graph_->add_operator(op);
}

std::shared_ptr<Graph> GraphManager::stop_recording() {
    std::shared_ptr<Graph> graph;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!recording_) {
            spdlog::warn("Graph is not recording. Please start recording first.");
            return nullptr;
        }
        if (capture_owner_ != std::this_thread::get_id()) {
            throw std::runtime_error("cannot stop graph recording: another thread owns the capture");
        }
        graph = std::exchange(graph_, nullptr);
    }
#ifdef USE_INFINIRT_GRAPH
    graph->instantiate();
#endif
    return graph;
}

void GraphManager::finish_recording() {
    std::lock_guard<std::mutex> lock{mutex_};
    INFINICORE_ASSERT(recording_ && capture_owner_ == std::this_thread::get_id());
    recording_ = false;
    capture_owner_ = {};
}

void GraphManager::cancel_recording() {
    std::shared_ptr<Graph> graph;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (recording_ && capture_owner_ == std::this_thread::get_id()) {
            recording_ = false;
            capture_owner_ = {};
            graph = std::exchange(graph_, nullptr);
        }
    }
    graph.reset();
}

} // namespace infinicore::graph
