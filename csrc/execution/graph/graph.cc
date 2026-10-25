#include "graph_manager.hpp"

#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"
#include <infini/rt/device_guard.h>

#ifdef USE_INFINIRT_GRAPH
#include <infini/rt.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace infinilm::execution::graph {

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

GraphTensor::GraphTensor(::infinilm::runtime::Runtime &runtime, const ::infinilm::runtime::Tensor &tensor)
    : ::infinilm::runtime::Tensor(runtime.isGraphRecording() ? tensor->to_blob_(runtime) : tensor) {
    runtime.assertOwnerThread();
}

GraphTensor::GraphTensor(::infinilm::runtime::Runtime &runtime, const ::infinilm::runtime::Tensor &tensor, SnapshotPolicy policy)
    : ::infinilm::runtime::Tensor(policy == SnapshotPolicy::kBlob || runtime.isGraphRecording()
                                      ? tensor->to_blob_(runtime)
                                      : tensor) {
    runtime.assertOwnerThread();
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
        infini::rt::DeviceGuard guard{device_type, device_index};
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
        infini::rt::DeviceGuard guard{device_type, device_index};
        INFINILM_ASSERT(guard.active());
        INFINILM_CHECK_ERROR(rt_runtime::GraphLaunch(exec, stream));
    }

    void synchronize() {
        infini::rt::DeviceGuard guard{device_type, device_index};
        INFINILM_ASSERT(guard.active());
        INFINILM_CHECK_ERROR(rt_runtime::StreamSynchronize(stream));
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

void Graph::retain_runtime(std::shared_ptr<::infinilm::runtime::Runtime> runtime,
                           std::shared_ptr<void> allocation_lease) {
    runtime_lease_ = std::move(runtime);
    allocation_lease_ = std::move(allocation_lease);
    owner_thread_ = std::this_thread::get_id();
}

void Graph::run() const {
    if (runtime_lease_ != nullptr) {
        if (owner_thread_ != std::this_thread::get_id()) {
            throw std::runtime_error("graph replay requires its recording worker");
        }
        runtime_lease_->assertOwnerThread();
    }
    // Only the validated recording worker may read this mutable state.
    if (replay_failed_) {
        throw std::runtime_error("graph replay is retired after a failed execution");
    }
    // Owner validation above is pre-submission and must not retire a
    // valid plan. Exceptions from its recorded execution path do retire it.
    try {
        if (segments_.empty()) {
            for (const auto &op : op_list_) {
                op->run();
            }
            return;
        }
        const Segment *previous = nullptr;
        for (const auto &segment : segments_) {
#ifdef USE_INFINIRT_GRAPH
            if (!segment->capture_safe && previous != nullptr
                && previous->device_graph_ != nullptr
                && previous->device_graph_->exec != nullptr
                && std::any_of(segment->ops.begin(), segment->ops.end(),
                               [](const auto &op) {
                                   return op->host_requires_synchronization();
                               })) {
                // Host code may inspect the preceding device result. Later device
                // work is submitted on the same stream, so it stays ordered.
                previous->device_graph_->synchronize();
            }
#endif
            segment->run();
            previous = segment.get();
        }
    } catch (...) {
        replay_failed_ = true;
        // Preserve the original error and all backing leases. Existing graph
        // destruction drains its owner stream before releasing resources.
        throw;
    }
}

Graph::ExecutionReport Graph::execution_report() const {
    return execution_report_;
}

void Graph::add_operator(std::shared_ptr<RecordedCall> op) {
    op_list_.push_back(std::move(op));
}

void Graph::instantiate() {
    execution_report_ = {};
    execution_report_.operator_count = op_list_.size();
#ifdef USE_INFINIRT_GRAPH
    segments_.clear();

    // Keep recorded-operator replay available as a diagnostic escape hatch.
    if (std::getenv("INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS") != nullptr) {
        if (require_device_graph_) {
            throw std::runtime_error("device graph is required but disabled by the environment");
        }
        execution_report_.host_segments = op_list_.empty() ? 0 : 1;
        execution_report_.fallback_reason = "disabled_by_environment";
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
        if (require_device_graph_) {
            throw std::runtime_error("device graph is required but no operators were recorded");
        }
        return;
    }

    // Only explicitly declared host calls may split captured device segments.
    // An unqualified legacy operator keeps the conservative whole-host path.
    const auto first_unqualified = std::find_if(
        op_list_.begin(), op_list_.end(),
        [](const auto &op) {
            return !op->is_device_graph_capture_safe()
                && !op->is_explicit_host_replay();
        });
    if (first_unqualified != op_list_.end()) {
        const auto index = std::distance(op_list_.begin(), first_unqualified);
        const auto detail = "operator #" + std::to_string(index) + " (" + (*first_unqualified)->name() + ")";
        if (require_device_graph_) {
            throw std::runtime_error(
                "device graph is required but " + detail + " is not capture-qualified");
        }
        execution_report_.host_segments = 1;
        execution_report_.fallback_reason = "unqualified_capture_operator";
        spdlog::warn("Device graph disabled: {} is not capture-qualified", detail);
        segments_.clear();
        return;
    }

    // An explicitly declared all-host plan does not attempt device capture.
    // Keep it available as a diagnostic, even without graph backend support.
    const bool has_device_segment = std::any_of(
        segments_.begin(), segments_.end(),
        [](const auto &segment) { return segment->capture_safe; });
    if (!has_device_segment) {
        if (require_device_graph_) {
            throw std::runtime_error("device graph is required but no segment was captured");
        }
        execution_report_.host_segments = segments_.size();
        return;
    }

    auto current_device = runtime_lease_->device();
    infini::rt::DeviceGuard device_guard{current_device.type(), current_device.index()};
    if (!device_guard.active()) {
        throw std::runtime_error("device graph candidate device selection failed");
    }

    for (auto &segment : segments_) {
        if (!segment->capture_safe) {
            continue; // Do not execute a host segment while assembling the plan.
        }
        segment->device_graph_ = std::make_unique<DeviceGraph>();
        segment->device_graph_->device_type = current_device.type();
        segment->device_graph_->device_index = current_device.index();
        segment->device_graph_->stream = runtime_lease_->stream();

        StreamCaptureGuard capture_guard{segment->device_graph_->stream};
        execution_report_.capture_attempted = true;
        auto begin_status = capture_guard.begin();
        if (begin_status != rt_runtime::kSuccess) {
            throw std::runtime_error("device graph segment capture could not begin (RT error "
                                     + std::to_string(static_cast<int>(begin_status)) + ")");
        }

        // Running the segment records only capture-safe operators.
        segment->run();

        auto end_status = capture_guard.end(&segment->device_graph_->graph);
        if (end_status != rt_runtime::kSuccess) {
            // A failed StreamEndCapture may leave the device stream in an
            // error state. Replaying on that stream would not be a safe
            // fallback, even when strict graph mode was not requested.
            throw std::runtime_error(
                "device graph segment capture could not end (RT error "
                + std::to_string(static_cast<int>(end_status)) + ")");
        }

        auto instantiate_status = rt_runtime::GraphInstantiate(
            &segment->device_graph_->exec,
            segment->device_graph_->graph);
        if (instantiate_status != rt_runtime::kSuccess) {
            throw std::runtime_error("device graph segment instantiation failed (RT error "
                                     + std::to_string(static_cast<int>(instantiate_status)) + ")");
        }
    }

    for (const auto &segment : segments_) {
        if (segment->device_graph_ != nullptr && segment->device_graph_->exec != nullptr) {
            ++execution_report_.device_graph_segments;
        } else {
            ++execution_report_.host_segments;
        }
    }
    if (require_device_graph_ && execution_report_.device_graph_segments == 0) {
        throw std::runtime_error("device graph is required but no segment was captured");
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

#else
    execution_report_.host_segments = op_list_.empty() ? 0 : 1;
    execution_report_.fallback_reason = "graph_runtime_not_built";
    if (require_device_graph_) {
        throw std::runtime_error("device graph is required but InfiniRT graph support is not built");
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

void GraphManager::start_recording(bool require_device_graph) {
    auto graph = std::make_shared<Graph>();
    graph->require_device_graph_ = require_device_graph;
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

void GraphManager::add_operator(std::shared_ptr<RecordedCall> op) {
    std::lock_guard<std::mutex> lock{mutex_};
    INFINILM_ASSERT(recording_ && capture_owner_ == std::this_thread::get_id());
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
    return graph;
}

void GraphManager::finish_recording() {
    std::lock_guard<std::mutex> lock{mutex_};
    INFINILM_ASSERT(recording_ && capture_owner_ == std::this_thread::get_id());
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

} // namespace infinilm::execution::graph
