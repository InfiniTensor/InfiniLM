#include "graph_manager.hpp"

#include "../context/runtime/runtime.hpp"
#include "../utils.hpp"
#include "infinicore/context/context.hpp"

#ifdef USE_INFINIRT_GRAPH
#include <infini/rt.h>
#endif

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

Graph::Graph() {
}

void Graph::retain_runtime(std::shared_ptr<::infinicore::Runtime> runtime,
                           std::shared_ptr<void> allocation_lease) {
    runtime_lease_ = std::move(runtime);
    allocation_lease_ = std::move(allocation_lease);
}

void Graph::run() const {
    (void)context::isGraphRecording();
#ifdef USE_INFINIRT_GRAPH
    if (device_graph_ != nullptr && device_graph_.get()->exec != nullptr) {
        device_graph_.get()->launch();
        return;
    }
#endif
    for (auto &op : op_list_) {
        op->run();
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate() {
#ifdef USE_INFINIRT_GRAPH
    // Reset device graph
    device_graph_ = std::make_unique<DeviceGraph>();
    auto current_device = context::getDevice();
    device_graph_->device_type = current_device.type();
    device_graph_->device_index = current_device.index();
    device_graph_->stream = context::getStream();
    RuntimeDeviceGuard device_guard{device_graph_->device_type, device_graph_->device_index};
    if (!device_guard.active()) {
        spdlog::warn("InfiniRT graph runtime failed to select the current device. Falling back to eager execution.");
        device_graph_.reset();
        return;
    }

    // warmup
    for (size_t iter = 0; iter < 5; ++iter) {
        this->run();
    }
    infinicore::context::syncStream();

    StreamCaptureGuard capture_guard{device_graph_->stream};
    auto begin_status = capture_guard.begin();
    if (begin_status != rt_runtime::kSuccess) {
        spdlog::warn("Fail to begin device graph capture.");
        device_graph_.reset();
        return;
    }

    // Run and record
    this->run();

    auto end_status = capture_guard.end(&device_graph_->graph);
    if (end_status != rt_runtime::kSuccess) {
        spdlog::warn("Fail to end device graph capture.");
        device_graph_.reset();
        return;
    }

    auto instantiate_status = rt_runtime::GraphInstantiate(
        &device_graph_->exec,
        device_graph_->graph);
    if (instantiate_status != rt_runtime::kSuccess) {
        static bool warned_once = false;
        if (!warned_once) {
            warned_once = true;
            spdlog::warn("Fail to instantiate device graph.");
        }
        device_graph_.reset();
        return;
    }
    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        spdlog::info("Using InfiniRT C++ graph runtime API for graph capture and replay.");
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
