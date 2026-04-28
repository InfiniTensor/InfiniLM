#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <infinirt.h>

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "../../infiniop/devices/nvidia/nvidia_handle.cuh"
#define INFINICORE_HAS_NVIDIA_HANDLE 1
#endif

namespace infinicore::graph {

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

struct Graph::DeviceGraph {
    infinirtGraph_t graph;
    infinirtGraphExec_t exec;
    infinirtGraphNode_t node;
    std::vector<char> log_buffer;

    DeviceGraph() {
        log_buffer.resize(4 * 1024);
    }

    ~DeviceGraph() {
        if (exec) {
            infinirtGraphExecDestroy(exec);
        }
        if (graph) {
            infinirtGraphDestroy(graph);
        }
    }

    void launch() {
        INFINICORE_CHECK_ERROR(infinirtGraphLuanch(exec, context::getStream()));
    }
};

Graph::Graph() {
}

void Graph::run() const {
    if (device_graph_ != nullptr && device_graph_.get()->exec != nullptr) {
        device_graph_.get()->launch();
    } else {
        for (auto &op : op_list_) {
            op->run();
        }
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate(const GraphInstantiateFence &fence) {
    device_graph_ = std::make_unique<DeviceGraph>();

    // Eager warmup. Empirically 4 iterations are needed for cublas algo
    // selection / autotuner caches to stabilize; without them the captured
    // graph references not-yet-pinned algo state and replays wrong.
    constexpr size_t kEagerWarmupIters = 4;
    for (size_t iter = 0; iter < kEagerWarmupIters; ++iter) {
        if (iter == 0 && fence) {
            // First-pass JIT-load: fence between every op so all ranks freeze
            // the same kernel at the same moment. After this pass kernels are
            // HSA-cached, so iter>=1 / capture-mode / capture replays don't
            // re-freeze and don't need the fence.
            for (auto &op : op_list_) {
                op->run();
                fence();
            }
        } else {
            this->run();
        }
    }
    // Full-device sync (not just stream): cublas/cudnn use helper streams that
    // can still have work in flight when only the active stream is drained,
    // racing with the kernels we are about to capture.
    context::syncDevice();

#ifdef INFINICORE_HAS_NVIDIA_HANDLE
    device::nvidia::Handle::Internal::setCaptureMode(true);
#endif

    // One more warmup under capture-mode flag so the dedicated capture
    // cublas/cudnn handles get lazily created and stream-bound before
    // BeginCapture; without this the first capture observes an unconfigured
    // handle and produces wrong output.
    this->run();
    context::syncDevice();

    auto begin_status = infinirtStreamBeginCapture(
        context::getStream(),
        INFINIRT_STREAM_CAPTURE_MODE_RELAXED);
    if (begin_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("[graph] StreamBeginCapture failed: status={}", (int)begin_status);
#ifdef INFINICORE_HAS_NVIDIA_HANDLE
        device::nvidia::Handle::Internal::setCaptureMode(false);
#endif
        return;
    }

    this->run();

    auto end_status = infinirtStreamEndCapture(
        context::getStream(),
        &device_graph_.get()->graph);

#ifdef INFINICORE_HAS_NVIDIA_HANDLE
    device::nvidia::Handle::Internal::setCaptureMode(false);
#endif

    if (end_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("[graph] StreamEndCapture failed: status={}", (int)end_status);
        return;
    }

    auto inst_status = infinirtGraphInstantiate(
        &device_graph_.get()->exec,
        device_graph_.get()->graph,
        &device_graph_.get()->node,
        device_graph_.get()->log_buffer.data(),
        device_graph_.get()->log_buffer.size());
    if (inst_status != INFINI_STATUS_SUCCESS) {
        static bool warned_once = false;
        if (!warned_once) {
            warned_once = true;
            spdlog::warn("Fail to instantiate device graph: {}",
                         std::string(device_graph_.get()->log_buffer.data()));
        }
    }
}

Graph::~Graph() = default;

/* =========================
 * GraphManager
 * ========================= */

bool GraphManager::is_recording() const {
    return recording_;
}

void GraphManager::start_recording() {
    if (is_recording()) {
        spdlog::warn("Graph is already recording. Previous recording will be dropped.");
    }
    recording_ = true;
    graph_ = std::make_shared<Graph>();
}

void GraphManager::add_operator(std::shared_ptr<GraphOperator> op) {
    INFINICORE_ASSERT(is_recording());

    graph_->add_operator(op);
}

std::shared_ptr<Graph> GraphManager::stop_recording(const GraphInstantiateFence &fence) {
    if (!is_recording()) {
        spdlog::warn("Graph is not recording. Please start recording first.");
        return nullptr;
    }
    recording_ = false;
#ifdef USE_INFINIRT_GRAPH
    graph_->instantiate(fence);
#else
    (void)fence;
#endif
    return std::exchange(graph_, nullptr);
}

} // namespace infinicore::graph
