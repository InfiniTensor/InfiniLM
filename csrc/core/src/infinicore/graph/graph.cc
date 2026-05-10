#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <infinirt.h>

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

    DeviceGraph() : graph(nullptr), exec(nullptr), node(nullptr) {
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

void Graph::instantiate(const GraphInstantiateFence &fence,
                        const GraphPinModeSetter &set_pin_mode) {
    device_graph_ = std::make_unique<DeviceGraph>();

    auto set_pinned = [&set_pin_mode](bool pinned) {
        if (set_pin_mode) {
            set_pin_mode(pinned);
        }
    };
    struct PinModeReset {
        const GraphPinModeSetter &set_pin_mode;
        ~PinModeReset() {
            if (set_pin_mode) {
                set_pin_mode(false);
            }
        }
    } pin_mode_reset{set_pin_mode};

    set_pinned(false);

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

    // One more warmup under capture-mode flag so the dedicated capture
    // cublas/cudnn handles get lazily created and stream-bound before
    // BeginCapture; without this the first capture observes an unconfigured
    // handle and produces wrong output.
    this->run();
    context::syncDevice();

    set_pinned(true);

    auto begin_status = infinirtStreamBeginCapture(
        context::getStream(),
        INFINIRT_STREAM_CAPTURE_MODE_RELAXED);
    if (begin_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("[graph] StreamBeginCapture failed: status={}", (int)begin_status);
        return;
    }

    this->run();

    auto end_status = infinirtStreamEndCapture(
        context::getStream(),
        &device_graph_.get()->graph);
    set_pinned(false);

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

std::shared_ptr<Graph> GraphManager::stop_recording(const GraphInstantiateFence &fence,
                                                    const GraphPinModeSetter &set_pin_mode) {
    if (!is_recording()) {
        spdlog::warn("Graph is not recording. Please start recording first.");
        return nullptr;
    }
    recording_ = false;
#ifdef USE_INFINIRT_GRAPH
    graph_->instantiate(fence, set_pin_mode);
#else
    (void)fence;
    (void)set_pin_mode;
#endif
    return std::exchange(graph_, nullptr);
}

} // namespace infinicore::graph
