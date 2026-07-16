#pragma once

#include "infinicore/graph/graph.hpp"

#include <memory>
#include <mutex>
#include <thread>

namespace infinicore::graph {

class GraphManager {
public:
    enum class CaptureState {
        kInactive,
        kActiveOwner,
        kActiveNonOwner,
    };

    GraphManager() = default;
    ~GraphManager() = default;

    CaptureState capture_state() const;
    bool is_recording() const;
    void start_recording();
    void add_operator(std::shared_ptr<GraphOperator> op);
    std::shared_ptr<Graph> stop_recording();
    void finish_recording();
    void cancel_recording();

private:
    mutable std::mutex mutex_;
    std::shared_ptr<Graph> graph_;
    std::thread::id capture_owner_;
    bool recording_ = false;
};

} // namespace infinicore::graph
