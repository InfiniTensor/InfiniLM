#pragma once

#include "execution/graph/graph.hpp"

#include <memory>
#include <mutex>
#include <thread>

namespace infinilm::execution::graph {

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
    void start_recording(bool require_device_graph);
    void add_operator(std::shared_ptr<RecordedCall> op);
    std::shared_ptr<Graph> stop_recording();
    void finish_recording();
    void cancel_recording();

private:
    mutable std::mutex mutex_;
    std::shared_ptr<Graph> graph_;
    std::thread::id capture_owner_;
    bool recording_ = false;
};

} // namespace infinilm::execution::graph
