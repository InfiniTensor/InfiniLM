#pragma once

#include "infinicore/graph/graph.hpp"

#include <memory>
#include <vector>

namespace infinicore::graph {

class GraphManager {
public:
    GraphManager() = default;
    ~GraphManager() = default;

    bool is_recording() const;
    void start_recording();
    void add_operator(std::shared_ptr<GraphOperator> op);
    std::shared_ptr<Graph> stop_recording(const GraphInstantiateFence &fence = nullptr);

private:
    std::shared_ptr<Graph> graph_;
    bool recording_ = false;
};

} // namespace infinicore::graph
