#pragma once

#include "../../models/infinilm_model.hpp"
#include "../rank_barrier.hpp"

namespace infinilm::engine {

class GraphRecordingGuard {
public:
    GraphRecordingGuard() {
        infinicore::context::startGraphRecording();
    }

    ~GraphRecordingGuard() noexcept {
        if (active_) {
            infinicore::context::cancelGraphRecording();
        }
    }

    GraphRecordingGuard(const GraphRecordingGuard &) = delete;
    GraphRecordingGuard &operator=(const GraphRecordingGuard &) = delete;
    GraphRecordingGuard(GraphRecordingGuard &&) = delete;
    GraphRecordingGuard &operator=(GraphRecordingGuard &&) = delete;

    std::shared_ptr<infinicore::graph::Graph> finish() {
        auto graph = infinicore::context::stopGraphRecording();
        active_ = false;
        return graph;
    }

private:
    bool active_ = true;
};

class GraphCompiler {
public:
    using Compiled = std::tuple<
        std::shared_ptr<infinicore::graph::Graph>,
        std::shared_ptr<InfinilmModel::Output>>;

    explicit GraphCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier) : model_(model), barrier_(barrier) {}
    virtual ~GraphCompiler() = default;

    virtual void compile() = 0;
    virtual Compiled get_compiled(const InfinilmModel::Input &input) = 0;

protected:
    std::shared_ptr<InfinilmModel> model_;
    RankBarrier *barrier_;
};

} // namespace infinilm::engine
