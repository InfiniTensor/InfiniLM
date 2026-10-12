#pragma once

#include "../../models/infinilm_model.hpp"
#include "../../global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include "../rank_barrier.hpp"

#include <cstdlib>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string_view>


namespace infinilm::engine {

class GraphRecordingGuard {
public:
    explicit GraphRecordingGuard(infinilm::runtime::Runtime &runtime)
        : runtime_(runtime) {
        runtime_.assertOwnerThread();
        if (runtime_.isGraphRecording()) {
            throw std::runtime_error("graph compiler cannot nest Runtime recording");
        }
        const char *strict = std::getenv("INFINILM_REQUIRE_DEVICE_GRAPH");
        runtime_.startGraphRecording(
            strict != nullptr && std::string_view(strict) == "1");
        active_ = true;
    }

    ~GraphRecordingGuard() noexcept {
        if (active_) {
            runtime_.cancelGraphRecording();
        }
    }

    GraphRecordingGuard(const GraphRecordingGuard &) = delete;
    GraphRecordingGuard &operator=(const GraphRecordingGuard &) = delete;
    GraphRecordingGuard(GraphRecordingGuard &&) = delete;
    GraphRecordingGuard &operator=(GraphRecordingGuard &&) = delete;

    std::shared_ptr<infinilm::execution::graph::Graph> finish() {
        try {
            auto graph = runtime_.stopGraphRecording();
            active_ = false;
            return graph;
        } catch (...) {
            auto error = std::current_exception();
            if (active_) {
                runtime_.cancelGraphRecording();
                active_ = false;
            }
            std::rethrow_exception(error);
        }
    }

private:
    infinilm::runtime::Runtime &runtime_;
    bool active_ = false;
};

class GraphCompiler {
public:
    using Compiled = std::tuple<
        std::shared_ptr<infinilm::execution::graph::Graph>,
        std::shared_ptr<InfinilmModel::Output>>;

    GraphCompiler(const std::shared_ptr<InfinilmModel> &model,
                  RankBarrier *barrier,
                  infinilm::runtime::Runtime &runtime,
                  infinilm::global_state::ForwardContext &forward_context)
        : model_(model), barrier_(barrier), runtime_(runtime),
          forward_context_(forward_context) {}
    virtual ~GraphCompiler() = default;

    virtual void compile() = 0;
    virtual Compiled get_compiled(const InfinilmModel::Input &input) = 0;

protected:
    std::shared_ptr<InfinilmModel> model_;
    RankBarrier *barrier_;
    infinilm::runtime::Runtime &runtime_;
    infinilm::global_state::ForwardContext &forward_context_;
};

} // namespace infinilm::engine
