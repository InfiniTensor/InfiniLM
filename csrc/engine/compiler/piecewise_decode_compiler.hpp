#pragma once

#include "../../global_state/piecewise_prefill_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "../rank_barrier.hpp"
#include "graph_compiler.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace infinilm::engine {

/// Per-layer decode CUDAGraphs: capturable pre / eager FA / capturable post-without-MoE /
/// eager MoE between Graph::run(). Mirrors prefill FA2 piecewise for MiniCPM5 MetaX.
class PiecewiseDecodeCompiler {
public:
    struct BatchGraphs {
        InfinilmModel::Input input;
        std::vector<std::shared_ptr<infinicore::graph::Graph>> pre_attn;
        std::vector<std::shared_ptr<infinicore::graph::Graph>> post_attn;
        std::shared_ptr<infinicore::graph::Graph> lm_head;
        infinicore::Tensor logits_holder;
        infinicore::Tensor hidden_states;
        infinicore::Tensor residual;
        infinicore::Tensor ar_staging;
        std::vector<global_state::PiecewiseLayerStaging> layer_staging;
        size_t device_segments{0};
    };

    PiecewiseDecodeCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile();
    bool enabled() const { return enabled_; }

    /// Replay piecewise graphs for a decode step. Returns logits on hit.
    std::optional<infinicore::Tensor> run_decode(const InfinilmModel::Input &input);

    const std::vector<size_t> &capture_batches() const { return capture_batches_; }
    size_t segment_replays() const { return segment_replays_; }
    size_t decode_hits() const { return decode_hits_; }
    size_t decode_misses() const { return decode_misses_; }
    size_t device_segments_captured() const { return device_segments_captured_; }

private:
    void allocate_layer_staging_(size_t batch, size_t num_layers);
    InfinilmModel::Input make_batch_input_(size_t batch, size_t nblocks) const;
    void capture_batch_(size_t batch);
    void copy_runtime_into_batch_(BatchGraphs &batch_graphs,
                                  const InfinilmModel::Input &runtime) const;

    std::shared_ptr<InfinilmModel> model_;
    RankBarrier *barrier_;
    bool enabled_{false};
    std::vector<size_t> capture_batches_;
    infinicore::Tensor block_tables_holder_;
    std::unordered_map<size_t, BatchGraphs> compiled_;
    size_t segment_replays_{0};
    size_t decode_hits_{0};
    size_t decode_misses_{0};
    size_t device_segments_captured_{0};
};

} // namespace infinilm::engine
