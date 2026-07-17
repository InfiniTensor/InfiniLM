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

/// Decode CUDAGraphs with MetaX M=1 segment fusion.
///
/// Legacy (``INFINI_DECODE_PIECEWISE_FUSE_LAYERS=1``): per-layer pre / eager FA /
/// post-without-MoE / eager MoE — ~57 separate Graph::run launches.
///
/// Span fuse (default, ``FUSE_LAYERS=0``): FA stays outside recording; each span is
/// ``post_i + MoE_hostbreak [+ pre_{i+1}]`` (smoke-proven host-break pattern).
/// Cuts Graph::run ~57→~30 while keeping host-break MoE.
class PiecewiseDecodeCompiler {
public:
    struct BatchGraphs {
        InfinilmModel::Input input;
        /// Legacy split capture (fuse_layers==1).
        std::vector<std::shared_ptr<infinicore::graph::Graph>> pre_attn;
        std::vector<std::shared_ptr<infinicore::graph::Graph>> post_attn;
        /// Fused multi-layer graphs (fuse_layers!=1); each covers a layer span.
        std::vector<std::shared_ptr<infinicore::graph::Graph>> layer_groups;
        /// Layer index where each layer_groups[i] starts.
        std::vector<size_t> group_layer0;
        std::shared_ptr<infinicore::graph::Graph> lm_head;
        infinicore::Tensor logits_holder;
        infinicore::Tensor hidden_states;
        infinicore::Tensor residual;
        infinicore::Tensor ar_staging;
        std::vector<global_state::PiecewiseLayerStaging> layer_staging;
        size_t device_segments{0};
        size_t fuse_layers{0}; // 0 = all layers in one group; 1 = legacy
        size_t capture_layers{0};
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
    void capture_batch_legacy_(size_t batch, BatchGraphs &graphs, size_t capture_layers);
    void capture_batch_fused_(size_t batch, BatchGraphs &graphs, size_t capture_layers);
    void copy_runtime_into_batch_(BatchGraphs &batch_graphs,
                                  const InfinilmModel::Input &runtime) const;

    std::shared_ptr<InfinilmModel> model_;
    RankBarrier *barrier_;
    bool enabled_{false};
    std::vector<size_t> capture_batches_;
    size_t fuse_layers_{0}; // parsed once; 0 = fuse all
    infinicore::Tensor block_tables_holder_;
    std::unordered_map<size_t, BatchGraphs> compiled_;
    size_t segment_replays_{0};
    size_t decode_hits_{0};
    size_t decode_misses_{0};
    size_t device_segments_captured_{0};
};

} // namespace infinilm::engine
