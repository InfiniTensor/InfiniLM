#pragma once

#include "../../global_state/piecewise_prefill_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "../rank_barrier.hpp"
#include "graph_compiler.hpp"
#include "paged_compiler.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace infinilm::engine {

class PiecewisePrefillCompiler {
public:
    struct BucketGraphs {
        InfinilmModel::Input input;
        std::vector<std::shared_ptr<infinicore::graph::Graph>> pre_attn;
        std::vector<std::shared_ptr<infinicore::graph::Graph>> post_attn;
        std::shared_ptr<infinicore::graph::Graph> lm_head;
        infinicore::Tensor logits_holder;
        infinicore::Tensor hidden_states;
        infinicore::Tensor residual;
        infinicore::Tensor ar_staging;
        std::vector<global_state::PiecewiseLayerStaging> layer_staging;
    };

    PiecewisePrefillCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile();
    bool enabled() const { return enabled_; }

    /// Replay piecewise graphs for a prefill request. Returns logits on hit.
    std::optional<infinicore::Tensor> run_prefill(const InfinilmModel::Input &input);

    size_t padded_bucket_for(size_t seq_len) const;
    size_t max_capture_req() const { return max_capture_req_; }
    const std::vector<size_t> &capture_buckets() const { return capture_buckets_; }
    size_t segment_replays() const { return segment_replays_; }
    size_t prefill_hits() const { return prefill_hits_; }
    size_t prefill_misses() const { return prefill_misses_; }

private:
    void allocate_layer_staging_(size_t bucket, size_t num_layers);
    InfinilmModel::Input make_bucket_input_(size_t bucket, size_t nblocks, size_t n_req) const;
    void capture_bucket_(size_t bucket);
    void warmup_inductor_segments_(size_t nblocks, size_t n_req);
    void copy_runtime_into_bucket_(BucketGraphs &bucket_graphs,
                                   const InfinilmModel::Input &runtime,
                                   size_t valid_seq_len) const;

    std::shared_ptr<InfinilmModel> model_;
    RankBarrier *barrier_;
    bool enabled_{false};
    size_t max_seq_len_{0};
    std::vector<size_t> capture_buckets_;
    std::vector<size_t> bs_to_padded_;
    size_t max_capture_req_{1};
    infinicore::Tensor block_tables_holder_;
    std::unordered_map<size_t, BucketGraphs> compiled_;
    size_t segment_replays_{0};
    size_t prefill_hits_{0};
    size_t prefill_misses_{0};
};

} // namespace infinilm::engine
