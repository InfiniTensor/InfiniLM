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

    /// M2: one max-bucket staging/I-O bank; smaller buckets use prefix views.
    struct SharedPrefillBanks {
        size_t max_bucket{0};
        size_t num_layers{0};
        size_t staging_physical_bytes{0};
        size_t io_physical_bytes{0};
        infinicore::Tensor hidden_states;
        infinicore::Tensor residual;
        infinicore::Tensor ar_staging;
        std::vector<global_state::PiecewiseLayerStaging> layer_staging;
        infinicore::Tensor input_ids;
        infinicore::Tensor position_ids;
        infinicore::Tensor past_sequence_lengths;
        infinicore::Tensor total_sequence_lengths;
        infinicore::Tensor input_offsets;
        infinicore::Tensor cu_seqlens;
        infinicore::Tensor slot_mapping;
        infinicore::Tensor logits_holder;
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
    bool last_prefill_executed() const { return last_prefill_executed_; }

private:
    void allocate_shared_banks_(size_t max_bucket, size_t num_layers, size_t n_req);
    /// Bind prefix views of the shared max bank into piecewise TLS for ``bucket``.
    void bind_bucket_staging_(size_t bucket, size_t num_layers);
    /// ``mid_chunk_capture``: past_len=chunk_size, positions continue from past (exact mid key).
    InfinilmModel::Input make_bucket_input_(size_t bucket,
                                            size_t nblocks,
                                            size_t n_req,
                                            bool mid_chunk_capture = false) const;
    /// MIXED dual-capture: n_req=2 continuing mid (~bucket-1 toks) + 1 decode tok @ chunk bucket.
    InfinilmModel::Input make_mixed_mid_bucket_input_(size_t bucket, size_t nblocks) const;
    void capture_bucket_(size_t bucket, bool mid_chunk_capture = false);
    void capture_mixed_mid_bucket_(size_t bucket);
    void warmup_inductor_segments_(size_t nblocks, size_t n_req);
    void copy_runtime_into_bucket_(BucketGraphs &bucket_graphs,
                                   const InfinilmModel::Input &runtime,
                                   size_t valid_seq_len) const;

    std::shared_ptr<InfinilmModel> model_;
    RankBarrier *barrier_;
    bool enabled_{false};
    size_t max_seq_len_{0};
    size_t prefill_chunk_size_{0};
    std::vector<size_t> capture_buckets_;
    std::vector<size_t> bs_to_padded_;
    size_t max_capture_req_{1};
    infinicore::Tensor block_tables_holder_;
    SharedPrefillBanks shared_banks_;
    std::unordered_map<size_t, BucketGraphs> compiled_;
    /// Exact-width mid-chunk graphs for ``bucket == prefill_chunk_size`` (dual capture).
    std::unordered_map<size_t, BucketGraphs> compiled_mid_;
    /// MIXED mid+decode graphs for ``bucket == prefill_chunk_size`` (n_req=2, includes lm_head).
    std::unordered_map<size_t, BucketGraphs> compiled_mixed_mid_;
    /// Capture width used for ``compiled_mixed_mid_`` (normally 2).
    size_t mixed_mid_capture_req_{2};
    size_t segment_replays_{0};
    size_t prefill_hits_{0};
    size_t prefill_misses_{0};
    bool last_prefill_executed_{false};
};

} // namespace infinilm::engine
