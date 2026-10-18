#pragma once

#include "graph_compiler.hpp"
#include "../../backends/attention_backends.hpp"
#include "paged_decode_bucket.hpp"

#include <optional>
#include <map>
#include <stdexcept>
#include <utility>

namespace infinilm::engine {

// Preparation policy belongs to the worker, not to the calling thread's TLS.
// It is fixed for a compiler's lifetime; topology/cache changes rebuild it.
struct PagedGraphConfig {
    backends::AttentionBackend attention_backend;
    int tp_size;
    int tp_rank;
    int pp_size; // Explicit caller topology; no default or TLS inference.

    void validate() const {
        if (tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size || pp_size <= 0) {
            throw std::invalid_argument("invalid paged graph TP topology");
        }
    }
};

// One qualification producer shared by worker eager normalization and compiler.
FinitePagedDecodeProfile make_finite_paged_decode_profile(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config,
    bool ordinary_kv_prepare,
    infinilm::runtime::Device::Type device_type,
    const PagedGraphConfig &graph_config);

// Shared legacy short-plan eligibility. The finite FlashKV bucket policy
// must not replace its established input selection with another plan family.
bool uses_short_decode_graph(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config,
    infinilm::runtime::Device::Type device_type,
    const PagedGraphConfig &graph_config);

class PagedCompiler : public GraphCompiler {
public:
    PagedCompiler(const std::shared_ptr<InfinilmModel> &model,
                  RankBarrier *barrier,
                  infinilm::runtime::Runtime &runtime,
                  infinilm::global_state::ForwardContext &forward_context,
                  PagedGraphConfig graph_config);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    const PagedGraphConfig graph_config_;
    std::vector<size_t> decode_batch_sizes_;

    infinilm::runtime::Tensor block_tables_holder_;
    infinilm::runtime::Tensor short_block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
    };

    std::optional<CompiledResult> compiled_short_decode_b1_;
    std::optional<CompiledResult> compiled_baichuan_prefill_b1_s10_;
    std::map<
        std::pair<size_t, size_t>, // (num_requests, logical KV bound; 0=legacy full width)
        CompiledResult>
        compiled_map_decode_;
};
} // namespace infinilm::engine
