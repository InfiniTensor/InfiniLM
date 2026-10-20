#pragma once
#include "weights/weights.hpp"
#include "layers/normalization/rmsnorm.hpp"
#include "deepseek_v4_paged_latent_cache.hpp"
#include <nlohmann/json.hpp>

namespace infinilm::models::deepseek_v4 {
// Real checkpoint compressor. D512 owns the public V4 paged latent cache;
// D128 exposes features for the indexer (its distinct cache is not implemented here).
class TemporalCompressor final : public weights::ModelWeights {
public:
    struct Request { size_t row, past, tokens; };
    struct Batch {
        runtime::Tensor score, pooled, rotated;
        std::vector<int64_t> boundaries, slots;
    private:
        friend class TemporalCompressor;
        runtime::Tensor next_state;
        std::vector<Request> requests;
        const TemporalCompressor *owner = nullptr;
        size_t revision = 0;
    };
    TemporalCompressor(runtime::Runtime &, const nlohmann::json &, size_t layer,
                       size_t dim, size_t rows, size_t compressed_capacity);
    void process_weights_after_loading(runtime::Runtime &, bool) override;
    Batch prepare(runtime::Runtime &, const runtime::Tensor &hidden,
                  const runtime::Tensor &positions, const std::vector<Request> &) const;
    // The C4 indexer fuses weighted norm/RoPE/Hadamard in F32 before FP8.
    Batch prepare_pooled(runtime::Runtime &, const runtime::Tensor &hidden,
                         const runtime::Tensor &positions, const std::vector<Request> &) const;
    void commit(runtime::Runtime &, Batch &);
    Batch forward(runtime::Runtime &, const runtime::Tensor &, const runtime::Tensor &,
                  const std::vector<Request> &);
    runtime::Tensor gather(runtime::Runtime &, const std::vector<std::pair<size_t,size_t>> &) const;
    void reset(runtime::Runtime &);
    const runtime::Tensor &state() const { return state_; }
    runtime::Tensor raw_cache() const { return cache_ ? cache_->raw() : runtime::Tensor{}; }
    size_t dim() const { return dim_; }
    size_t ratio() const { return ratio_; }
private:
    Batch prepare_impl(runtime::Runtime &, const runtime::Tensor &, const runtime::Tensor &,
                       const std::vector<Request> &, bool postprocess) const;
    void check_owner(runtime::Runtime &) const;
    void rotate(runtime::Runtime &, runtime::Tensor, const std::vector<int64_t> &) const;
    runtime::Runtime *owner_;
    size_t dim_, rows_, ratio_, channels_, maximum_, capacity_, revision_ = 0;
    double theta_, factor_, beta_fast_, beta_slow_;
    size_t original_;
    bool ready_ = false;
    runtime::Tensor wkv_, wgate_, ape_, packed_ape_, state_;
    std::shared_ptr<layers::RMSNorm> norm_;
    std::unique_ptr<PagedLatentCache> cache_;
    std::vector<size_t> lengths_;
};
}
