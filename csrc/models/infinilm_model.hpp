#pragma once

#include "infinicore/nn/module.hpp"

#include "../cache/cache.hpp"

#include <any>
#include <cstddef>

#include <optional>

namespace infinilm {
namespace cache {
struct KVCompressionConfig;
} // namespace cache

class InfinilmModel : public infinicore::nn::Module {
public:
    struct Config {
        std::string model_type;

        virtual ~Config() = default;
    };

    struct Input {
        /// Token IDs tensor of shape `[batch, seq_len]`.
        std::optional<infinicore::Tensor> input_ids;
        /// Image pixel values for multi-modal models.
        /// Shape is model-specific (e.g. LLaVA: [batch, 3, H, W], MiniCPM-V: [batch, 3, patch, seq_len * patch]).
        std::optional<infinicore::Tensor> pixel_values;
        /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
        std::optional<infinicore::Tensor> position_ids;
        /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> past_sequence_lengths;
        /// ToTal Lengths for each request sequence, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> total_sequence_lengths;
        /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
        std::optional<infinicore::Tensor> input_offsets;
        /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
        std::optional<infinicore::Tensor> block_tables;
        /// Slot ids for each token `[seq]`. Used for paged cache.
        std::optional<infinicore::Tensor> slot_mapping;
        /// Image placeholder bounds for MiniCPM-V style replacement.
        /// Tensor shape: [batch, max_ranges, 2] (start, end).
        std::optional<infinicore::Tensor> image_bound;
        /// Target patch sizes for each image (MiniCPM-V).
        /// Tensor shape: [batch, 2] or [batch, max_slices, 2] if pre-flattened.
        std::optional<infinicore::Tensor> tgt_sizes;
    };

    struct Output {
        /// Logits.
        infinicore::Tensor logits;
    };

    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input) const = 0;

    virtual void reset_cache(const cache::CacheConfig *cache_config) = 0;

    virtual uint32_t compress_kv_cache_inplace(uint32_t seq_len,
                                               size_t batch_size,
                                               const cache::KVCompressionConfig &cfg) {
        (void)seq_len;
        (void)batch_size;
        (void)cfg;
        return seq_len;
    }
};
} // namespace infinilm
