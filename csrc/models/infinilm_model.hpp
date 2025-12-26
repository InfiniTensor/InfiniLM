#pragma once

#include "infinicore/nn/module.hpp"

#include "../cache/cache.hpp"

#include <any>

#include <optional>

namespace infinilm {
class InfinilmModel : public infinicore::nn::Module {
public:
    struct Config {
        std::string model_type;

        virtual ~Config() = default;
    };

    struct Input {
        /// Token IDs tensor of shape `[batch, seq_len]`.
        std::optional<infinicore::Tensor> input_ids;
        /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
        std::optional<infinicore::Tensor> position_ids;
        /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> cache_lengths;
        /// Input Lengths of each request in a continous-batched sequence, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> input_lengths;
        /// Offsets of each request in a continous-batched sequence, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> input_offsets;
        /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
        std::optional<infinicore::Tensor> block_tables;
        /// Slot ids for each token `[seq]`. Used for paged cache.
        std::optional<infinicore::Tensor> slot_mapping;

        float random_val{0.1};

        float topp{0.8};

        int topk{1};

        float temperature{1};
    };

    struct Output {
        /// Output token IDs.
        infinicore::Tensor output_ids;
    };

    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input) const = 0;

    virtual void reset_cache(const cache::CacheConfig *cache_config) = 0;
};
} // namespace infinilm
