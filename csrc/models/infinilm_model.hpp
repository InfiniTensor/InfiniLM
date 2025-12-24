#pragma once

#include "infinicore/nn/module.hpp"

#include "../cache/cache.hpp"

#include <any>

namespace infinilm {
class InfinilmModel : public infinicore::nn::Module {
public:
    struct Config {
        std::string model_type;

        virtual ~Config() = default;
    };

    struct Input {
        /// Token IDs tensor of shape `[batch, seq_len]`.
        infinicore::Tensor input_ids;
        /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
        infinicore::Tensor position_ids;
        /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
        infinicore::Tensor cache_positions;
    };

    struct Output {
        /// Output tensor of shape [batch, seq_len, vocab_size].
        infinicore::Tensor logits;
    };

    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input) const = 0;

    virtual void reset_cache(const cache::CacheConfig *cache_config) = 0;
};
} // namespace infinilm
