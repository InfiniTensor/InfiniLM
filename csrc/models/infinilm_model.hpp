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

        /// Optional model-level KV cache for incremental decoding. Defaults to `nullptr`.
        void *kv_cache = nullptr;
    };

    struct Output {
        /// Output tensor of shape [batch, seq_len, vocab_size].
        infinicore::Tensor logits;
    };

    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input) const = 0;
    // Optional: reset cache; default no-op for models without cache
    virtual void reset_cache(size_t pos = 0) {}
    virtual void reset_cache(const cache::CacheConfig &new_config, size_t pos = 0) = 0;
};
} // namespace infinilm
