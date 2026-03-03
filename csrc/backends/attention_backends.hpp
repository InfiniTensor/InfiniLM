#pragma once

#include <stdexcept>
#include <string>

namespace infinilm::backends {

enum class AttentionBackend {
    Default,
    FlashAttn,
};

inline AttentionBackend parse_attention_backend(const std::string &backend) {
    if (backend == "default") {
        return AttentionBackend::Default;
    }
    if (backend == "flash-attn") {
        return AttentionBackend::FlashAttn;
    }

    throw std::invalid_argument(
        "Invalid attention_backend: " + backend + ". Valid options are: default, flash-attn");
}

} // namespace infinilm::backends
