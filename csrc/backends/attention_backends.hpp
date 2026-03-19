#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace infinilm::backends {

enum class AttentionBackend {
    StaticAttn,
    PagedAttn,
    FlashAttn,
    FlashInferAttn,
    Default = StaticAttn // 与 StaticAttn 共享同一底层值（0）
};

inline std::ostream &operator<<(std::ostream &os, AttentionBackend backend) {
    switch (backend) {
    case AttentionBackend::StaticAttn: // Default 与 StaticAttn 共享底层值 0
        return os << "AttentionBackend::StaticAttn";
    case AttentionBackend::PagedAttn:
        return os << "AttentionBackend::PagedAttn";
    case AttentionBackend::FlashAttn:
        return os << "AttentionBackend::FlashAttn";
    default:
        throw std::invalid_argument("Invalid attention backend: " + std::to_string(static_cast<int>(backend)));
        break;
    }
}

inline AttentionBackend parse_attention_backend(const std::string &backend) {
    if ("default" == backend) {
        return AttentionBackend::Default;
    }
    if ("static-attn" == backend) {
        return AttentionBackend::StaticAttn;
    }
    if ("paged-attn" == backend) {
        return AttentionBackend::PagedAttn;
    }
    if ("flash-attn" == backend) {
        return AttentionBackend::FlashAttn;
    }

    throw std::invalid_argument(
        "Invalid attention_backend: " + backend + ". Valid options are: default, flash-attn");
}

} // namespace infinilm::backends
