#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace infinilm::backends {

/**
 * @brief Enumeration of all supported attention backends.
 */
enum class AttentionBackend {
    STATIC_ATTN,
    PAGED_ATTN,
    FLASH_ATTN,
    FLASHINFER,
    Default = STATIC_ATTN
};

inline std::ostream &operator<<(std::ostream &os, AttentionBackend backend) {
    switch (backend) {
    case AttentionBackend::STATIC_ATTN:
        return os << "AttentionBackend::STATIC_ATTN";
    case AttentionBackend::PAGED_ATTN:
        return os << "AttentionBackend::PAGED_ATTN";
    case AttentionBackend::FLASH_ATTN:
        return os << "AttentionBackend::FLASH_ATTN";
    case AttentionBackend::FLASHINFER:
        return os << "AttentionBackend::FLASHINFER";
    default:
        throw std::invalid_argument("infinilm::backends: invalid attention backend: " + std::to_string(static_cast<int>(backend)));
        break;
    }
}

inline AttentionBackend parse_attention_backend(const std::string &backend) {
    if (backend == "default") {
        return AttentionBackend::Default;
    }
    if (backend == "static-attn") {
        return AttentionBackend::STATIC_ATTN;
    }
    if (backend == "paged-attn") {
        return AttentionBackend::PAGED_ATTN;
    }
    if (backend == "flash-attn") {
        return AttentionBackend::FLASH_ATTN;
    }
    if (backend == "flashinfer") {
        return AttentionBackend::FLASHINFER;
    }

    throw std::invalid_argument(
        "Invalid attention_backend: " + backend + ". Valid options are: static-attn, paged-attn, flash-attn, flashinfer");
}

} // namespace infinilm::backends
