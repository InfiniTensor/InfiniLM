#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace infinilm::backends {

/*
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py


class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"

    pass
*/
enum class AttentionBackend {
    STATIC_ATTN,
    PAGED_ATTN,
    FLASH_ATTN,
    FLASHINFER,
    Default = STATIC_ATTN // 与 STATIC_ATTN 共享同一底层值（0）
};

inline std::ostream &operator<<(std::ostream &os, AttentionBackend backend) {
    switch (backend) {
    case AttentionBackend::STATIC_ATTN: // Default 与 STATIC_ATTN 共享底层值 0
        return os << "AttentionBackend::STATIC_ATTN";
    case AttentionBackend::PAGED_ATTN:
        return os << "AttentionBackend::PAGED_ATTN";
    case AttentionBackend::FLASH_ATTN:
        return os << "AttentionBackend::FLASH_ATTN";
    case AttentionBackend::FLASHINFER:
        return os << "AttentionBackend::FLASHINFER";
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
        return AttentionBackend::STATIC_ATTN;
    }
    if ("paged-attn" == backend) {
        return AttentionBackend::PAGED_ATTN;
    }
    if ("flash-attn" == backend) {
        return AttentionBackend::FLASH_ATTN;
    }
    if ("flashinfer-attn" == backend || "flashinfer" == backend) {
        return AttentionBackend::FLASHINFER;
    }

    throw std::invalid_argument(
        "Invalid attention_backend: " + backend + ". Valid options are: default, static-attn, paged-attn, flash-attn, flashinfer-attn");
}

} // namespace infinilm::backends
