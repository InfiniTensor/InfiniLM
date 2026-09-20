#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace infinilm::backends {

/**
 * @brief Enumeration of all supported attention backends.
 *
 * 各后端说明：
 * - STATIC_ATTN：静态 attention，prefill/decode 走同一套实现（默认）
 * - PAGED_ATTN：自研 paged-attention，prefill 为 PagedAttentionPrefill，decode 为 splitkv
 * - FLASH_ATTN：FlashAttention-2（mha_varlen_fwd / mha_fwd_kvcache）
 * - FLASHINFER：FlashInfer 后端
 * - HYBRID：按阶段分离路由——prefill 走 FA2 varlen，decode 走自研 paged kernel
 *   （见 HybridAttentionImpl）
 */
enum class AttentionBackend {
    STATIC_ATTN,
    PAGED_ATTN,
    FLASH_ATTN,
    FLASHINFER,
    HYBRID, // prefill → FlashAttention (FA2 varlen), decode → PagedAttention
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
    case AttentionBackend::HYBRID:
        return os << "AttentionBackend::HYBRID";
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
    if (backend == "hybrid") {
        // "hybrid"：prefill→FA2 varlen，decode→自研 paged-attention（splitkv）
        return AttentionBackend::HYBRID;
    }

    throw std::invalid_argument(
        "Invalid attention_backend: " + backend + ". Valid options are: static-attn, paged-attn, flash-attn, flashinfer, hybrid");
}

} // namespace infinilm::backends
