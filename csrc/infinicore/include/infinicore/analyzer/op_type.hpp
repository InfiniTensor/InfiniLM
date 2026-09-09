#pragma once

#include <cstdint>
#include <string>

namespace infinicore::analyzer {

/// Op types recognized by the analyzer for phase detection.
/// This is not an exhaustive list of operations, only the ones
/// relevant for phase classification in LLM inference.
enum class OpType : uint8_t {
    UNKNOWN = 0,

    // --- Attention-related ---
    ATTENTION,
    FLASH_ATTENTION,
    CAUSAL_SOFTMAX,
    PAGED_ATTENTION,
    PAGED_ATTENTION_PREFILL,
    MHA_KVCACHE,
    MHA_VARLEN,
    SOFTMAX,

    // --- GEMM / MLP ---
    GEMM,
    LINEAR,
    MATMUL,
    INT8_GEMM,
    SCALED_MM_I8,

    // --- Activation ---
    SILU,
    SILU_AND_MUL,
    GELU,
    SWIGLU,
    RELU,
    SIGMOID,

    // --- Norm ---
    RMS_NORM,
    ADD_RMS_NORM,
    LAYER_NORM,

    // --- Embedding / Positional ---
    EMBEDDING,
    ROPE,

    // --- KV Cache ---
    KV_CACHING,
    PAGED_CACHING,

    // --- Elementwise / Reduce ---
    ADD,
    MUL,
    SUB,
    SUM,
    RECIPROCAL,

    // --- Quantization ---
    PER_TENSOR_QUANT_I8,
    PER_TENSOR_DEQUANT_I8,
    PER_CHANNEL_QUANT_I8,

    // --- Sampling ---
    RANDOM_SAMPLE,
    TOPK,
    TOPK_ROUTER,
    TOPK_SOFTMAX,

    // --- Communication (future) ---
    ALLREDUCE,

    // --- Misc ---
    REARRANGE,
    ONES,
    ZEROS,
    TAKE,

    OP_TYPE_COUNT,
};

/// Convert OpType to human-readable string.
inline const char *opTypeToString(OpType type) {
    switch (type) {
    case OpType::ATTENTION:
        return "attention";
    case OpType::FLASH_ATTENTION:
        return "flash_attention";
    case OpType::CAUSAL_SOFTMAX:
        return "causal_softmax";
    case OpType::PAGED_ATTENTION:
        return "paged_attention";
    case OpType::PAGED_ATTENTION_PREFILL:
        return "paged_attention_prefill";
    case OpType::MHA_KVCACHE:
        return "mha_kvcache";
    case OpType::MHA_VARLEN:
        return "mha_varlen";
    case OpType::SOFTMAX:
        return "softmax";
    case OpType::GEMM:
        return "gemm";
    case OpType::LINEAR:
        return "linear";
    case OpType::MATMUL:
        return "matmul";
    case OpType::INT8_GEMM:
        return "int8_gemm";
    case OpType::SCALED_MM_I8:
        return "scaled_mm_i8";
    case OpType::SILU:
        return "silu";
    case OpType::SILU_AND_MUL:
        return "silu_and_mul";
    case OpType::GELU:
        return "gelu";
    case OpType::SWIGLU:
        return "swiglu";
    case OpType::RELU:
        return "relu";
    case OpType::SIGMOID:
        return "sigmoid";
    case OpType::RMS_NORM:
        return "rms_norm";
    case OpType::ADD_RMS_NORM:
        return "add_rms_norm";
    case OpType::LAYER_NORM:
        return "layer_norm";
    case OpType::EMBEDDING:
        return "embedding";
    case OpType::ROPE:
        return "rope";
    case OpType::KV_CACHING:
        return "kv_caching";
    case OpType::PAGED_CACHING:
        return "paged_caching";
    case OpType::ADD:
        return "add";
    case OpType::MUL:
        return "mul";
    case OpType::SUB:
        return "sub";
    case OpType::SUM:
        return "sum";
    case OpType::RECIPROCAL:
        return "reciprocal";
    case OpType::PER_TENSOR_QUANT_I8:
        return "per_tensor_quant_i8";
    case OpType::PER_TENSOR_DEQUANT_I8:
        return "per_tensor_dequant_i8";
    case OpType::PER_CHANNEL_QUANT_I8:
        return "per_channel_quant_i8";
    case OpType::RANDOM_SAMPLE:
        return "random_sample";
    case OpType::TOPK:
        return "topk";
    case OpType::TOPK_ROUTER:
        return "topk_router";
    case OpType::TOPK_SOFTMAX:
        return "topk_softmax";
    case OpType::ALLREDUCE:
        return "allreduce";
    case OpType::REARRANGE:
        return "rearrange";
    case OpType::ONES:
        return "ones";
    case OpType::ZEROS:
        return "zeros";
    case OpType::TAKE:
        return "take";
    default:
        return "unknown";
    }
}

/// Check if an op type belongs to the attention family.
inline bool isAttentionOp(OpType type) {
    switch (type) {
    case OpType::ATTENTION:
    case OpType::FLASH_ATTENTION:
    case OpType::CAUSAL_SOFTMAX:
    case OpType::PAGED_ATTENTION:
    case OpType::PAGED_ATTENTION_PREFILL:
    case OpType::MHA_KVCACHE:
    case OpType::MHA_VARLEN:
        return true;
    default:
        return false;
    }
}

/// Check if an op type belongs to the GEMM/MLP family.
inline bool isGemmMlpOp(OpType type) {
    switch (type) {
    case OpType::GEMM:
    case OpType::LINEAR:
    case OpType::MATMUL:
    case OpType::INT8_GEMM:
    case OpType::SCALED_MM_I8:
        return true;
    default:
        return false;
    }
}

/// Check if an op type is an activation function.
inline bool isActivationOp(OpType type) {
    switch (type) {
    case OpType::SILU:
    case OpType::SILU_AND_MUL:
    case OpType::GELU:
    case OpType::SWIGLU:
    case OpType::RELU:
    case OpType::SIGMOID:
        return true;
    default:
        return false;
    }
}

/// Check if an op type is KV cache related.
inline bool isKvCacheOp(OpType type) {
    switch (type) {
    case OpType::KV_CACHING:
    case OpType::PAGED_CACHING:
        return true;
    default:
        return false;
    }
}

} // namespace infinicore::analyzer
