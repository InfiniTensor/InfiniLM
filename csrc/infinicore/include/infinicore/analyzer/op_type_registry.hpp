#pragma once

#include "op_type.hpp"
#include <string>
#include <unordered_map>

namespace infinicore::analyzer {

/// Centralized class-name → OpType registry.
/// New ops only need one line added here — no changes to op headers.
inline OpType opTypeFromName(const char *name) {
    static const std::unordered_map<std::string, OpType> registry = {
        // Attention
        {"FlashAttention", OpType::FLASH_ATTENTION},
        {"CausalSoftmax", OpType::CAUSAL_SOFTMAX},
        {"PagedAttention", OpType::PAGED_ATTENTION},
        {"MhaKVCache", OpType::MHA_KVCACHE},
        {"MultiheadAttentionVarlen", OpType::MHA_VARLEN},
        // GEMM / MLP
        {"Gemm", OpType::GEMM},
        {"I8Gemm", OpType::SCALED_MM_I8},
        // Activation
        {"SiluAndMul", OpType::SILU_AND_MUL},
        {"SwiGLU", OpType::SWIGLU},
        // Norm
        {"RMSNorm", OpType::RMS_NORM},
        {"AddRMSNorm", OpType::ADD_RMS_NORM},
        // Embedding / Positional
        {"Embedding", OpType::EMBEDDING},
        {"RoPE", OpType::ROPE},
        // KV Cache
        {"KVCaching", OpType::KV_CACHING},
        {"PagedCaching", OpType::PAGED_CACHING},
        // Elementwise
        {"Add", OpType::ADD},
        {"Mul", OpType::MUL},
        // Quantization
        {"PerTensorQuantI8", OpType::PER_TENSOR_QUANT_I8},
        {"PerTensorDequantI8", OpType::PER_TENSOR_DEQUANT_I8},
        {"PerChannelQuantI8", OpType::PER_CHANNEL_QUANT_I8},
        // Misc
        {"Rearrange", OpType::REARRANGE},
    };
    auto it = registry.find(name);
    return it != registry.end() ? it->second : OpType::UNKNOWN;
}

} // namespace infinicore::analyzer
