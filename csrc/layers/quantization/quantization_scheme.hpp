#pragma once

namespace infinilm::quantization {

enum class QuantScheme {
    NONE,
    COMPRESSED_TENSOR_W8A8I8,
    AWQ_W4A16,
    AWQ_MARLIN_W4A16,
    GPTQ_W4A16_QY,
    GPTQ_W4A16,
    GPTQ_MARLIN_W4A16,
};

enum class KVQuantAlgo {
    NONE,
    INT8,
};

} // namespace infinilm::quantization
