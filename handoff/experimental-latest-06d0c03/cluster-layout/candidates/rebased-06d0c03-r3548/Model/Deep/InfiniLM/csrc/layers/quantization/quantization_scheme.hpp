#pragma once

namespace infinilm::quantization {

enum class QuantScheme {
    NONE,
    COMPRESSED_TENSOR_W8A8I8,
    AWQ_W4A16,
    GPTQ_W4A16,
    MODEL_NATIVE,
};

enum class KVQuantAlgo {
    NONE,
    INT8,
};

} // namespace infinilm::quantization
