// quant.hpp
#pragma once

namespace infinicore::quantization {

enum class QuantScheme {
    NONE,
    COMPRESSED_TENSOR_W8A8I8,
    AWQ_W4A16,
    GPTQ_W4A16_QY,
};

enum class KVQuantAlgo {
    NONE,
    INT8,
};

} // namespace infinicore::quantization
