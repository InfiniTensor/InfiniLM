#pragma once

#include "base_quantization.hpp"
namespace infinicore::quantization {

class CompressedTensors : public BaseQuantization {
    // This is a temporary class that currently only returns COMPRESSED_TENSOR_W8A8I8.
    // Future enhancements should parse quant_config to extract detailed quantization
    // information and support multiple quantization schemes.
public:
    explicit CompressedTensors(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config){};

    infinicore::quantization::QuantScheme
    get_quant_scheme() const override {
        return infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8;
    };
};

} // namespace infinicore::quantization
