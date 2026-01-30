#pragma once

#include "../config/quant_config.hpp"
#include "base_quantization.hpp"
namespace infinilm::quantization {

class CompressedTensors : public BaseQuantization {
    // This is a temporary class that currently only returns COMPRESSED_TENSOR_W8A8I8.
    // Future enhancements should parse quant_config to extract detailed quantization
    // information and support multiple quantization schemes.
public:
    explicit CompressedTensors(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {};

    infinicore::nn::QuantScheme
    get_quant_scheme() const override {
        return infinicore::nn::QuantScheme::COMPRESSED_TENSOR_W8A8I8;
    };
};

} // namespace infinilm::quantization
