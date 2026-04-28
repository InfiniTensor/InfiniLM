#pragma once

#include "base_quantization.hpp"
namespace infinicore::quantization {

class NoneQuantization : public BaseQuantization {
    // This is a temporary class that currently only returns COMPRESSED_TENSOR_W8A8I8.
    // Future enhancements should parse quant_config to extract detailed quantization
    // information and support multiple quantization schemes.
public:
    explicit NoneQuantization(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config){};

    infinicore::quantization::QuantScheme
    get_quant_scheme() const override {
        return infinicore::quantization::QuantScheme::NONE;
    };
};

} // namespace infinicore::quantization
