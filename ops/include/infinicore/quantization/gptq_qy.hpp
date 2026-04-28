#pragma once
#include "base_quantization.hpp"
namespace infinicore::quantization {

class GPTQ_QY : public BaseQuantization {
    // This is a temporary class that currently only returns GPTQ W4A16.
    // Future enhancements should parse quant_config to extract detailed quantization
    // information and support multiple quantization schemes.
public:
    explicit GPTQ_QY(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {};

    infinicore::quantization::QuantScheme
    get_quant_scheme() const override {
        return infinicore::quantization::QuantScheme::GPTQ_W4A16_QY;
    };

    int get_packing_num() const {
        // For GPTQ, we pack 8 int4 weights into a single int32 value.
        return 32 / this->get_or<int>("bits", 4); // Default to 8 if not specified in config
    }

    int get_group_size() const {
        // For simplicity, we return a fixed group size here. In a more complete implementation,
        // this could be extracted from quant_config_ to support different group sizes.
        return this->get_or<int>("group_size", 128); // Standard GPTQ group size
    }
};

} // namespace infinicore::quantization
