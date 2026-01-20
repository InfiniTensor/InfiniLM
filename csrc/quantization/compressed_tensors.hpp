#pragma once
#include "quantization.hpp"
// #include "utils.hpp"
namespace infinilm::quantization {

class CompressedTensors : public BaseQuantization {
public:
    CompressedTensors(const infinilm::config::global_config::GlobalConfig &global_config)
        : BaseQuantization(global_config) {
        quant_config_ = global_config.get_quant_config_json();
    }

    infinicore::nn::QuantScheme
    get_quant_scheme() const override {
        return infinicore::nn::QuantScheme::COMPRESSED_TENSOR_W8A8I8;
    }
};

} // namespace infinilm::quantization