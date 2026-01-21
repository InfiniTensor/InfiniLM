#pragma once
// #include "../config/global_config.hpp"
#include "../config/quant_config.hpp"
#include "quantization.hpp"
// #include "utils.hpp"
namespace infinilm::quantization {

class CompressedTensors : public BaseQuantization {
public:
    explicit CompressedTensors(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {
              // quant_config_ = global_config.get_quant_config_json();
          };

    infinicore::nn::QuantScheme
    get_quant_scheme() const override {
        return infinicore::nn::QuantScheme::COMPRESSED_TENSOR_W8A8I8;
    };
};

} // namespace infinilm::quantization