#pragma once
#include "compressed_tensors.hpp"

// #include "../config/quant_config.hpp"
#include "../config/global_config.hpp"
#include "infinicore/nn/quantization.hpp"

namespace infinilm::quantization {
class BaseQuantization {
public:
    explicit BaseQuantization(const infinilm::config::global_config::GlobalConfig &global_config) {};
    virtual ~BaseQuantization() = default;

    virtual infinicore::nn::QuantScheme get_quant_scheme() const = 0;

protected:
    infinilm::config::quantization::QuantConfig quant_config_;
}
} // namespace infinilm::quantization