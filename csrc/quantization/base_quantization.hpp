// #pragma once
// #include "../config/quant_config.hpp"
// #include "infinicore/nn/quantization.hpp"
// #include "nlohmann/json.hpp"

// namespace infinilm::quantization {
// class BaseQuantization {
//     // Base class for quantization schemes. Intended to be extended to support various quantization methods.
// public:
//     explicit BaseQuantization(const nlohmann::json &quant_config) : quant_config_(quant_config) {};
//     virtual ~BaseQuantization() = default;

//     virtual infinicore::nn::QuantScheme get_quant_scheme() const = 0;

// protected:
//     nlohmann::json quant_config_;
// };
// } // namespace infinilm::quantization
