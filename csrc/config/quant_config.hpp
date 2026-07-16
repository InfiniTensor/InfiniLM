#pragma once
#include "../layers/quantization/quantization.hpp"
#include "../utils.hpp"
#include "nlohmann/json.hpp"
#include <optional>
#include <stdexcept>

namespace infinilm::config {

class QuantConfig {
    // QuantConfig is used to store and parse the "quantization" field from config.json.
    // This is currently a basic version and will be extended in the future.
public:
    QuantConfig() = default;
    QuantConfig(const nlohmann::json &json);

    std::shared_ptr<infinilm::quantization::BaseQuantization> get_quantization_method() const;

    infinilm::quantization::QuantScheme get_quant_scheme() const {
        if (quantization_method != nullptr) {
            return quantization_method->get_quant_scheme();
        } else {
            return infinilm::quantization::QuantScheme::NONE;
        }
    }

    void set_kv_quant_scheme(infinicore::DataType kv_cache_dtype) {
        throw std::runtime_error(
            "KV cache INT8 quantization is unsupported until its kernels are available in InfiniOps; requested dtype `"
            + infinicore::toString(kv_cache_dtype) + "`.");
    }

    infinilm::quantization::KVQuantAlgo get_kv_quant_scheme() const {
        return kv_quant_scheme;
    }

    std::optional<infinicore::DataType> get_kv_cache_dtype() const {
        if (this->kv_cache_dtype_.has_value()) {
            return this->kv_cache_dtype_;
        }
        return std::nullopt;
    }

private:
    nlohmann::json quantization_config;
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization_method;

    infinilm::quantization::KVQuantAlgo kv_quant_scheme = infinilm::quantization::KVQuantAlgo::NONE;
    std::optional<infinicore::DataType> kv_cache_dtype_ = std::nullopt;
};

} // namespace infinilm::config
