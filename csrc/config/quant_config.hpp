#pragma once
#include "../utils.hpp"
#include "infinicore/quantization.hpp"
#include "nlohmann/json.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::config {

class QuantConfig {
    // QuantConfig is used to store and parse the "quantization" field from config.json.
    // This is currently a basic version and will be extended in the future.
public:
    QuantConfig() = default;
    QuantConfig(const nlohmann::json &json);

    std::shared_ptr<infinicore::quantization::BaseQuantization> get_quantization_method() const;

    infinicore::quantization::QuantScheme get_quant_scheme() const {
        if (quantization_method != nullptr) {
            return quantization_method->get_quant_scheme();
        } else {
            return infinicore::quantization::QuantScheme::NONE;
        }
    }

    void set_kv_quant_scheme(std::string kv_cache_dtype) {
        if (kv_cache_dtype.empty()) {
            this->kv_quant_scheme = infinicore::quantization::KVQuantAlgo::NONE;
            // spdlog::debug("kv_cache_dtype is empty, using default NONE");
            return;
        }

        try {
            switch (parse_dtype(kv_cache_dtype)) {
            case infinicore::DataType::I8: {
                this->kv_quant_scheme = infinicore::quantization::KVQuantAlgo::INT8;
                break;
            }
            default: {
                spdlog::warn("Unsupported kv_cache_dtype: '{}', fallback to NONE", kv_cache_dtype);
                this->kv_quant_scheme = infinicore::quantization::KVQuantAlgo::NONE;
                break;
            }
            }
        } catch (const std::exception &e) {
            spdlog::error("Failed to parse kv_cache_dtype '{}': {}", kv_cache_dtype, e.what());
            this->kv_quant_scheme = infinicore::quantization::KVQuantAlgo::NONE;
        }
    }

    infinicore::quantization::KVQuantAlgo get_kv_quant_scheme() const {
        return kv_quant_scheme;
    }

private:
    nlohmann::json quantization_config;
    std::shared_ptr<infinicore::quantization::BaseQuantization> quantization_method;
    infinicore::quantization::KVQuantAlgo kv_quant_scheme = infinicore::quantization::KVQuantAlgo::NONE;
};

} // namespace infinilm::config
