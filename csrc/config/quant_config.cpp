#include "quant_config.hpp"
#include <stdexcept>

namespace infinilm::config {
QuantConfig::QuantConfig(const nlohmann::json &json) : quantization_config(json) {
    this->quantization_method = get_quantization_method();
}

std::shared_ptr<infinilm::quantization::BaseQuantization>
QuantConfig::get_quantization_method() const {
    if (quantization_config.is_null()) {
        return std::make_shared<infinilm::quantization::NoneQuantization>(quantization_config); // Default case if no matching scheme
    }

    if (!quantization_config.is_object() || !quantization_config.contains("quant_method")
        || !quantization_config.at("quant_method").is_string()
        || quantization_config.at("quant_method").get<std::string>().empty()) {
        throw std::invalid_argument(
            "quantization_config must be null or an object with a nonempty string quant_method");
    }

    const std::string quant_method = quantization_config.at("quant_method").get<std::string>();

    if (quant_method == "compressed-tensors") {
        throw std::runtime_error(
            "`compressed-tensors` quantization is unsupported because InfiniLM lacks the end-to-end activation quantization and scaled-GEMM integration.");
    } else if (quant_method == "awq" || quant_method == "gptq") {
        const bool awq = quant_method == "awq";
        const std::string name = awq ? "AWQ" : "GPTQ";
        const auto require_value = [&](const char *key, const char *alias, int expected) {
            if (!quantization_config.contains(key) && !quantization_config.contains(alias))
                throw std::invalid_argument(name + " requires explicit " + key);
            for (const auto *field : {key, alias}) {
                if (quantization_config.contains(field)
                    && (!quantization_config.at(field).is_number_integer()
                        || quantization_config.at(field).get<int64_t>() != expected))
                    throw std::invalid_argument(name + " unsupported or conflicting " + field);
            }
        };
        require_value("bits", "w_bit", 4);
        require_value("group_size", "q_group_size", 128);
        if (awq && (!quantization_config.contains("version")
                    || (quantization_config.at("version") != "GEMM" && quantization_config.at("version") != "gemm")
                    || !quantization_config.contains("zero_point")
                    || quantization_config.at("zero_point") != true))
            throw std::invalid_argument("AWQ requires GEMM packing and zero_point=true");
        if (!awq && (!quantization_config.contains("sym") || quantization_config.at("sym") != true
                     || !quantization_config.contains("checkpoint_format")
                     || quantization_config.at("checkpoint_format") != "gptq"
                     || !quantization_config.contains("desc_act") || quantization_config.at("desc_act") != false))
            throw std::invalid_argument("GPTQ requires sym=true, checkpoint_format=gptq and desc_act=false");
        if (quantization_config.contains("modules_to_not_convert")
            && !quantization_config.at("modules_to_not_convert").is_null()
            && quantization_config.at("modules_to_not_convert") != nlohmann::json::array()
            && quantization_config.at("modules_to_not_convert") != nlohmann::json::array({"lm_head"}))
            throw std::invalid_argument(name + " only permits the existing dense lm_head exemption");
        for (const char *key : {"modules_to_not_quantize", "modules_to_skip", "skip_modules",
                                "modules_to_fuse", "exllama_config", "packing_format"}) {
            if (quantization_config.contains(key) && !quantization_config.at(key).is_null()
                && !quantization_config.at(key).empty())
                throw std::invalid_argument(name + " custom configuration is unsupported: " + key);
        }
        if (!awq && quantization_config.contains("modules_in_block_to_quantize")
            && !quantization_config.at("modules_in_block_to_quantize").is_null()
            && !quantization_config.at("modules_in_block_to_quantize").empty())
            throw std::invalid_argument("GPTQ custom configuration is unsupported: modules_in_block_to_quantize");
        if ((quantization_config.contains("do_fuse") && quantization_config.at("do_fuse") != false)
            || (quantization_config.contains("quantize_lm_head") && quantization_config.at("quantize_lm_head") != false)
            || (awq && quantization_config.contains("checkpoint_format")
                && quantization_config.at("checkpoint_format") != "awq"
                && quantization_config.at("checkpoint_format") != "GEMM"
                && quantization_config.at("checkpoint_format") != "gemm"))
            throw std::invalid_argument(name + " fused/custom packing or quantized lm_head is unsupported");
        if (awq) return std::make_shared<infinilm::quantization::AWQ>(quantization_config);
        return std::make_shared<infinilm::quantization::GPTQ>(quantization_config);
    } else if (quant_method == "quark") {
        throw std::runtime_error(
            "MXFP4 quantization is unsupported because InfiniLM has no InfiniOps-backed MXFP4 execution path.");
    } else {
        throw std::invalid_argument("Unsupported quantization method: " + quant_method);
}
}
} // namespace infinilm::config
