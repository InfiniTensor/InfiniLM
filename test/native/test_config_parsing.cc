#include "config/config_factory.hpp"
#include "config/quant_config.hpp"
#include "models/models_registry.hpp"

#include <cassert>
#include <memory>
#include <stdexcept>

int main() {
    using infinicore::DataType;
    using infinilm::config::ConfigFactory;
    using infinilm::config::ModelConfig;
    using infinilm::config::QuantConfig;
    using infinilm::quantization::KVQuantAlgo;
    using infinilm::quantization::QuantScheme;

    bool processed = false;
    infinilm::models::register_model_config(
        "test_registered_model", [&](std::shared_ptr<ModelConfig> config) {
            processed = true;
            config->get_config_json()["normalized"] = true;
            return config;
        });
    const auto model = ConfigFactory::createConfig(
        R"({"model_type":"test_registered_model","hidden_size":128})");
    assert(processed);
    assert(model->get<bool>("normalized"));
    assert(model->get<size_t>("hidden_size") == 128);

    bool rejected = false;
    try {
        ConfigFactory::createConfig(R"({"model_type":"test_unregistered_model"})");
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    const auto awq_json = nlohmann::json::parse(
        R"({"quant_method":"awq","bits":4,"group_size":64})");
    QuantConfig awq(awq_json);
    assert(awq.get_quant_scheme() == QuantScheme::AWQ_W4A16);
    const auto awq_method = std::dynamic_pointer_cast<infinilm::quantization::AWQ>(
        awq.get_quantization_method());
    assert(awq_method && awq_method->get_group_size() == 64);
    assert(awq_method->get_packing_num() == 8);

    QuantConfig gptq(nlohmann::json::parse(
        R"({"quant_method":"gptq","bits":4,"group_size":32})"));
    assert(gptq.get_quant_scheme() == QuantScheme::GPTQ_W4A16);
    const auto gptq_method = std::dynamic_pointer_cast<infinilm::quantization::GPTQ>(
        gptq.get_quantization_method());
    assert(gptq_method && gptq_method->get_group_size() == 32);
    assert(gptq_method->get_packing_num() == 8);

    QuantConfig compressed(nlohmann::json::parse(
        R"({"quant_method":"compressed-tensors"})"));
    assert(compressed.get_quant_scheme() == QuantScheme::COMPRESSED_TENSOR_W8A8I8);
    QuantConfig mxfp4(nlohmann::json::parse(R"({"quant_method":"quark"})"));
    assert(mxfp4.get_quant_scheme() == QuantScheme::MXFP4_W4A16);
    QuantConfig unquantized(nlohmann::json(nullptr));
    assert(unquantized.get_quant_scheme() == QuantScheme::NONE);

    unquantized.set_kv_quant_scheme(DataType::kInt8);
    assert(unquantized.get_kv_quant_scheme() == KVQuantAlgo::INT8);
    assert(unquantized.get_kv_cache_dtype() == DataType::kInt8);
    for (const auto dtype : {DataType::kFloat16, DataType::kBFloat16}) {
        unquantized.set_kv_quant_scheme(dtype);
        assert(unquantized.get_kv_quant_scheme() == KVQuantAlgo::NONE);
        assert(unquantized.get_kv_cache_dtype() == dtype);
    }
    rejected = false;
    try {
        unquantized.set_kv_quant_scheme(DataType::kFloat32);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}
