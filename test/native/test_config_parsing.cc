#include "config/config_factory.hpp"
#include "config/quant_config.hpp"
#include "models/models_registry.hpp"

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void expect_unsupported(const infinilm::quantization::BaseQuantization &quantization,
                        const std::string &scheme) {
    bool rejected = false;
    try {
        quantization.forward({}, {}, false);
    } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        rejected = message.find(scheme) != std::string::npos
                && message.find("unsupported") != std::string::npos;
    }
    assert(rejected);
}

} // namespace

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

    infinilm::quantization::AWQMarlin awq_marlin(awq_json, 128, 64);
    assert(awq_marlin.get_quant_scheme() == QuantScheme::AWQ_MARLIN_W4A16);
    assert(awq_marlin.get_config() == awq_json);
    assert(awq_marlin.get_param_layout(128, 64, -1, 0, 1, -1, DataType::kFloat16, false).empty());
    awq_marlin.reset_runtime_state();
    expect_unsupported(awq_marlin, "AWQ Marlin");

    infinilm::quantization::GPTQMarlin gptq_marlin(gptq_method->get_config(), 128, 64, true);
    assert(gptq_marlin.get_quant_scheme() == QuantScheme::GPTQ_MARLIN_W4A16);
    assert(gptq_marlin.get_param_layout(128, 64, -1, 0, 1, -1, DataType::kFloat16, false).empty());
    gptq_marlin.reset_runtime_state();
    expect_unsupported(gptq_marlin, "GPTQ Marlin");

    infinilm::quantization::GPTQ_QY gptq_qy(gptq_method->get_config());
    assert(gptq_qy.get_quant_scheme() == QuantScheme::GPTQ_W4A16_QY);
    assert(gptq_qy.get_packing_num() == 8);
    assert(gptq_qy.get_group_size() == 32);
    const auto qy_layout = gptq_qy.get_param_layout(128, 64, 0, 1, 2, -1, DataType::kFloat16, true);
    assert(qy_layout.size() == 5);
    assert(qy_layout[0].name == "qweight");
    assert(qy_layout[0].shape == std::vector<size_t>({64, 64}));
    assert(qy_layout[0].dtype == DataType::kUInt8);
    assert(qy_layout[0].split_dim == 1);
    assert(qy_layout[0].tp_rank == 1 && qy_layout[0].tp_size == 2);
    assert(qy_layout[1].shape == std::vector<size_t>({4, 64}));
    assert(qy_layout[3].dtype == DataType::kInt32);
    assert(qy_layout[4].name == "bias");
    expect_unsupported(gptq_qy, "GPTQ QY");

    infinilm::quantization::ParamsMap qy_params;
    rejected = false;
    try {
        infinilm::quantization::GPTQ_QY::convert_from_gptq(
            qy_params, infinicore::Device(infinicore::Device::Type::kCpu, 0), gptq_method->get_config());
    } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()).find("GPTQ QY conversion is unsupported") != std::string::npos;
    }
    assert(rejected);
    assert(qy_params.empty());

    rejected = false;
    try {
        infinilm::quantization::GPTQ_QY invalid_qy(nlohmann::json{{"group_size", 0}});
        invalid_qy.get_param_layout(128, 64, -1, 0, 1, -1, DataType::kFloat16, false);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

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
