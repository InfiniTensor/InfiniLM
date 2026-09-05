#include "csrc/config/quant_config.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using infinilm::config::QuantConfig;
using infinilm::quantization::QuantScheme;
using nlohmann::json;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_invalid_method(
    const json &config,
    const std::string &expected_message) {
    try {
        QuantConfig(config).get_quantization_method(
            "model.layers.0.mlp.up_proj", "Linear");
    } catch (const std::invalid_argument &error) {
        expect(
            std::string(error.what()).find(expected_message) != std::string::npos,
            "unexpected quantization-method error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected unsupported quantization method");
}

void test_module_aware_selection() {
    const auto config = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "ignore": ["model.layers.0.self_attn.o_proj"],
        "config_groups": {
            "W8A8": ["Linear"],
            "UNQUANTIZED": ["lm_head"]
        }
    })json");
    const QuantConfig quant_config(config);

    expect(
        quant_config
                .get_quantization_method(
                    "model.layers.0.self_attn.q_proj", "Linear")
                ->get_quant_scheme()
            == QuantScheme::COMPRESSED_TENSOR_W8A8I8,
        "matching W8A8 module did not select compressed-tensors");
    expect(
        quant_config
                .get_quantization_method(
                    "model.layers.0.self_attn.o_proj", "Linear")
                ->get_quant_scheme()
            == QuantScheme::NONE,
        "ignored module was quantized");
    expect(
        quant_config.get_quantization_method("lm_head", "Linear")
                ->get_quant_scheme()
            == QuantScheme::NONE,
        "UNQUANTIZED group selected a quantized scheme");
    expect(
        quant_config.get_quantization_method("model.embed_tokens", "Embedding")
                ->get_quant_scheme()
            == QuantScheme::NONE,
        "unmatched module was quantized");
}

void test_unsupported_schemes_are_rejected() {
    const json fakequant_config = {
        {"quant_method", "compressed-tensors"},
        {"format", "fakequant"},
        {"config_groups", {{"W8A8", {"Linear"}}}},
    };
    const auto unsupported_scheme = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "W4A8": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": true,
                    "strategy": "channel",
                    "dynamic": false
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": true,
                    "strategy": "token",
                    "dynamic": true
                }
            }
        }
    })json");

    expect_invalid_method(fakequant_config, "format `fakequant`");
    expect_invalid_method(unsupported_scheme, "group `W4A8`");
}

void test_existing_methods_are_preserved() {
    const QuantConfig quant_config(json{{"quant_method", "awq"}});
    expect(
        quant_config.get_quantization_method("model.layers.0.mlp.up_proj", "Linear")
                ->get_quant_scheme()
            == QuantScheme::AWQ_W4A16,
        "module-aware lookup changed the existing AWQ selection");
}

} // namespace

int main() {
    try {
        test_module_aware_selection();
        test_unsupported_schemes_are_rejected();
        test_existing_methods_are_preserved();
    } catch (const std::exception &error) {
        std::cerr << "quant_config_test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "quant_config_test passed\n";
    return 0;
}
