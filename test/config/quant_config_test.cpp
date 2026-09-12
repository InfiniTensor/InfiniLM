#include "csrc/config/quant_config.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using infinilm::config::QuantConfig;
using infinilm::config::QuantizationStatus;
using infinilm::quantization::QuantScheme;
using nlohmann::json;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_scheme(
    const QuantConfig &config,
    std::string_view module_name,
    std::string_view module_type,
    QuantScheme expected,
    const std::string &message) {
    expect(
        config.get_quantization_method(module_name, module_type)
                ->get_quant_scheme()
            == expected,
        message);
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

json real_qwen3_w8a8_config() {
    return json::parse(R"json({
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "input_activations": {
                    "actorder": null,
                    "block_structure": null,
                    "dynamic": true,
                    "group_size": null,
                    "num_bits": 8,
                    "observer": null,
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": true,
                    "type": "int"
                },
                "output_activations": null,
                "targets": ["Linear"],
                "weights": {
                    "actorder": null,
                    "block_structure": null,
                    "dynamic": false,
                    "group_size": null,
                    "num_bits": 8,
                    "observer": "mse",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": true,
                    "type": "int"
                }
            }
        },
        "global_compression_ratio": null,
        "kv_cache_scheme": null,
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.13.0"
    })json");
}

void test_real_qwen3_w8a8_config() {
    const auto config = real_qwen3_w8a8_config();
    const QuantConfig quant_config(config);

    const auto &parsed = quant_config.get_compressed_tensors_config();
    expect(parsed.has_value(), "real Qwen3 config was not parsed");
    expect(
        parsed->quantization_status == QuantizationStatus::COMPRESSED,
        "real Qwen3 quantization status was not parsed");
    expect(
        parsed->raw_config.at("version") == "0.13.0",
        "real Qwen3 compressed-tensors version was not preserved");

    for (const auto *module_name : {
             "model.layers.0.self_attn.q_proj",
             "model.layers.0.self_attn.k_proj",
             "model.layers.0.self_attn.v_proj",
             "model.layers.0.self_attn.o_proj",
             "model.layers.0.mlp.gate_proj",
             "model.layers.0.mlp.up_proj",
             "model.layers.0.mlp.down_proj",
         }) {
        expect_scheme(
            quant_config,
            module_name,
            "Linear",
            QuantScheme::COMPRESSED_TENSOR_W8A8I8,
            std::string("real Qwen3 config did not quantize ") + module_name);
    }
    expect_scheme(
        quant_config,
        "lm_head",
        "Linear",
        QuantScheme::NONE,
        "real Qwen3 config did not preserve its lm_head ignore rule");
}

void test_unquantized_and_unmatched_modules() {
    const QuantConfig quant_config(json{
        {"quant_method", "compressed-tensors"},
        {"config_groups", {{"UNQUANTIZED", {"lm_head"}}}},
    });

    expect_scheme(
        quant_config,
        "lm_head",
        "Linear",
        QuantScheme::NONE,
        "UNQUANTIZED group selected a quantized scheme");
    expect_scheme(
        quant_config,
        "model.embed_tokens",
        "Embedding",
        QuantScheme::NONE,
        "unmatched module was quantized");
}

void test_unsupported_schemes_are_rejected() {
    const json fakequant_config = {
        {"quant_method", "compressed-tensors"},
        {"format", "fakequant"},
        {"config_groups", {{"W8A8", {"Linear"}}}},
    };
    auto w4a8_config = real_qwen3_w8a8_config();
    w4a8_config["config_groups"]["group_0"]["weights"]["num_bits"] = 4;

    expect_invalid_method(fakequant_config, "format `fakequant`");
    expect_invalid_method(w4a8_config, "group `group_0`");
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
        test_real_qwen3_w8a8_config();
        test_unquantized_and_unmatched_modules();
        test_unsupported_schemes_are_rejected();
        test_existing_methods_are_preserved();
    } catch (const std::exception &error) {
        std::cerr << "quant_config_test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "quant_config_test passed\n";
    return 0;
}
