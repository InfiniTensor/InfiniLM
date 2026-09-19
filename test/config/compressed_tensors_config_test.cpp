#include "csrc/config/compressed_tensors_config.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using infinilm::config::CompressedTensorsConfig;
using infinilm::config::QuantizationDynamicMode;
using infinilm::config::QuantizationStatus;
using infinilm::config::QuantizationStrategy;
using infinilm::config::QuantizationValueType;
using nlohmann::json;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_invalid_config(
    const json &config,
    const std::string &expected_message) {
    try {
        CompressedTensorsConfig::from_json(config);
    } catch (const std::invalid_argument &error) {
        expect(
            std::string(error.what()).find(expected_message) != std::string::npos,
            "unexpected validation error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected invalid compressed-tensors config");
}

void test_explicit_w8a8_group() {
    const auto config = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head", "re:model\\.layers\\.0\\..*"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
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

    const auto parsed = CompressedTensorsConfig::from_json(config);
    expect(parsed.format == "int-quantized", "format was not parsed");
    expect(
        parsed.quantization_status == QuantizationStatus::COMPRESSED,
        "quantization_status was not parsed");
    expect(parsed.ignore.size() == 2, "ignore list was not parsed");
    expect(parsed.config_groups.size() == 1, "config group was not parsed");

    const auto &group = parsed.config_groups.front();
    expect(group.name == "group_0", "group name was not preserved");
    expect(group.targets == std::vector<std::string>{"Linear"}, "targets were not parsed");
    expect(group.weights.has_value(), "weights were not parsed");
    expect(group.input_activations.has_value(), "input activations were not parsed");
    expect(group.weights->num_bits == 8, "weight bit width was not parsed");
    expect(group.weights->type == QuantizationValueType::INT, "weight type was not parsed");
    expect(group.weights->strategy == QuantizationStrategy::CHANNEL, "weight strategy was not parsed");
    expect(
        group.weights->dynamic == QuantizationDynamicMode::STATIC,
        "weight dynamic mode was not parsed");
    expect(
        group.input_activations->strategy == QuantizationStrategy::TOKEN,
        "activation strategy was not parsed");
    expect(
        group.input_activations->dynamic == QuantizationDynamicMode::DYNAMIC,
        "activation dynamic mode was not parsed");
    expect(parsed.raw_config == config, "raw config was not preserved");
}

void test_w8a8_preset() {
    const json config = {
        {"quant_method", "compressed-tensors"},
        {"config_groups", {{"W8A8", {"Linear"}}}},
    };

    const auto group = CompressedTensorsConfig::from_json(config).config_groups.front();
    expect(group.weights.has_value(), "W8A8 preset did not create weights");
    expect(group.input_activations.has_value(), "W8A8 preset did not create activations");
    expect(group.weights->num_bits == 8, "W8A8 preset did not use 8-bit weights");
    expect(group.weights->strategy == QuantizationStrategy::CHANNEL, "W8A8 weight strategy is incorrect");
    expect(group.input_activations->strategy == QuantizationStrategy::TOKEN, "W8A8 activation strategy is incorrect");
    expect(group.input_activations->dynamic == QuantizationDynamicMode::DYNAMIC, "W8A8 activations are not dynamic");
}

void test_defaults_and_optional_fields() {
    const auto config = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "global_compression_ratio": 0.5,
        "kv_cache_scheme": {
            "num_bits": 8,
            "type": "int",
            "strategy": "channel",
            "dynamic": "local"
        },
        "config_groups": {
            "INT8": ["Linear"],
            "UNQUANTIZED": ["lm_head"],
            "optional": {
                "targets": ["re:model\\.layers\\..*"],
                "format": "fakequant",
                "output_activations": {
                    "type": "float",
                    "strategy": "tensor_group",
                    "dynamic": "local"
                }
            }
        }
    })json");

    const auto parsed = CompressedTensorsConfig::from_json(config);
    expect(parsed.format == "fakequant", "default format is incorrect");
    expect(
        parsed.quantization_status == QuantizationStatus::INITIALIZED,
        "default quantization status is incorrect");
    expect(parsed.ignore.empty(), "default ignore list is not empty");
    expect(parsed.global_compression_ratio == 0.5, "compression ratio was not parsed");
    expect(parsed.kv_cache_scheme.has_value(), "KV-cache scheme was not parsed");
    expect(
        parsed.kv_cache_scheme->dynamic == QuantizationDynamicMode::LOCAL,
        "local dynamic mode was not parsed");

    const auto find_group = [&parsed](const std::string &name) -> const auto & {
        for (const auto &group : parsed.config_groups) {
            if (group.name == name) {
                return group;
            }
        }
        throw std::runtime_error("missing config group: " + name);
    };

    const auto &int8 = find_group("INT8");
    expect(int8.weights.has_value(), "INT8 preset did not create weights");
    expect(int8.input_activations.has_value(), "INT8 preset did not create activations");

    const auto &unquantized = find_group("UNQUANTIZED");
    expect(!unquantized.weights.has_value(), "UNQUANTIZED preset created weights");
    expect(!unquantized.input_activations.has_value(), "UNQUANTIZED preset created activations");

    const auto &optional = find_group("optional");
    expect(optional.format == "fakequant", "group format was not parsed");
    expect(optional.output_activations.has_value(), "output activations were not parsed");
    expect(
        optional.output_activations->type == QuantizationValueType::FLOAT,
        "floating-point value type was not parsed");
    expect(
        optional.output_activations->strategy == QuantizationStrategy::TENSOR_GROUP,
        "tensor-group strategy was not parsed");
}

void test_group_resolution() {
    const auto config = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "ignore": [
            "model.layers.3.self_attn.o_proj",
            "re:model\\.layers\\.4\\."
        ],
        "config_groups": {
            "type_group": {
                "targets": ["Linear"]
            },
            "regex_group": {
                "targets": ["re:model\\.layers\\.3\\."]
            },
            "exact_group": {
                "targets": ["model.layers.3.self_attn.q_proj"]
            }
        }
    })json");
    const auto parsed = CompressedTensorsConfig::from_json(config);

    const auto *exact = parsed.resolve_group(
        "model.layers.3.self_attn.q_proj", "Linear");
    expect(exact != nullptr, "exact group was not resolved");
    expect(exact->name == "exact_group", "exact group did not take priority");

    const auto *regex = parsed.resolve_group(
        "model.layers.3.self_attn.v_proj", "Linear");
    expect(regex != nullptr, "regex group was not resolved");
    expect(regex->name == "regex_group", "regex group did not take priority over type");

    const auto *type = parsed.resolve_group(
        "model.layers.2.self_attn.q_proj", "Linear");
    expect(type != nullptr, "module-type group was not resolved");
    expect(type->name == "type_group", "incorrect module-type group was resolved");

    expect(
        parsed.resolve_group("model.embed_tokens", "Embedding") == nullptr,
        "unmatched module resolved to a group");
    expect(
        parsed.is_ignored("model.layers.3.self_attn.o_proj", "Linear"),
        "exact ignore rule did not match");
    expect(
        parsed.resolve_group("model.layers.3.self_attn.o_proj", "Linear")
            == nullptr,
        "ignored module resolved to a group");
    expect(
        parsed.resolve_group("model.layers.4.self_attn.q_proj", "Linear")
            == nullptr,
        "regex-ignored module resolved to a group");
}

void test_group_resolution_checks_ambiguity_at_highest_specificity() {
    const auto lower_priority_tie = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "config_groups": {
            "type_a": {"targets": ["Linear"]},
            "type_b": {"targets": ["Linear"]},
            "z_exact": {"targets": ["model.layers.0.self_attn.q_proj"]}
        }
    })json");

    const auto parsed = CompressedTensorsConfig::from_json(lower_priority_tie);
    const auto *resolved = parsed.resolve_group(
        "model.layers.0.self_attn.q_proj", "Linear");
    expect(resolved != nullptr, "exact group was not resolved");
    expect(
        resolved->name == "z_exact",
        "a lower-specificity tie incorrectly overrode the exact match");

    const auto highest_priority_tie = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "config_groups": {
            "exact_a": {"targets": ["model.layers.0.self_attn.q_proj"]},
            "exact_b": {"targets": ["model.layers.0.self_attn.q_proj"]}
        }
    })json");

    try {
        CompressedTensorsConfig::from_json(highest_priority_tie).resolve_group("model.layers.0.self_attn.q_proj", "Linear");
    } catch (const std::invalid_argument &error) {
        expect(
            std::string(error.what()).find("equal specificity") != std::string::npos,
            "unexpected group-resolution error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected equally specific groups to be ambiguous");
}

void test_validation_errors_include_paths() {
    const auto invalid_dynamic = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {"dynamic": "global"}
            }
        }
    })json");
    const auto invalid_num_bits = json::parse(R"json({
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {"num_bits": 0}
            }
        }
    })json");

    expect_invalid_config(json::array(), "quantization_config: expected an object");
    expect_invalid_config(
        {{"quant_method", "compressed-tensors"}},
        "quantization_config.config_groups");
    expect_invalid_config(
        {{"quant_method", "gptq"}, {"config_groups", json::object()}},
        "quantization_config.quant_method");
    expect_invalid_config(
        {{"quant_method", "compressed-tensors"}, {"config_groups", {{"group_0", json::object()}}}},
        "quantization_config.config_groups.group_0.targets");
    expect_invalid_config(
        invalid_dynamic,
        "quantization_config.config_groups.group_0.input_activations.dynamic");
    expect_invalid_config(
        invalid_num_bits,
        "quantization_config.config_groups.group_0.weights.num_bits");
    expect_invalid_config(
        {{"quant_method", "compressed-tensors"}, {"config_groups", {{"FP8", {"Linear"}}}}},
        "unsupported preset \"FP8\"");
}

} // namespace

int main() {
    try {
        test_explicit_w8a8_group();
        test_w8a8_preset();
        test_defaults_and_optional_fields();
        test_group_resolution();
        test_group_resolution_checks_ambiguity_at_highest_specificity();
        test_validation_errors_include_paths();
    } catch (const std::exception &error) {
        std::cerr << "compressed_tensors_config_test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "compressed_tensors_config_test passed\n";
    return 0;
}
