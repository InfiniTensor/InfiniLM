#include <iostream>
#include <cassert>
#include "models/llama/llama.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/device.hpp"

using namespace infinilm::models::llama;
using namespace infinicore;

int main() {
    std::cout << "==============================================\n"
              << "Llama Model Skeleton Validation Test\n"
              << "==============================================\n" << std::endl;

    // Create test configuration (TinyLlama-like)
    LlamaConfig config;
    config.vocab_size = 32000;
    config.hidden_size = 2048;
    config.intermediate_size = 5632;
    config.num_hidden_layers = 4;  // Small for testing
    config.num_attention_heads = 32;
    config.num_key_value_heads = 4;  // GQA
    config.head_dim = 64;  // hidden_size // num_attention_heads
    config.max_position_embeddings = 2048;
    config.rms_norm_eps = 1e-6;
    config.rope_theta = 10000.0;
    config.attention_bias = false;
    config.mlp_bias = false;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;
    std::cout << "  hidden_size: " << config.hidden_size << std::endl;
    std::cout << "  intermediate_size: " << config.intermediate_size << std::endl;
    std::cout << "  num_hidden_layers: " << config.num_hidden_layers << std::endl;
    std::cout << "  num_attention_heads: " << config.num_attention_heads << std::endl;
    std::cout << "  num_key_value_heads: " << config.num_key_value_heads << " (GQA)" << std::endl;
    std::cout << "  head_dim: " << config.head_dim << std::endl;
    std::cout << "  max_position_embeddings: " << config.max_position_embeddings << std::endl;
    std::cout << "  rms_norm_eps: " << config.rms_norm_eps << std::endl;
    std::cout << std::endl;

    // Validate configuration
    if (!config.validate()) {
        std::cerr << "ERROR: Configuration validation failed!" << std::endl;
        return 1;
    }
    std::cout << "✓ Configuration validated" << std::endl;

    // Create device
    Device device;

    // Test 1: Create LlamaModel
    std::cout << "\n1. Creating LlamaModel..." << std::endl;
    LlamaModel model(config, device);
    std::cout << "   ✓ Model created successfully" << std::endl;

    // Test 2: Get state_dict
    std::cout << "\n2. Retrieving state_dict()..." << std::endl;
    auto state_dict = model.state_dict();
    std::cout << "   ✓ Found " << state_dict.size() << " parameters" << std::endl;

    // Test 3: Validate parameter structure
    std::cout << "\n3. Validating parameter structure..." << std::endl;

    size_t kv_dim = config.kv_dim();
    size_t expected_param_count = 0;

    // Count expected parameters
    expected_param_count += 1;  // embed_tokens.weight
    expected_param_count += config.num_hidden_layers * 9;  // 9 params per layer
    expected_param_count += 1;  // norm.weight

    if (state_dict.size() != expected_param_count) {
        std::cerr << "   ✗ Parameter count mismatch: got " << state_dict.size()
                  << ", expected " << expected_param_count << std::endl;
        return 1;
    }
    std::cout << "   ✓ Parameter count matches: " << state_dict.size() << std::endl;

    // Validate specific parameter shapes
    bool all_shapes_ok = true;

    // Check embed_tokens.weight
    auto embed_it = state_dict.find("embed_tokens.weight");
    if (embed_it == state_dict.end()) {
        std::cerr << "   ✗ Missing embed_tokens.weight" << std::endl;
        all_shapes_ok = false;
    } else {
        auto shape = embed_it->second->shape();
        if (shape.size() != 2 || shape[0] != config.vocab_size || shape[1] != config.hidden_size) {
            std::cerr << "   ✗ embed_tokens.weight shape mismatch" << std::endl;
            all_shapes_ok = false;
        }
    }

    // Check layer parameters
    for (size_t i = 0; i < config.num_hidden_layers; ++i) {
        std::string prefix = "layers." + std::to_string(i) + ".";

        // Check input_layernorm.weight
        auto it = state_dict.find(prefix + "input_layernorm.weight");
        if (it == state_dict.end() || it->second->shape()[0] != config.hidden_size) {
            std::cerr << "   ✗ " << prefix << "input_layernorm.weight missing or wrong shape" << std::endl;
            all_shapes_ok = false;
        }

        // Check attention projections
        std::vector<std::pair<std::string, std::vector<size_t>>> attn_params = {
            {"self_attn.q_proj.weight", {config.hidden_size, config.hidden_size}},
            {"self_attn.k_proj.weight", {kv_dim, config.hidden_size}},
            {"self_attn.v_proj.weight", {kv_dim, config.hidden_size}},
            {"self_attn.o_proj.weight", {config.hidden_size, config.hidden_size}},
        };

        for (const auto& [name, expected_shape] : attn_params) {
            auto it = state_dict.find(prefix + name);
            if (it == state_dict.end()) {
                std::cerr << "   ✗ Missing " << prefix << name << std::endl;
                all_shapes_ok = false;
            } else {
                auto actual_shape = it->second->shape();
                if (actual_shape.size() != expected_shape.size() ||
                    (actual_shape[0] != expected_shape[0] || actual_shape[1] != expected_shape[1])) {
                    std::cerr << "   ✗ " << prefix << name << " shape mismatch" << std::endl;
                    all_shapes_ok = false;
                }
            }
        }

        // Check MLP projections
        std::vector<std::pair<std::string, std::vector<size_t>>> mlp_params = {
            {"mlp.gate_proj.weight", {config.intermediate_size, config.hidden_size}},
            {"mlp.up_proj.weight", {config.intermediate_size, config.hidden_size}},
            {"mlp.down_proj.weight", {config.hidden_size, config.intermediate_size}},
        };

        for (const auto& [name, expected_shape] : mlp_params) {
            auto it = state_dict.find(prefix + name);
            if (it == state_dict.end()) {
                std::cerr << "   ✗ Missing " << prefix << name << std::endl;
                all_shapes_ok = false;
            } else {
                auto actual_shape = it->second->shape();
                if (actual_shape.size() != expected_shape.size() ||
                    (actual_shape[0] != expected_shape[0] || actual_shape[1] != expected_shape[1])) {
                    std::cerr << "   ✗ " << prefix << name << " shape mismatch" << std::endl;
                    all_shapes_ok = false;
                }
            }
        }

        // Check post_attention_layernorm.weight
        it = state_dict.find(prefix + "post_attention_layernorm.weight");
        if (it == state_dict.end() || it->second->shape()[0] != config.hidden_size) {
            std::cerr << "   ✗ " << prefix << "post_attention_layernorm.weight missing or wrong shape" << std::endl;
            all_shapes_ok = false;
        }
    }

    // Check final norm.weight
    auto norm_it = state_dict.find("norm.weight");
    if (norm_it == state_dict.end() || norm_it->second->shape()[0] != config.hidden_size) {
        std::cerr << "   ✗ norm.weight missing or wrong shape" << std::endl;
        all_shapes_ok = false;
    }

    if (all_shapes_ok) {
        std::cout << "   ✓ All parameter shapes validated" << std::endl;
    } else {
        std::cerr << "   ✗ Some parameter shapes failed validation" << std::endl;
        return 1;
    }

    // Test 4: Create LlamaForCausalLM
    std::cout << "\n4. Creating LlamaForCausalLM..." << std::endl;
    LlamaForCausalLM causal_model(config, device);
    auto causal_state_dict = causal_model.state_dict();

    // Should have model params + lm_head.weight
    size_t expected_causal_params = expected_param_count + 1;
    if (causal_state_dict.size() >= expected_causal_params) {
        std::cout << "   ✓ LlamaForCausalLM created with " << causal_state_dict.size() << " parameters" << std::endl;

        // Check lm_head.weight
        auto lm_it = causal_state_dict.find("lm_head.weight");
        if (lm_it != causal_state_dict.end()) {
            auto lm_shape = lm_it->second->shape();
            if (lm_shape.size() == 2 && lm_shape[0] == config.vocab_size && lm_shape[1] == config.hidden_size) {
                std::cout << "   ✓ lm_head.weight shape correct: [" << lm_shape[0] << ", " << lm_shape[1] << "]" << std::endl;
            } else {
                std::cerr << "   ✗ lm_head.weight shape mismatch" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "   ✗ Missing lm_head.weight" << std::endl;
            return 1;
        }
    } else {
        std::cerr << "   ✗ Parameter count mismatch: got " << causal_state_dict.size()
                  << ", expected at least " << expected_causal_params << std::endl;
        return 1;
    }

    std::cout << "\n==============================================" << std::endl;
    std::cout << "✓ All tests PASSED!" << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
