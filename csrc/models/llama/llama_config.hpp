#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../infinilm_model.hpp"

#include <infinicore/nn/rope.hpp>

namespace infinilm::models::llama {

/**
 * @brief Configuration structure for Llama model architecture
 *
 * This struct holds all hyperparameters needed to construct a Llama model.
 * It follows the same structure as HuggingFace's LlamaConfig.
 */
struct LlamaConfig : public InfinilmModel::Config {
    // Data type
    infinicore::DataType dtype = infinicore::DataType::F32;

    // Vocabulary and embedding
    size_t vocab_size = 32000;        // Vocabulary size
    size_t hidden_size = 4096;        // Hidden dimension size
    size_t intermediate_size = 11008; // MLP intermediate dimension

    // Architecture
    size_t num_hidden_layers = 32;   // Number of decoder layers
    size_t num_attention_heads = 32; // Number of attention heads
    size_t num_key_value_heads = 32; // Number of key-value heads (for GQA)
    size_t head_dim = 128;           // Attention head dimension (hidden_size / num_attention_heads)

    // Position embeddings
    size_t max_position_embeddings = 2048; // Maximum sequence length
    double rope_theta = 10000.0;           // RoPE base frequency

    std::shared_ptr<infinicore::nn::RoPE::ScalingConfig> rope_scaling = nullptr; // RoPE scaling type

    // Normalization
    double rms_norm_eps = 1e-6; // RMSNorm epsilon

    // Activation
    std::string hidden_act = "silu";  // Activation function (typically "silu")
    std::string model_type = "llama"; // Model type identifier (matches HF configs)

    // Optional features
    bool use_cache = true;              // Whether to use KV cache
    bool attention_bias = true;         // Whether to use bias in Q/K/V projections (default true for 9G7B compatibility)
    bool attention_output_bias = false; // Whether to use bias in output projection (o_proj)
    bool mlp_bias = false;              // Whether to use bias in MLP projections
    bool tie_word_embeddings = false;   // Whether to tie input/output embeddings
    bool qk_norm = false;               // Whether to use QK RMSNorm

    // Training/initialization parameters
    double attention_dropout = 0.0;  // Dropout ratio for attention probabilities
    double initializer_range = 0.02; // Standard deviation for weight initialization
    size_t pretraining_tp = 1;       // Tensor parallelism rank used during pretraining

    // Model metadata
    std::string name_or_path = ""; // Model name or path identifier

    // Token IDs
    int64_t pad_token_id = -1;               // Padding token ID (optional)
    std::vector<int64_t> bos_token_id = {1}; // Beginning of sequence token ID(s)
    std::vector<int64_t> eos_token_id = {2}; // End of sequence token ID(s)

    /**
     * @brief Compute key-value dimension for Grouped Query Attention (GQA)
     * @return The dimension for key/value projections
     */
    size_t kv_dim() const {
        return hidden_size * num_key_value_heads / num_attention_heads;
    }

    /**
     * @brief Validate configuration parameters
     * @return true if configuration is valid
     */
    bool validate() const {
        if (hidden_size % num_attention_heads != 0) {
            return false;
        }
        if (num_attention_heads % num_key_value_heads != 0) {
            return false;
        }
        if (head_dim != hidden_size / num_attention_heads) {
            return false;
        }
        return true;
    }
};

} // namespace infinilm::models::llama
