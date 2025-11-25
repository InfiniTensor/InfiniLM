#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace infinilm::models::llama {

/**
 * @brief Configuration structure for Llama model architecture
 *
 * This struct holds all hyperparameters needed to construct a Llama model.
 * It follows the same structure as HuggingFace's LlamaConfig.
 */
struct LlamaConfig {
    // Vocabulary and embedding
    size_t vocab_size = 32000;              // Vocabulary size
    size_t hidden_size = 4096;               // Hidden dimension size
    size_t intermediate_size = 11008;        // MLP intermediate dimension

    // Architecture
    size_t num_hidden_layers = 32;           // Number of decoder layers
    size_t num_attention_heads = 32;         // Number of attention heads
    size_t num_key_value_heads = 32;         // Number of key-value heads (for GQA)
    size_t head_dim = 128;                   // Attention head dimension (hidden_size / num_attention_heads)

    // Position embeddings
    size_t max_position_embeddings = 2048;   // Maximum sequence length
    double rope_theta = 10000.0;             // RoPE base frequency

    // Normalization
    double rms_norm_eps = 1e-6;              // RMSNorm epsilon

    // Activation
    std::string hidden_act = "silu";         // Activation function (typically "silu")

    // Optional features
    bool use_cache = true;                   // Whether to use KV cache
    bool attention_bias = false;             // Whether to use bias in attention projections
    bool mlp_bias = false;                   // Whether to use bias in MLP projections
    bool tie_word_embeddings = false;        // Whether to tie input/output embeddings

    // Token IDs
    int64_t pad_token_id = -1;               // Padding token ID (optional)
    int64_t bos_token_id = 1;                // Beginning of sequence token ID
    int64_t eos_token_id = 2;                // End of sequence token ID

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
