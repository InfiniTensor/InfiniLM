#pragma once

/**
 * @file llama.hpp
 * @brief Main header file for Llama model architecture
 *
 * This header includes all components of the Llama model architecture
 * built using InfiniCore::nn::Module pattern.
 *
 * Components:
 * - LlamaConfig: Model configuration structure
 * - LlamaAttention: Multi-head self-attention module
 * - LlamaMLP: Feed-forward network module
 * - LlamaDecoderLayer: Single transformer decoder layer
 * - LlamaModel: Core transformer model (without LM head)
 * - LlamaForCausalLM: Complete model with language modeling head
 */

#include "llama_config.hpp"
#include "llama_attention.hpp"
#include "llama_mlp.hpp"
#include "llama_decoder_layer.hpp"
#include "llama_model.hpp"
#include "llama_for_causal_lm.hpp"
