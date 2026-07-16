#pragma once

#include "../context/context.hpp"
#include "../tensor.hpp"
#include "module.hpp"
#include "rope_scaling_configs.hpp"
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace infinicore::nn {

class RoPE : public Module {
public:
    /**
     * @brief RoPE algorithm type
     */
    enum class Algo {
        GPT_J = 0,    // GPT-J style RoPE algorithm (Interleave even and odd dimensions)
        GPT_NEOX = 1, // GPT-NeoX style RoPE algorithm (First half dimensions for sin, second half for cos)
    };

    /**
     * @brief Construct a RoPE layer
     *
     * @param head_dim Dimension of each attention head (must be even)
     * @param rotary_dim Number of dimensions to apply rotation to (must be even).
     *                   For full rotation models, this equals head_dim;
     *                   for partial rotation models, this equals head_dim * partial_rotary_factor.
     * @param max_seq_len Maximum sequence length for pre-computed cache
     * @param theta Base frequency for rotary embeddings (default: 10000.0)
     * @param algo RoPE algorithm type (default: Algo::GPT_J)
     * @param dtype Data type for sin/cos cache (default: DataType::kFloat32)
     * @param device Device to create the cache on
     * @param scaling RoPE scaling configuration (default: nullptr)
     * @param mrope_section Optional MRoPE section sizes [t, h, w], whose sum must equal rotary_dim / 2.
     *                       When set, pair forward overloads apply MRoPE to q/k using positions [3, num_tokens].
     * @param mrope_interleaved Whether to interleave MRoPE axes/frequency sections.
     */
    RoPE(size_t head_dim,
         size_t rotary_dim,
         size_t max_seq_len,
         double theta = 10000.0,
         Algo algo = Algo::GPT_J,
         const DataType &dtype = DataType::kFloat32,
         const Device &device = Device(),
         std::shared_ptr<RopeScalingConfig> scaling = nullptr,
         std::optional<std::vector<int>> mrope_section = std::nullopt,
         bool mrope_interleaved = false);

    /**
     * @brief Forward pass: apply standard RoPE to a tensor
     *
     * @param x Input tensor of shape (..., rotary_dim) where ... is any number of dimensions
     * @param pos Position IDs tensor of shape (*,) typically [seq_len] or [batch, seq_len]
     * @param in_place If true, modify input tensor in place (default: false)
     * @return Rotated tensor with same shape as input
     */
    Tensor forward(const Tensor &x, const Tensor &pos, bool in_place = false) const;

    /**
     * @brief Apply MRoPE to q and k.
     *
     * Requires construction with mrope_section. q/k may be either
     * [num_tokens, num_heads * head_dim] or [num_tokens, num_heads, head_dim].
     * positions is [3, num_tokens] with axes ordered as t, h, w.
     */
    std::pair<Tensor, Tensor> forward(const Tensor &q, const Tensor &k, const Tensor &positions) const;

    /**
     * @brief Apply MRoPE to q and k into caller-provided outputs.
     */
    std::pair<Tensor, Tensor> forward(const Tensor &q_out,
                                      const Tensor &k_out,
                                      const Tensor &q,
                                      const Tensor &k,
                                      const Tensor &positions) const;

    // Module information
    size_t rotary_dim() const { return rotary_dim_; }
    size_t head_dim() const { return head_dim_; }
    size_t max_seq_len() const { return max_seq_len_; }
    double theta() const { return theta_; }
    Algo algo() const { return algo_; }
    DataType dtype() const { return dtype_; }
    const std::optional<std::vector<int>> &mrope_section() const { return mrope_section_; }
    bool mrope_interleaved() const { return mrope_interleaved_; }

    // String representation
    std::string extra_repr() const;

protected:
    // Buffers (sin and cos cache tables) - not exposed in state_dict
    INFINICORE_NN_BUFFER(sin_cache);
    INFINICORE_NN_BUFFER(cos_cache);

private:
    void initialize_cache();
    size_t rotary_dim_;                          // Number of dimensions to apply rotation to (must be even).
    size_t head_dim_;                            // Dimension of each attention head
    size_t max_seq_len_;                         // Maximum sequence length
    double theta_;                               // Base frequency for rotary embeddings
    Algo algo_;                                  // RoPE algorithm type
    DataType dtype_;                             // Data type for cache tables
    std::shared_ptr<RopeScalingConfig> scaling_; // RoPE scaling configuration
    std::optional<std::vector<int>> mrope_section_;
    bool mrope_interleaved_;
};

} // namespace infinicore::nn
