#pragma once

#include "runtime/context.hpp"
#include "runtime/tensor.hpp"
#include "../../weights/weights.hpp"
#include "rope_scaling_configs.hpp"
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace infinilm::layers {
using infinilm::runtime::Tensor;
using infinilm::runtime::Runtime;
using infinilm::runtime::DataType;
using infinilm::runtime::Device;
using infinilm::runtime::Shape;
using infinilm::runtime::Size;


class RoPE : public infinilm::weights::ModelWeights {
public:
    /**
     * @brief RoPE algorithm type
     */
    enum class Algo { GPT_J = 0, GPT_NEOX = 1 };

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
    RoPE(Runtime &runtime, size_t head_dim,
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
    Tensor forward(Runtime &runtime, const Tensor &x, const Tensor &pos, bool in_place = false) const;

    /**
     * @brief Apply MRoPE to q and k.
     *
     * Requires construction with mrope_section. q/k may be either
     * [num_tokens, num_heads * head_dim] or [num_tokens, num_heads, head_dim].
     * positions is [3, num_tokens] with axes ordered as t, h, w.
     */
    std::pair<Tensor, Tensor> forward(Runtime &runtime, const Tensor &q, const Tensor &k, const Tensor &positions) const;
    std::pair<Tensor, Tensor> forward(Runtime &runtime, Tensor q_out, Tensor k_out,
                                      const Tensor &q, const Tensor &k, const Tensor &positions) const;


    /**
     * @brief Apply MRoPE to q and k into caller-provided outputs.
     */

    // Module information
    size_t rotary_dim() const { return rotary_dim_; }
    size_t head_dim() const { return head_dim_; }
    size_t max_seq_len() const { return max_seq_len_; }
    double theta() const { return theta_; }
    Algo algo() const { return algo_; }
    DataType dtype() const { return dtype_; }
    const std::optional<std::vector<int>> &mrope_section() const { return mrope_section_; }
    bool mrope_interleaved() const { return mrope_interleaved_; }
    // Internal caches are read-only. Packed non-MRoPE getters share the [cos|sin]
    // table: shape [M, R/2], strides [R, 1], cos offset 0, sin offset R/2 elements.
    // Nonpacked/MRoPE tables remain contiguous. Copy the returned Tensor value to
    // retain its shared storage beyond the RoPE lifetime, not this const reference.
    const Tensor &sin_cache() const { return sin_cache_; }
    const Tensor &cos_cache() const { return cos_cache_; }

    // String representation
    std::string extra_repr() const;

protected:
    // RoPE cache tables are not exposed in state_dict.
    infinilm::runtime::Tensor sin_cache_;
    infinilm::runtime::Tensor cos_cache_;
    infinilm::runtime::Tensor cos_sin_cache_;

private:
    infinilm::runtime::Device device_;
    void initialize_cache(Runtime &runtime);
    std::shared_ptr<Runtime> cache_runtime_owner_;
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

} // namespace infinilm::layers
