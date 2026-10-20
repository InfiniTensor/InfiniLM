#pragma once

#include "../../weights/weights.hpp"

namespace infinilm::layers {
using infinilm::runtime::Tensor;
using infinilm::runtime::Runtime;
using infinilm::runtime::DataType;
using infinilm::runtime::Device;
using infinilm::runtime::Shape;
using infinilm::runtime::Size;


class LayerNorm : public infinilm::weights::ModelWeights {
public:
    LayerNorm(Runtime &runtime,
              size_t normalized_shape,
              double eps = 1e-6,
              const DataType &dtype = DataType::kFloat32);

    /**
     * @brief Forward pass: apply LayerNorm
     *
     * @param x Input tensor of shape (*, normalized_shape) where * is any number of dimensions
     * @return Normalized tensor with same shape as input
     *
     * The normalization is applied over the last dimension.
     * For example:
     *   Input: [batch, seq_len, hidden_size] -> normalize over hidden_size
     *   Input: [batch, hidden_size] -> normalize over hidden_size
     */
    Tensor forward(Runtime &runtime, const Tensor &x) const;

    // Module information
    size_t normalized_shape() const { return normalized_shape_; }
    double eps() const { return eps_; }
    DataType dtype() const { return dtype_; }

    // String representation
    std::string extra_repr() const;

    // Accessors for parameters
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }

protected:
    // Parameters
    infinilm::weights::Weight weight_;
    infinilm::weights::Weight bias_;

private:
    infinilm::runtime::Device device_;
    size_t normalized_shape_; // Size of the feature dimension
    double eps_;              // Epsilon for numerical stability
    DataType dtype_;          // Data type for weight
};

} // namespace infinilm::layers
