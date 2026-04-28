#pragma once

#include "../ops.hpp"
#include "module.hpp"

namespace infinicore::nn {

/**
 * @brief Root Mean Square Layer Normalization (RMSNorm)
 *
 * Applies Root Mean Square Layer Normalization over the last dimension.
 * Unlike LayerNorm, RMSNorm doesn't subtract mean and doesn't use bias.
 *
 * Formula: y = (x / RMS(x)) * weight
 * where RMS(x) = sqrt(mean(x^2) + eps)
 *
 * Used in LLaMA, Galactica, and other modern language models as a
 * simpler and faster alternative to LayerNorm.
 *
 * Example:
 * @code
 *   // Create RMSNorm for hidden size 4096
 *   auto norm = RMSNorm(4096);
 *
 *   // Input: [batch, seq_len, hidden_size]
 *   auto input = Tensor::randn({2, 10, 4096});
 *
 *   // Output: [batch, seq_len, hidden_size]
 *   auto output = norm.forward(input);
 * @endcode
 */
class RMSNorm : public Module {
public:
    /**
     * @brief Construct a RMSNorm layer
     *
     * @param normalized_shape Size of the feature dimension to normalize (typically hidden_size)
     * @param eps Small constant for numerical stability (default: 1e-6)
     * @param dtype Data type for the weight (default: DataType::F32)
     * @param device Device to create the weight on
     */
    RMSNorm(size_t normalized_shape,
            double eps = 1e-6,
            const DataType &dtype = DataType::F32,
            const Device &device = Device());

    /**
     * @brief Forward pass: apply RMSNorm
     *
     * @param x Input tensor of shape (*, normalized_shape) where * is any number of dimensions
     * @return Normalized tensor with same shape as input
     *
     * The normalization is applied over the last dimension.
     * For example:
     *   Input: [batch, seq_len, hidden_size] -> normalize over hidden_size
     *   Input: [batch, hidden_size] -> normalize over hidden_size
     */
    Tensor forward(const Tensor &x) const;

    /**
     * @brief Forward pass: apply RMSNorm in-place with residual
     *
     * @param x Input tensor of shape (*, normalized_shape) where * is any number of dimensions.
     *       Will be modified in-place to the normalized output.
     * @param residual Residual tensor to add to input before normalization.
     *       Will be modified in-place to the sum of input and residual.
     *
     * The normalization is applied over the last dimension.
     * For example:
     *   Input: [batch, seq_len, hidden_size] -> normalize over hidden_size
     *   Input: [batch, hidden_size] -> normalize over hidden_size
     */
    void forward_inplace(Tensor &x, Tensor &residual) const;

    // Module information
    size_t normalized_shape() const { return normalized_shape_; }
    double eps() const { return eps_; }
    DataType dtype() const { return dtype_; }

    // String representation
    std::string extra_repr() const;

    // Accessors for parameters
    Tensor weight() const { return weight_; }

protected:
    // Parameters
    INFINICORE_NN_PARAMETER(weight);

private:
    size_t normalized_shape_; // Size of the feature dimension
    double eps_;              // Epsilon for numerical stability
    DataType dtype_;          // Data type for weight
};

} // namespace infinicore::nn
