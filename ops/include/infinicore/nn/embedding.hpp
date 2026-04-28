#pragma once

#include "../ops.hpp"
#include "module.hpp"
#include <optional>

namespace infinicore::nn {

/**
 * @brief Embedding layer that maps indices to dense vectors
 *
 * A simple lookup table that stores embeddings of a fixed dictionary and size.
 * This module is often used to store word embeddings and retrieve them using indices.
 * The input to the module is a tensor of indices, and the output is the corresponding
 * embedding vectors.
 *
 * Similar to PyTorch's nn.Embedding:
 * https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
 *
 * Example:
 * @code
 *   // Create embedding: 10000 words, 300-dimensional embeddings
 *   auto embedding = Embedding(10000, 300);
 *
 *   // Input: tensor of indices [batch_size, seq_len]
 *   auto indices = Tensor::from_data({2, 5}, {3, 5, 12, 8, 99, 0, 1, 45, 67, 23});
 *
 *   // Output: [batch_size, seq_len, embedding_dim] = [2, 5, 300]
 *   auto embeddings = embedding.forward(indices);
 * @endcode
 */
class Embedding : public Module {
public:
    /**
     * @brief Construct an Embedding layer
     *
     * @param num_embeddings Size of the dictionary of embeddings (vocabulary size)
     * @param embedding_dim The size of each embedding vector
     * @param padding_idx If specified, the entries at padding_idx do not contribute to gradient
     *                    and the embedding vector at padding_idx is not updated during training
     * @param dtype Data type for the embedding weights (default: DataType::F32)
     * @param device Device to create the embedding weight on
     */
    Embedding(size_t num_embeddings,
              size_t embedding_dim,
              std::optional<int64_t> padding_idx = std::nullopt,
              const DataType &dtype = DataType::F32,
              const Device &device = Device());

    /**
     * @brief Forward pass: lookup embeddings for given indices
     *
     * @param indices Tensor containing indices into the embedding matrix.
     *                Can be any shape (*), typically [batch_size] or [batch_size, seq_len]
     * @return Tensor containing the embedding vectors.
     *         Shape: (*, embedding_dim) where * matches the input shape
     *
     * Example:
     *   Input shape: [2, 3] -> Output shape: [2, 3, embedding_dim]
     *   Input shape: [10] -> Output shape: [10, embedding_dim]
     */
    Tensor forward(const Tensor &indices) const;

    // Module information
    size_t num_embeddings() const { return num_embeddings_; }
    size_t embedding_dim() const { return embedding_dim_; }
    std::optional<int64_t> padding_idx() const { return padding_idx_; }
    DataType dtype() const { return dtype_; }

    // String representation
    std::string extra_repr() const;

    // Accessors for parameters
    Tensor weight() const { return weight_; }

protected:
    // Parameters
    INFINICORE_NN_PARAMETER(weight);

private:
    size_t num_embeddings_;              // Vocabulary size
    size_t embedding_dim_;               // Embedding dimension
    std::optional<int64_t> padding_idx_; // Optional padding index
    DataType dtype_;                     // Data type for embedding weights
};

} // namespace infinicore::nn
