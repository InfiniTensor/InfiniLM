#ifndef __GQA_INFO_H__
#define __GQA_INFO_H__

#include "../../../utils.h" // Mocked: Assumes a utility header for Result<>
#include "../../tensor.h"    // Mocked: Assumes a header for infiniopTensorDescriptor_t

namespace op::gqa {

/**
 * @brief Holds validated tensor dimensions and metadata for the GQA operation.
 *
 * This class is responsible for parsing input tensor descriptors, validating their
 * shapes and data types for a GQA operation, and storing the relevant dimensions.
 */
class GQAInfo {
    // Private constructor to enforce creation via the static factory method.
    GQAInfo() = default;

public:
    // --- Member Variables ---
    int batch_size;
    int seq_len;
    int num_q_heads;
    int num_kv_heads;
    int head_size;
    infiniDtype_t data_type;

    /**
     * @brief Factory method to create and validate a GQAInfo object.
     *
     * @param q_desc Descriptor for the Query tensor (Q).
     * @param k_desc Descriptor for the Key tensor (K).
     * @param v_desc Descriptor for the Value tensor (V).
     * @param output_desc Descriptor for the Output tensor.
     * @return A Result object containing either the GQAInfo instance on success
     * or an error status on failure.
     */
    static utils::Result<GQAInfo>
    create(infiniopTensorDescriptor_t q_desc,
           infiniopTensorDescriptor_t k_desc,
           infiniopTensorDescriptor_t v_desc,
           infiniopTensorDescriptor_t output_desc) {

        // --- Data Type Validation ---
        auto data_type = q_desc->dtype();
        if (k_desc->dtype() != data_type || v_desc->dtype() != data_type ||
            output_desc->dtype() != data_type) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (data_type != INFINI_DTYPE_F16 && data_type != INFINI_DTYPE_BF16 &&
            data_type != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // --- Shape (NDim) Validation ---
        if (q_desc->ndim() != 4 || k_desc->ndim() != 4 ||
            v_desc->ndim() != 4 || output_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // --- Dimension Consistency Validation ---
        int batch_size = q_desc->shape()[0];
        int num_q_heads = q_desc->shape()[1];
        int seq_len = q_desc->shape()[2];
        int head_size = q_desc->shape()[3];
        int num_kv_heads = k_desc->shape()[1];
		std::cout << "--- C++ side: Received k_desc shape ---" << std::endl;
		const auto& k_shape_vec = k_desc->shape();
		std::cout << "k_desc->shape() has " << k_shape_vec.size() << " dimensions: [";
		for (size_t i = 0; i < k_shape_vec.size(); ++i) {
			std::cout << k_shape_vec[i] << (i == k_shape_vec.size() - 1 ? "" : ", ");
		}
		std::cout << "]" << std::endl;
		std::cout << "-----------------------------------------" << std::endl;
        // GQA constraint: num_q_heads must be a multiple of num_kv_heads
        if (num_q_heads % num_kv_heads != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check consistency of K, V, and Output tensors against Q
        if (k_desc->shape()[0] != (size_t)batch_size || k_desc->shape()[2] != (size_t)seq_len || k_desc->shape()[3] != (size_t)head_size ||
            v_desc->shape()[0] != (size_t)batch_size || v_desc->shape()[1] != (size_t)num_kv_heads || v_desc->shape()[2] != (size_t)seq_len || v_desc->shape()[3] != (size_t)head_size ||
            output_desc->shape()[0] != (size_t)batch_size || output_desc->shape()[1] != (size_t)num_q_heads || output_desc->shape()[2] != (size_t)seq_len || output_desc->shape()[3] != (size_t)head_size)
        {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // --- Stride Validation ---
        // Ensure the innermost dimension is contiguous for simple memory access patterns.
        if (q_desc->stride(3) != 1 || k_desc->stride(3) != 1 ||
            v_desc->stride(3) != 1 || output_desc->stride(3) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        // --- Success ---
        // If all checks pass, construct and return the info object.
        return utils::Result<GQAInfo>(GQAInfo{
            batch_size, seq_len, num_q_heads, num_kv_heads, head_size, data_type
        });
    }
};

} // namespace op::gqa

#endif // __GQA_INFO_H__
