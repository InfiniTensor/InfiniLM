#include "../../../../utils.h"
#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "../../../tensor.h"
#include "embedding_moore_kernel.h"
#include "embedding_moore.h"
#include <musa_runtime.h>

template <typename T, typename IndexType>
INFINIOP_MOORE_KERNEL embeddingKernel(
    T *__restrict__ output,
    const IndexType *__restrict__ indices,
    const T *__restrict__ weight,
    size_t num_indices,
    size_t embedding_dim,
    size_t vocab_size) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_indices) {
        // Get the index value with Moore-optimized memory access
        IndexType index_val = indices[idx];

        // Bounds check - handle negative indices gracefully
        if (index_val >= 0 && static_cast<size_t>(index_val) < vocab_size) {
            // Copy embedding vector from weight to output
            const T *src = weight + static_cast<size_t>(index_val) * embedding_dim;
            T *dst = output + idx * embedding_dim;

            // Choose optimal copy strategy based on type and alignment
            if constexpr (std::is_same_v<T, float>) {
                // Check alignment for float4 (16 bytes)
                bool aligned_16 = is_aligned(src, 16) && is_aligned(dst, 16);
                if (aligned_16 && embedding_dim >= 4 && embedding_dim % 4 == 0) {
                    copyVectorizedFloat4<IndexType>(dst, src, embedding_dim);
                } else if (embedding_dim >= 2 && embedding_dim % 2 == 0) {
                    // Try float2 if not aligned to 16 bytes
                    copyVectorizedFloat2<IndexType>(dst, src, embedding_dim);
                } else {
                    copyScalar<T, IndexType>(dst, src, embedding_dim);
                }
            } else if constexpr (std::is_same_v<T, half>) {
                // Use half2 for vectorized access
                if (embedding_dim >= 2 && embedding_dim % 2 == 0) {
                    copyVectorizedHalf2<IndexType>(dst, src, embedding_dim);
                } else {
                    copyScalar<T, IndexType>(dst, src, embedding_dim);
                }
            } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                // Use mt_bfloat162 for vectorized access (Moore-specific type)
                if (embedding_dim >= 2 && embedding_dim % 2 == 0) {
                    copyVectorizedBFloat162<IndexType>(dst, src, embedding_dim);
                } else {
                    copyScalar<T, IndexType>(dst, src, embedding_dim);
                }
            } else {
                // Fallback to scalar copy with Moore-optimized memory access
                copyScalar<T, IndexType>(dst, src, embedding_dim);
            }
        }
    }
}

namespace op::embedding::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc) {

    auto input_shape = input_desc->shape();
    auto weight_shape = weight_desc->shape();

    // Validate shapes
    CHECK_OR_RETURN(weight_shape.size() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->shape().size() == input_shape.size() + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

    // Check output shape matches input shape + embedding_dim
    auto output_shape = output_desc->shape();
    size_t embedding_dim = weight_shape[1];
    CHECK_OR_RETURN(output_shape.back() == embedding_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);

    for (size_t i = 0; i < input_shape.size(); ++i) {
        CHECK_OR_RETURN(output_shape[i] == input_shape[i], INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    // Validate dtypes
    auto input_dtype = input_desc->dtype();
    auto weight_dtype = weight_desc->dtype();
    CHECK_OR_RETURN(input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(weight_dtype == INFINI_DTYPE_F32 || weight_dtype == INFINI_DTYPE_F16 || weight_dtype == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->dtype() == weight_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    // Calculate number of indices (supporting batch dimension)
    size_t num_indices = 1;
    for (auto dim : input_shape) {
        num_indices *= dim;
    }

    size_t vocab_size = weight_shape[0];

    *desc_ptr = new Descriptor(
        num_indices,
        embedding_dim,
        vocab_size,
        input_dtype,
        weight_dtype,
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    const void *input,
    const void *weight,
    void *stream) const {

    if (_num_indices == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    // Dynamic block size optimization based on embedding_dim
    // Moore platform typically has different performance characteristics
    size_t block_size = 256; // Default for Moore
    if (_embedding_dim <= 64) {
        block_size = 512; // Small embedding_dim: use larger block for better occupancy
    } else if (_embedding_dim >= 1024) {
        block_size = 128; // Large embedding_dim: use smaller block to reduce register pressure
    } else if (_embedding_dim <= 256) {
        block_size = 384; // Medium embedding_dim: balanced configuration
    }

    size_t grid_size = (_num_indices + block_size - 1) / block_size;

    // Launch kernel based on dtypes
    // Note: Moore uses __mt_bfloat16 instead of __nv_bfloat16
    if (_input_dtype == INFINI_DTYPE_I32) {
        const int32_t *indices_ptr = reinterpret_cast<const int32_t *>(input);

        if (_weight_dtype == INFINI_DTYPE_F32) {
            embeddingKernel<float, int32_t><<<grid_size, block_size, 0, musa_stream>>>(
                reinterpret_cast<float *>(output),
                indices_ptr,
                reinterpret_cast<const float *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_F16) {
            embeddingKernel<half, int32_t><<<grid_size, block_size, 0, musa_stream>>>(
                reinterpret_cast<half *>(output),
                indices_ptr,
                reinterpret_cast<const half *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_BF16) {
            // Use Moore's bfloat16 type
            embeddingKernel<__mt_bfloat16, int32_t><<<grid_size, block_size, 0, musa_stream>>>(
                reinterpret_cast<__mt_bfloat16 *>(output),
                indices_ptr,
                reinterpret_cast<const __mt_bfloat16 *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_input_dtype == INFINI_DTYPE_I64) {
        const int64_t *indices_ptr = reinterpret_cast<const int64_t *>(input);

        if (_weight_dtype == INFINI_DTYPE_F32) {
            embeddingKernel<float, int64_t><<<grid_size, block_size, 0, musa_stream>>>(
                reinterpret_cast<float *>(output),
                indices_ptr,
                reinterpret_cast<const float *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_F16) {
            embeddingKernel<half, int64_t><<<grid_size, block_size, 0, musa_stream>>>(
                reinterpret_cast<half *>(output),
                indices_ptr,
                reinterpret_cast<const half *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_BF16) {
            embeddingKernel<__mt_bfloat16, int64_t><<<grid_size, block_size, 0, musa_stream>>>(
                reinterpret_cast<__mt_bfloat16 *>(output),
                indices_ptr,
                reinterpret_cast<const __mt_bfloat16 *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check for kernel launch errors
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::embedding::moore
