#ifndef __PER_CHANNEL_DEQUANT_INT8_KERNEL_CUH__
#define __PER_CHANNEL_DEQUANT_INT8_KERNEL_CUH__
/**
 * @brief Symmetric dequantization kernel for post-processing quantized matrix multiplication
 *
 * This kernel performs symmetric dequantization on the packed integer output from
 * a quantized matrix multiplication. It converts integer results back to floating-point
 * values by applying per-tensor scaling factors from both input and weight tensors,
 * then adds bias terms.
 *
 * The dequantization formula is:
 *   y = x_scale * w_scale * y_packed + bias
 *
 * @tparam Tdata Output data type (typically bfloat16 or half)
 *
 * @param[out] y Output tensor after dequantization
 *               Shape: [M, N], Data type: Tdata
 *
 * @param[in] y_packed Packed integer output from quantized matmul
 *                     Shape: [M, N], Data type: int32_t
 *                     Contains integer results of: x_packed[i,:] * w_packed[:,j]
 *
 * @param[in] bias Bias tensor to add after dequantization
 *                 Shape: [N], Data type: Tdata
 *                 Broadcasted across all rows
 *
 * @param[in] x_packed Packed quantized input tensor (not directly used here)
 *                     Shape: [M, K], Data type: int8_t
 *                     Included for context of the computation pipeline
 *
 * @param[in] x_scale Per-tensor scaling factors for input
 *                    Shape: [M], Data type: float
 *                    One scale value per input row
 *
 * @param[in] w_packed Packed quantized weight tensor (not directly used here)
 *                     Shape: [K, N], Data type: int8_t
 *                     Included for context of the computation pipeline
 *
 * @param[in] w_scale Per-tensor scaling factors for weights
 *                    Shape: [N], Data type: float
 *                    One scale value per output column
 *
 * @param[in] M Batch size / number of input rows
 *
 * @param[in] K Inner dimension of matrix multiplication
 *
 * @param[in] N Output dimension / number of output columns
 *
 * @note This kernel assumes symmetric quantization (zero-point = 0)
 * @note Each thread processes one element of the output matrix
 * @note Grid and block dimensions should be configured to cover [M, N] output space
 */
template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const Tdata *bias, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output1 = x_scale[row] * w_scale[col] * ((float)y_packed[idx]);

    float output = output1 + (float)bias[col];

    y[idx] = static_cast<Tdata>(output);
}
// y = x_scale * w_scale * y_packed
template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output = x_scale[row] * w_scale[col] * ((float)y_packed[idx]);

    y[idx] = static_cast<Tdata>(output);
}
#endif // __PER_CHANNEL_DEQUANT_INT8_KERNEL_CUH__
