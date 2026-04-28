#ifndef __PER_TENSOR_DEQUANT_INT8_KERNEL_CUH__
#define __PER_TENSOR_DEQUANT_INT8_KERNEL_CUH__

template <typename Tin, typename Tout>
__device__ void perTensorDequantI8SymKernel(
    Tout *x, const Tin *x_packed, const float *x_scale,
    size_t batch_size, size_t channel, size_t hidden_dim, size_t width,
    ptrdiff_t strides_0, ptrdiff_t strides_1, ptrdiff_t strides_2, ptrdiff_t strides_3,
    ptrdiff_t p_strides_0, ptrdiff_t p_strides_1, ptrdiff_t p_strides_2, ptrdiff_t p_strides_3,
    int num_elements) {

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;
    float x_scale_val = x_scale[0];
    for (int ind = gid; ind < num_elements; ind += grid_size) {
        int tid = ind;
        int w = tid % (int)width;
        tid = tid / (int)width;

        int h = tid % (int)hidden_dim;
        tid = tid / (int)hidden_dim;

        int c = tid % (int)channel;
        tid = tid / (int)channel;

        int b = tid % (int)batch_size;

        int index = w * (int)strides_3 + h * (int)strides_2 + c * (int)strides_1 + b * (int)strides_0;
        int p_index = w * (int)p_strides_3 + h * (int)p_strides_2 + c * (int)p_strides_1 + b * (int)p_strides_0;

        float val = static_cast<float>(x_packed[p_index]) * x_scale_val;
        x[index] = static_cast<Tout>(val);
    }
}

#endif // __PER_TENSOR_DEQUANT_INT8_KERNEL_CUH__
