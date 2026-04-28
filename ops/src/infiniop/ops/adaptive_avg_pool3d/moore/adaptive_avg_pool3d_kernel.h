#ifndef ADAPTIVE_AVG_POOL3D_KERNEL_H_
#define ADAPTIVE_AVG_POOL3D_KERNEL_H_

template <typename T, unsigned int BLOCK_SIZE>
__device__ void adaptiveAvgPool3DKernel(T *y, const T *x, size_t N, size_t C,
                                        size_t x_d, size_t x_h, size_t x_w,
                                        size_t y_d, size_t y_h, size_t y_w,
                                        const ptrdiff_t *x_strides,
                                        const ptrdiff_t *y_strides) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * C * y_d * y_h * y_w;
    size_t n, c, od, oh, ow;
    n = index / (C * y_d * y_h * y_w);
    c = (index / (y_d * y_h * y_w)) % C;
    od = (index / (y_h * y_w)) % y_d;
    oh = (index / y_w) % y_h;
    ow = index % y_w;

    if (index < total) {
        size_t x_d_start = (od * x_d) / y_d;
        size_t x_d_end = ((od + 1) * x_d + y_d - 1) / y_d;
        size_t x_h_start = (oh * x_h) / y_h;
        size_t x_h_end = ((oh + 1) * x_h + y_h - 1) / y_h;
        size_t x_w_start = (ow * x_w) / y_w;
        size_t x_w_end = ((ow + 1) * x_w + y_w - 1) / y_w;

        T sum = static_cast<T>(0);
        size_t count = (x_d_end - x_d_start) * (x_h_end - x_h_start) * (x_w_end - x_w_start);
        for (size_t id = x_d_start; id < x_d_end; ++id) {
            for (size_t ih = x_h_start; ih < x_h_end; ++ih) {
                for (size_t iw = x_w_start; iw < x_w_end; ++iw) {
                    size_t x_index = n * x_strides[0] + c * x_strides[1] + id * x_strides[2] + ih * x_strides[3] + iw * x_strides[4];
                    sum += x[x_index];
                }
            }
        }
        size_t y_index = n * y_strides[0] + c * y_strides[1] + od * y_strides[2] + oh * y_strides[3] + ow * y_strides[4];
        y[y_index] = sum / static_cast<T>(count);
    }
}
#endif
