#ifndef __INFINIOP_ROPE_CUDA_KERNEL_CUH__
#define __INFINIOP_ROPE_CUDA_KERNEL_CUH__

template <bool IsGPTJ, typename Tdata, typename Tindex, typename Tangle>
__device__ void ropeThreadPerItemBlock(
    Tdata *y_,
    const Tdata *x_,
    const Tindex *__restrict__ pos_ids,
    const Tangle *__restrict__ sin_table,
    const Tangle *__restrict__ cos_table,
    size_t table_dim,
    size_t pos_stride_batch, // Stride for batch dimension in pos_ids (0 if 1D)
    bool pos_has_batch_dim,  // Whether pos_ids has batch dimension
    bool has_batch_dim,      // Whether tensors have batch dimension
    ptrdiff_t y_stride_batch,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t x_stride_batch,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead) {

    // Calculate batch index: use blockIdx.z for 4D, 0 for 3D
    const size_t batch_idx = has_batch_dim ? blockIdx.z : 0;
    const size_t seq_idx = blockIdx.x;
    const size_t head_idx = blockIdx.y;

    // Calculate memory offsets
    auto y_offset = (has_batch_dim ? batch_idx * y_stride_batch : 0) + seq_idx * y_stride_seqlen + head_idx * y_stride_nhead;

    auto x_offset = (has_batch_dim ? batch_idx * x_stride_batch : 0) + seq_idx * x_stride_seqlen + head_idx * x_stride_nhead;

    // Calculate position ID offset
    size_t pos_offset;
    if (pos_has_batch_dim) {
        // Per-batch position IDs
        pos_offset = batch_idx * pos_stride_batch + seq_idx;
    } else {
        // Shared position IDs across batch
        pos_offset = seq_idx;
    }

    size_t pos_id = size_t(pos_ids[pos_offset]);
    auto table_offset = pos_id * table_dim;

    for (size_t i = threadIdx.x; i < table_dim; i += blockDim.x) {
        Tangle sin__ = sin_table[table_offset + i],
               cos__ = cos_table[table_offset + i];

        if constexpr (IsGPTJ) {
            if constexpr (std::is_same<Tdata, half>::value) {
                auto &y = reinterpret_cast<half2 &>(y_[y_offset + 2 * i]);
                auto &x = reinterpret_cast<const half2 &>(x_[x_offset + 2 * i]);
                Tangle y0 = x.x * cos__ - x.y * sin__,
                       y1 = x.x * sin__ + x.y * cos__;
                y = half2(y0, y1);
            } else if constexpr (std::is_same<Tdata, cuda_bfloat16>::value) {
                auto &y = reinterpret_cast<cuda_bfloat162 &>(y_[y_offset + 2 * i]);
                auto &x = reinterpret_cast<const cuda_bfloat162 &>(x_[x_offset + 2 * i]);

                Tangle x0 = __low2bfloat16(x);
                Tangle x1 = __high2bfloat16(x);

                Tangle y0 = x0 * cos__ - x1 * sin__;
                Tangle y1 = x0 * sin__ + x1 * cos__;

                y = __floats2bfloat162_rn(y0, y1);
            } else {
                Tangle x0 = x_[x_offset + 2 * i],
                       x1 = x_[x_offset + 2 * i + 1];
                y_[y_offset + 2 * i] = Tdata(x0 * cos__ - x1 * sin__);
                y_[y_offset + 2 * i + 1] = Tdata(x0 * sin__ + x1 * cos__);
            }
        } else {
            size_t pos0 = i;
            size_t pos1 = i + table_dim;

            if constexpr (std::is_same<Tdata, half>::value) {
                Tangle x0 = __half2float(x_[x_offset + pos0]);
                Tangle x1 = __half2float(x_[x_offset + pos1]);

                Tangle y0 = x0 * cos__ - x1 * sin__;
                Tangle y1 = x0 * sin__ + x1 * cos__;

                y_[y_offset + pos0] = __float2half(y0);
                y_[y_offset + pos1] = __float2half(y1);
            } else if constexpr (std::is_same<Tdata, cuda_bfloat16>::value) {
                Tangle x0 = __bfloat162float(x_[x_offset + pos0]);
                Tangle x1 = __bfloat162float(x_[x_offset + pos1]);

                Tangle y0 = x0 * cos__ - x1 * sin__;
                Tangle y1 = x0 * sin__ + x1 * cos__;

                y_[y_offset + pos0] = __float2bfloat16(y0);
                y_[y_offset + pos1] = __float2bfloat16(y1);
            } else {
                Tangle x0 = x_[x_offset + pos0];
                Tangle x1 = x_[x_offset + pos1];

                y_[y_offset + pos0] = x0 * cos__ - x1 * sin__;
                y_[y_offset + pos1] = x0 * sin__ + x1 * cos__;
            }
        }
    }
}

#endif
