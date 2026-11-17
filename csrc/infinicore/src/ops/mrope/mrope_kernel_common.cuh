#ifndef __INFINIOP_MROPE_CUDA_KERNEL_CUH__
#define __INFINIOP_MROPE_CUDA_KERNEL_CUH__

#include <cstddef>
#include <cstdint>
#include <maca_fp16.h>
#include <maca_bfloat16.h>
using half = __half;
using cuda_bfloat16 = __maca_bfloat16;
#include <type_traits>

namespace op::mrope::cuda {

template <typename Tdata, typename Tangle>
__device__ inline Tangle loadAs(Tdata value) {
    return Tangle(value);
}

template <>
__device__ inline float loadAs<half, float>(half value) {
    return __half2float(value);
}

template <>
__device__ inline float loadAs<cuda_bfloat16, float>(cuda_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename Tdata, typename Tangle>
__device__ inline Tdata storeAs(Tangle value) {
    return Tdata(value);
}

template <>
__device__ inline half storeAs<half, float>(float value) {
    return __float2half(value);
}

template <>
__device__ inline cuda_bfloat16 storeAs<cuda_bfloat16, float>(float value) {
    return __float2bfloat16(value);
}

template <typename Tdata, typename Tangle, typename Tpos>
__device__ void rotateOne(
    Tdata *out,
    const Tdata *in,
    const Tdata *cos,
    const Tdata *sin,
    const Tpos *positions,
    size_t head_idx,
    size_t num_heads,
    size_t head_size,
    size_t rotary_dim,
    size_t half_rotary_dim,
    ptrdiff_t out_stride_token,
    ptrdiff_t out_stride_head,
    ptrdiff_t in_stride_token,
    ptrdiff_t in_stride_head,
    ptrdiff_t cos_stride_position,
    ptrdiff_t sin_stride_position,
    ptrdiff_t positions_stride_axis,
    ptrdiff_t positions_stride_token,
    size_t max_position_embeddings,
    size_t section_t,
    size_t section_h,
    size_t section_w,
    bool positions_has_axes,
    bool interleaved) {
    if (head_idx >= num_heads) {
        return;
    }
    const size_t token_idx = blockIdx.x;
    const ptrdiff_t out_offset = token_idx * out_stride_token + head_idx * out_stride_head;
    const ptrdiff_t in_offset = token_idx * in_stride_token + head_idx * in_stride_head;
    const size_t t_end = section_t;
    const size_t h_end = section_t + section_h;

    auto axis_for_dim = [=] __device__(size_t dim) {
        if (interleaved) {
            const size_t mod = dim % 3;
            const bool h_mask = mod == 1 && dim < section_h * 3;
            const bool w_mask = mod == 2 && dim < section_w * 3;
            return h_mask ? size_t(1) : (w_mask ? size_t(2) : size_t(0));
        }
        return dim < t_end ? size_t(0) : (dim < h_end ? size_t(1) : size_t(2));
    };

    for (size_t i = threadIdx.x; i < half_rotary_dim; i += blockDim.x) {
        const size_t axis = axis_for_dim(i);
        const ptrdiff_t pos_offset = positions_has_axes
                                       ? static_cast<ptrdiff_t>(axis) * positions_stride_axis + static_cast<ptrdiff_t>(token_idx) * positions_stride_token
                                       : static_cast<ptrdiff_t>(token_idx) * positions_stride_token;
        const int64_t raw_pos = static_cast<int64_t>(positions[pos_offset]);
        const size_t position = (raw_pos >= 0 && static_cast<size_t>(raw_pos) < max_position_embeddings) ? static_cast<size_t>(raw_pos) : 0;
        const ptrdiff_t table_offset = static_cast<ptrdiff_t>(position) * cos_stride_position + i;
        const Tangle cos_v = loadAs<Tdata, Tangle>(cos[table_offset]);
        const Tangle sin_v = loadAs<Tdata, Tangle>(sin[static_cast<ptrdiff_t>(position) * sin_stride_position + i]);
        const Tangle x0 = loadAs<Tdata, Tangle>(in[in_offset + i]);
        const Tangle x1 = loadAs<Tdata, Tangle>(in[in_offset + i + half_rotary_dim]);
        out[out_offset + i] = storeAs<Tdata, Tangle>(x0 * cos_v - x1 * sin_v);
        out[out_offset + i + half_rotary_dim] = storeAs<Tdata, Tangle>(x1 * cos_v + x0 * sin_v);
    }
    for (size_t dim = rotary_dim + threadIdx.x; dim < head_size; dim += blockDim.x) {
        out[out_offset + dim] = in[in_offset + dim];
    }
}

template <typename Tdata, typename Tangle, typename Tpos>
__device__ void mropeBlock(
    Tdata *q_out,
    Tdata *k_out,
    const Tdata *q,
    const Tdata *k,
    const Tdata *cos,
    const Tdata *sin,
    const Tpos *positions,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_size,
    size_t rotary_dim,
    size_t half_rotary_dim,
    ptrdiff_t q_out_stride_token,
    ptrdiff_t q_out_stride_head,
    ptrdiff_t k_out_stride_token,
    ptrdiff_t k_out_stride_head,
    ptrdiff_t q_stride_token,
    ptrdiff_t q_stride_head,
    ptrdiff_t k_stride_token,
    ptrdiff_t k_stride_head,
    ptrdiff_t cos_stride_position,
    ptrdiff_t sin_stride_position,
    ptrdiff_t positions_stride_axis,
    ptrdiff_t positions_stride_token,
    size_t max_position_embeddings,
    size_t section_t,
    size_t section_h,
    size_t section_w,
    bool positions_has_axes,
    bool interleaved) {
    const size_t head_idx = blockIdx.y;
    rotateOne<Tdata, Tangle, Tpos>(q_out, q, cos, sin, positions, head_idx, num_q_heads, head_size, rotary_dim, half_rotary_dim,
                                   q_out_stride_token, q_out_stride_head, q_stride_token, q_stride_head,
                                   cos_stride_position, sin_stride_position,
                                   positions_stride_axis, positions_stride_token, max_position_embeddings,
                                   section_t, section_h, section_w, positions_has_axes, interleaved);
    rotateOne<Tdata, Tangle, Tpos>(k_out, k, cos, sin, positions, head_idx, num_kv_heads, head_size, rotary_dim, half_rotary_dim,
                                   k_out_stride_token, k_out_stride_head, k_stride_token, k_stride_head,
                                   cos_stride_position, sin_stride_position,
                                   positions_stride_axis, positions_stride_token, max_position_embeddings,
                                   section_t, section_h, section_w, positions_has_axes, interleaved);
}

} // namespace op::mrope::cuda

#endif // __INFINIOP_MROPE_CUDA_KERNEL_CUH__
