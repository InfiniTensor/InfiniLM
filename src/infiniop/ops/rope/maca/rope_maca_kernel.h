#ifndef __INFINIOP_ROPE_MACA_KERNEL_H__
#define __INFINIOP_ROPE_MACA_KERNEL_H__

#include "../../../devices/maca/maca_kernel_common.h"

template <typename Tdata, typename Tindex, typename Tangle>
INFINIOP_MACA_KERNEL ropeThreadPerItem(
    Tdata *y_,
    const Tdata *x_,
    const Tindex *__restrict__ pos_ids,
    const Tangle *__restrict__ sin_table,
    const Tangle *__restrict__ cos_table,
    size_t table_dim,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead) {

    auto y_offset = blockIdx.x * y_stride_seqlen + blockIdx.y * y_stride_nhead;
    auto x_offset = blockIdx.x * x_stride_seqlen + blockIdx.y * x_stride_nhead;
    size_t pos_id = size_t(pos_ids[blockIdx.x]);
    auto table_offset = pos_id * table_dim;

    for (size_t i = threadIdx.x; i < table_dim; i += blockDim.x) {
        Tangle sin__ = sin_table[table_offset + i],
               cos__ = cos_table[table_offset + i];
        if constexpr (std::is_same<Tdata, half>::value) {
            auto &y = reinterpret_cast<half2 &>(y_[y_offset + 2 * i]);
            auto &x = reinterpret_cast<const half2 &>(x_[x_offset + 2 * i]);
            Tangle y0 = x.x * cos__ - x.y * sin__,
                   y1 = x.x * sin__ + x.y * cos__;
            y = half2(y0, y1);
        } else {
            Tangle x0 = x_[x_offset + 2 * i],
                   x1 = x_[x_offset + 2 * i + 1];
            y_[y_offset + 2 * i] = Tdata(x0 * cos__ - x1 * sin__);
            y_[y_offset + 2 * i + 1] = Tdata(x0 * sin__ + x1 * cos__);
        }
    }
}

#endif
