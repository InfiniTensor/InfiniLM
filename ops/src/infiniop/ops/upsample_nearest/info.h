#ifndef __UPSAMPLE_NEAREST_INFO_H__
#define __UPSAMPLE_NEAREST_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::upsample_nearest {

class UpsampleNearestInfo {
    UpsampleNearestInfo() = default;

public:
    int _dtype;
    size_t _n;
    size_t _c;
    size_t _h_in;
    size_t _w_in;
    size_t _h_out;
    size_t _w_out;

    int dtype() const { return _dtype; }
    size_t n() const { return _n; }
    size_t c() const { return _c; }
    size_t h_in() const { return _h_in; }
    size_t w_in() const { return _w_in; }
    size_t h_out() const { return _h_out; }
    size_t w_out() const { return _w_out; }

    UpsampleNearestInfo(int dtype,
                        size_t n, size_t c,
                        size_t h_in, size_t w_in,
                        size_t h_out, size_t w_out)
        : _dtype(dtype),
          _n(n), _c(c),
          _h_in(h_in), _w_in(w_in),
          _h_out(h_out), _w_out(w_out) {}

    static utils::Result<UpsampleNearestInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc) {

        size_t ndim = input_desc->ndim();
        // 允许 3D (N, C, W) 和 4D (N, C, H, W)
        if (ndim < 3 || ndim > 4) {
            // 如果为了兼容性，也可以保留 ndim=2 的逻辑，但通常 upsample 至少有 batch/channel
            if (ndim != 2 && ndim != 3 && ndim != 4) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
        if (out_desc->ndim() != ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->dtype() != out_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        size_t n = 1;
        size_t c = 1;
        size_t h_in = 1, w_in = 1;
        size_t h_out = 1, w_out = 1;

        if (ndim == 3) {
            // Case: [N, C, W] -> Treat as H=1
            n = input_desc->shape()[0];
            c = input_desc->shape()[1];
            w_in = input_desc->shape()[2];

            // 检查输出维度一致性
            if (out_desc->shape()[0] != n || out_desc->shape()[1] != c) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            w_out = out_desc->shape()[2];

            // H 固定为 1
            h_in = 1;
            h_out = 1;
        } else if (ndim == 4) {
            // Case: [N, C, H, W]
            n = input_desc->shape()[0];
            c = input_desc->shape()[1];
            h_in = input_desc->shape()[2];
            w_in = input_desc->shape()[3];

            if (out_desc->shape()[0] != n || out_desc->shape()[1] != c) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            h_out = out_desc->shape()[2];
            w_out = out_desc->shape()[3];
        } else {
            // Fallback for ndim=2 or others, previous logic
            // Assuming [H, W] or similar
            for (size_t i = 0; i < ndim - 2; ++i) {
                if (input_desc->shape()[i] != out_desc->shape()[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
                c *= input_desc->shape()[i];
            }
            h_in = input_desc->shape()[ndim - 2];
            w_in = input_desc->shape()[ndim - 1];
            h_out = out_desc->shape()[ndim - 2];
            w_out = out_desc->shape()[ndim - 1];
        }

        if (h_in == 0 || w_in == 0 || h_out == 0 || w_out == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<UpsampleNearestInfo>(UpsampleNearestInfo{
            input_desc->dtype(),
            n, c,
            h_in, w_in,
            h_out, w_out});
    }
};

} // namespace op::upsample_nearest

#endif // __UPSAMPLE_NEAREST_INFO_H__
