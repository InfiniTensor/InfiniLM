#ifndef __UPSAMPLE_BILINEAR_INFO_H__
#define __UPSAMPLE_BILINEAR_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::upsample_bilinear {

class UpsampleBilinearInfo {
    UpsampleBilinearInfo() = default;

public:
    int _dtype;          // 数据类型
    bool _align_corners; // 是否对齐角点

    // 形状信息缓存
    // 通常 Upsample Bilinear 处理最后两个维度 (H, W)
    // 这里我们将前面的维度视为 Batch/Channel 的乘积，或者分别存储
    size_t _n;     // Batch Size (如果不适用则为 1)
    size_t _c;     // Channels (如果不适用则为 1)
    size_t _h_in;  // Input Height
    size_t _w_in;  // Input Width
    size_t _h_out; // Output Height
    size_t _w_out; // Output Width

    int dtype() const { return _dtype; }
    bool align_corners() const { return _align_corners; }
    size_t n() const { return _n; }
    size_t c() const { return _c; }
    size_t h_in() const { return _h_in; }
    size_t w_in() const { return _w_in; }
    size_t h_out() const { return _h_out; }
    size_t w_out() const { return _w_out; }

    // 构造函数
    UpsampleBilinearInfo(int dtype, bool align_corners,
                         size_t n, size_t c,
                         size_t h_in, size_t w_in,
                         size_t h_out, size_t w_out)
        : _dtype(dtype), _align_corners(align_corners),
          _n(n), _c(c),
          _h_in(h_in), _w_in(w_in),
          _h_out(h_out), _w_out(w_out) {}

    static utils::Result<UpsampleBilinearInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        int align_corners) { // C 接口通常传入 int 替代 bool

        // 1. 检查维度数量
        // 至少需要 2 维 (H, W)
        // 修复: 使用 size_t 避免与 ndim() 返回值比较时的 signed/unsigned 警告
        size_t ndim = input_desc->ndim();
        if (ndim < 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (out_desc->ndim() != ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 2. 检查数据类型
        // Input 和 Output 类型必须一致
        if (input_desc->dtype() != out_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 3. 检查 Batch/Channel 维度一致性
        // 除了最后两维 (H, W)，前面的维度必须完全匹配
        size_t n = 1;
        size_t c = 1;

        // 解析 N 和 C 用于 Info 缓存
        // 逻辑：
        // ndim = 4: [N, C, H, W] -> n=dims[0], c=dims[1]
        // ndim = 3: [C, H, W]    -> n=1,       c=dims[0]
        // ndim = 2: [H, W]       -> n=1,       c=1
        // 其他情况将所有非 spatial 维度累乘到 c 中 (视为 flattened channels)

        for (size_t i = 0; i < ndim - 2; ++i) { // 循环变量 i 也建议改为 size_t
            if (input_desc->shape()[i] != out_desc->shape()[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            // 简单 heuristic 来填充 n 和 c
            if (ndim == 4 && i == 0) {
                n = input_desc->shape()[i];
            } else if (ndim == 4 && i == 1) {
                c = input_desc->shape()[i];
            } else if (ndim == 3 && i == 0) {
                c = input_desc->shape()[i];
            } else {
                // 对于 >4 维的情况，简单地归约为 c
                c *= input_desc->shape()[i];
            }
        }

        // 4. 获取空间维度
        size_t h_in = input_desc->shape()[ndim - 2];
        size_t w_in = input_desc->shape()[ndim - 1];
        size_t h_out = out_desc->shape()[ndim - 2];
        size_t w_out = out_desc->shape()[ndim - 1];

        // 5. 零尺寸检查
        if (h_in == 0 || w_in == 0 || h_out == 0 || w_out == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<UpsampleBilinearInfo>(UpsampleBilinearInfo{
            input_desc->dtype(),
            static_cast<bool>(align_corners),
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out});
    }
};

} // namespace op::upsample_bilinear

#endif // __UPSAMPLE_BILINEAR_INFO_H__
