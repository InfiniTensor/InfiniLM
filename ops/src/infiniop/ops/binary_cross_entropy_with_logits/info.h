#ifndef __BINARY_CROSS_ENTROPY_WITH_LOGITS_INFO_H__
#define __BINARY_CROSS_ENTROPY_WITH_LOGITS_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/binary_cross_entropy_with_logits.h"
#include <numeric>
#include <vector>

namespace op::bce_with_logits {

/**
 * 描述 BCE 算子中各张量的内存布局
 * 动态申请 dims 和 stride，支持任意维度的张量
 */
struct BCETensorInfo {
    size_t total_elements = 0;
    size_t ndim = 0;
    std::vector<size_t> dims;      // 动态存储维度
    std::vector<ptrdiff_t> stride; // 动态存储步长

    BCETensorInfo() = default;

    static utils::Result<BCETensorInfo> create(infiniopTensorDescriptor_t desc) {
        if (desc == nullptr) {
            return INFINI_STATUS_SUCCESS;
        }

        BCETensorInfo info;
        info.ndim = desc->ndim();
        info.total_elements = 1;

        // 动态调整 vector 大小
        info.dims.reserve(info.ndim);
        info.stride.reserve(info.ndim);

        for (size_t i = 0; i < info.ndim; ++i) {
            size_t d = desc->dim(i);
            info.dims.push_back(d);
            info.stride.push_back(desc->stride(i));
            info.total_elements *= d;
        }
        return utils::Result<BCETensorInfo>(std::move(info));
    }

    // 辅助方法：获取最后一维大小（用于 pos_weight 校验）
    size_t last_dim() const {
        return ndim > 0 ? dims.back() : 0;
    }
};

class BCEWithLogitsInfo {
public:
    BCETensorInfo logits;
    BCETensorInfo target;
    BCETensorInfo weight;
    BCETensorInfo pos_weight;
    BCETensorInfo out;

    size_t num_elements;
    infiniopReduction_t reduction;

    // 由于 BCETensorInfo 内部使用了 vector，BCEWithLogitsInfo 现在是可移动且安全的
    BCEWithLogitsInfo() = default;

    static utils::Result<BCEWithLogitsInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t logits_desc,
        infiniopTensorDescriptor_t target_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t pos_weight_desc,
        infiniopReduction_t reduction) {

        auto logits_res = BCETensorInfo::create(logits_desc);
        CHECK_RESULT(logits_res);
        auto target_res = BCETensorInfo::create(target_desc);
        CHECK_RESULT(target_res);
        auto out_res = BCETensorInfo::create(out_desc);
        CHECK_RESULT(out_res);

        BCEWithLogitsInfo info;
        info.logits = logits_res.take();
        info.target = target_res.take();
        info.out = out_res.take();
        info.reduction = reduction;
        info.num_elements = info.logits.total_elements;

        // 1. 基本形状一致性校验
        if (info.logits.ndim != info.target.ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        for (size_t i = 0; i < info.logits.ndim; ++i) {
            if (info.logits.dims[i] != info.target.dims[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 2. 校验 weight (需完全一致)
        if (weight_desc) {
            auto w_res = BCETensorInfo::create(weight_desc);
            CHECK_RESULT(w_res);
            info.weight = w_res.take();

            // 允许两种情况：
            // 1. 完全一致
            // 2. weight 是一个向量，且长度等于 logits 的最后一维 (常见广播场景)
            bool is_full_match = (info.weight.total_elements == info.logits.total_elements);
            bool is_last_dim_match = (info.weight.total_elements == info.logits.last_dim());

            if (!is_full_match && !is_last_dim_match) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 3. 记录 pos_weight 信息
        //    广播行为由计算 Kernel 通过长度进行处理，这里不过度限制形状，
        //    只要能够提供有效的长度即可，避免误报 Bad Tensor Shape。
        if (pos_weight_desc) {
            auto pw_res = BCETensorInfo::create(pos_weight_desc);
            CHECK_RESULT(pw_res);
            info.pos_weight = pw_res.take();
        }

        // 4. 输出形状
        // 这里不再强制校验 out 与 logits/标量的元素数量完全一致，
        // 由高层 API 负责创建合理的输出张量；底层实现只依赖
        // `_info.out` 的 stride 在 reduction==NONE 且逐元素写回时使用。

        return utils::Result<BCEWithLogitsInfo>(std::move(info));
    }
};

} // namespace op::bce_with_logits

#endif // __BINARY_CROSS_ENTROPY_WITH_LOGITS_INFO_H__
