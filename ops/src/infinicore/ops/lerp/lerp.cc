#include "infinicore/ops/lerp.hpp"
#include <algorithm> // for std::max
#include <stdexcept> // for std::runtime_error
#include <string>

namespace infinicore::op {

// ========================================================================
// 0. 内部辅助函数：手动实现形状广播推导
// ========================================================================
namespace {

Shape compute_broadcast_shape(const std::vector<Shape> &shapes) {
    if (shapes.empty()) {
        return {};
    }

    // 1. 找出最大的维度数 (Max Rank)
    size_t max_ndim = 0;
    for (const auto &shape : shapes) {
        max_ndim = std::max(max_ndim, shape.size());
    }

    Shape out_shape(max_ndim);

    // 2. 从右向左遍历每一个维度 (Standard Broadcasting Rule)
    for (size_t i = 0; i < max_ndim; ++i) {
        size_t current_dim_val = 1;
        bool set = false;

        for (const auto &shape : shapes) {
            // 计算当前 shape 对应的维度索引 (从右对齐)
            // 比如 max_ndim=4, 当前 shape_ndim=2, i=0 (最右边)
            // shape index = 2 - 1 - 0 = 1
            if (i < shape.size()) {
                size_t dim = shape[shape.size() - 1 - i];

                if (dim == 1) {
                    continue; // 1 可以被广播，忽略
                }

                if (!set) {
                    current_dim_val = dim;
                    set = true;
                } else if (current_dim_val != dim) {
                    // 维度不相等，且都不为 1，无法广播
                    throw std::runtime_error(
                        "Lerp: Shapes are not broadcastable. Mismatch at dimension offset " + std::to_string(i));
                }
            }
        }
        // 填充输出形状 (从右向左填，或者填好后由 vector 自动管理)
        out_shape[max_ndim - 1 - i] = current_dim_val;
    }

    return out_shape;
}

} // namespace

// ========================================================================
// 1. 定义 Dispatcher 单例
// ========================================================================

template <>
common::OpDispatcher<Lerp::schema_t> &Lerp::dispatcher<Lerp::schema_t>() {
    static common::OpDispatcher<Lerp::schema_t> dispatcher_;
    return dispatcher_;
}

template <>
common::OpDispatcher<Lerp::schema_s> &Lerp::dispatcher<Lerp::schema_s>() {
    static common::OpDispatcher<Lerp::schema_s> dispatcher_;
    return dispatcher_;
}

// ========================================================================
// 2. Execute 静态方法实现
// ========================================================================

void Lerp::execute(Tensor output, Tensor start, Tensor end, Tensor weight) {
    dispatcher<schema_t>().lookup(context::getDevice().getType())(output, start, end, weight);
}

void Lerp::execute(Tensor output, Tensor start, Tensor end, float weight) {
    dispatcher<schema_s>().lookup(context::getDevice().getType())(output, start, end, weight);
}

// ========================================================================
// 3. 函数式接口 (Functional API) - 集成形状推导
// ========================================================================

Tensor lerp(Tensor start, Tensor end, Tensor weight) {
    // 1. 调用本地实现的推导函数，计算 start, end, weight 三者的广播形状
    Shape output_shape = compute_broadcast_shape({start->shape(),
                                                  end->shape(),
                                                  weight->shape()});

    // 2. 分配输出内存
    auto output = Tensor::empty(output_shape, start->dtype(), start->device());

    // 3. 执行计算
    lerp_(output, start, end, weight);
    return output;
}

Tensor lerp(Tensor start, Tensor end, float weight) {
    // 1. 计算 start, end 两者的广播形状 (标量 weight 不参与形状计算)
    Shape output_shape = compute_broadcast_shape({start->shape(),
                                                  end->shape()});

    // 2. 分配输出内存
    auto output = Tensor::empty(output_shape, start->dtype(), start->device());

    // 3. 执行计算
    lerp_(output, start, end, weight);
    return output;
}

// ========================================================================
// 4. In-place / Output-buffer 接口
// ========================================================================

void lerp_(Tensor output, Tensor start, Tensor end, Tensor weight) {
    Lerp::execute(output, start, end, weight);
}

void lerp_(Tensor output, Tensor start, Tensor end, float weight) {
    Lerp::execute(output, start, end, weight);
}

} // namespace infinicore::op
