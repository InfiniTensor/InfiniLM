#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Scatter {
public:
    using schema = void (*)(Tensor, Tensor, int64_t, Tensor, Tensor, int64_t);

    static void execute(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor src, int64_t reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor scatter(Tensor input, int64_t dim, Tensor index, Tensor src, int64_t reduction = 0);

// In-place / 指定 Output 接口
void scatter_(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor src, int64_t reduction);

} // namespace infinicore::op
