#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class IndexAdd {
public:
    using schema = void (*)(Tensor, Tensor, int64_t, Tensor, Tensor, float);
    static void execute(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor source, float alpha);

    static common::OpDispatcher<schema> &dispatcher();
};

Tensor index_add(Tensor input, int64_t dim, Tensor index, Tensor source, float alpha = 1.0f);
void index_add_(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor source, float alpha);

} // namespace infinicore::op
