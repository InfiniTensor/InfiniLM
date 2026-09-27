#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class IndexCopy {
public:
    using schema = void (*)(Tensor, Tensor, int64_t, Tensor, Tensor);
    static void execute(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor source);

    static common::OpDispatcher<schema> &dispatcher();
};
Tensor index_copy(Tensor input, int64_t dim, Tensor index, Tensor source);
void index_copy_(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor source);

} // namespace infinicore::op
