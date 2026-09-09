#pragma once

#include "../device.hpp"
#include "common/op.hpp"
namespace infinicore::op {
class TopK {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, size_t, size_t, bool, bool);
    static void execute(Tensor values_output, Tensor indices_output, Tensor input, size_t k, size_t dim, bool largest = true, bool sorted = true);
    static common::OpDispatcher<schema> &dispatcher();
};

std::pair<Tensor, Tensor> topk(Tensor input, size_t k, size_t dim, bool largest = true, bool sorted = true);
void topk_(Tensor values_output, Tensor indices_output, Tensor input, size_t k, size_t dim, bool largest = true, bool sorted = true);

} // namespace infinicore::op
