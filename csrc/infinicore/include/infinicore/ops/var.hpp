#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <utility>
#include <vector>
namespace infinicore::op {
class Var {
public:
    using schema = void (*)(Tensor, Tensor, std::vector<size_t>, bool, bool); // var_output, input, dim, unbiased, keepdim
    static void execute(Tensor var_output, Tensor input, std::vector<size_t> dim, bool unbiased = true, bool keepdim = false);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor var(Tensor input, std::vector<size_t> dim, bool unbiased = true, bool keepdim = false);
void var_(Tensor var_output, Tensor input, std::vector<size_t> dim, bool unbiased = true, bool keepdim = false);

} // namespace infinicore::op
