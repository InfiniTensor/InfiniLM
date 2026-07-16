#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <utility>
#include <vector>
namespace infinicore::op {
class Var_Mean {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, std::vector<size_t>, bool, bool); // var_output, mean_output, input, dim, unbiased, keepdim
    static void execute(Tensor var_output, Tensor mean_output, Tensor input, std::vector<size_t> dim, bool unbiased = true, bool keepdim = false);
    static common::OpDispatcher<schema> &dispatcher();
};

std::pair<Tensor, Tensor> var_mean(Tensor input, std::vector<size_t> dim, bool unbiased = true, bool keepdim = false);
void var_mean_(Tensor var_output, Tensor mean_output, Tensor input, std::vector<size_t> dim, bool unbiased = true, bool keepdim = false);

} // namespace infinicore::op
