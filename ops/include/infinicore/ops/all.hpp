#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <vector>
namespace infinicore::op {
class All {
public:
    using schema = void (*)(Tensor, Tensor, std::vector<size_t>, bool);
    static void execute(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim = false);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor all(Tensor input, std::vector<size_t> dim, bool keepdim = false);
void all_(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim = false);

} // namespace infinicore::op
