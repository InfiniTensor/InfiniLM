#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {

class Kthvalue {
public:
    // Schema signature: values(out), indices(out), input, k, dim, keepdim
    using schema = void (*)(Tensor, Tensor, Tensor, int64_t, int64_t, bool);

    static void execute(Tensor values, Tensor indices, Tensor input, int64_t k, int64_t dim, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

// Functional API: Returns a tuple containing (values, indices)
std::tuple<Tensor, Tensor> kthvalue(Tensor input, int64_t k, int64_t dim = -1, bool keepdim = false);

// In-place/Output-provided API
void kthvalue_(Tensor values, Tensor indices, Tensor input, int64_t k, int64_t dim, bool keepdim);

} // namespace infinicore::op
