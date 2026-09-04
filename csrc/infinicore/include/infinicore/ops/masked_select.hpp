#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class MaskedSelect {
public:
    using schema = void (*)(Tensor, Tensor, void **, size_t *);
    static void execute(Tensor input, Tensor mask, void **data_ptr, size_t *dlen_ptr);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor masked_select(Tensor input, Tensor mask);

} // namespace infinicore::op
