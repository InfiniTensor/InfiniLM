#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <cstddef>
#include <vector>

namespace infinicore::op {
class Conv2d {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor,
                            const size_t *, const size_t *, const size_t *, size_t);
    static void execute(Tensor output,
                        Tensor input,
                        Tensor weight,
                        Tensor bias,
                        const size_t *pads,
                        const size_t *strides,
                        const size_t *dilations,
                        size_t n);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor conv2d(Tensor input,
              Tensor weight,
              Tensor bias,
              const std::vector<size_t> &pads,
              const std::vector<size_t> &strides,
              const std::vector<size_t> &dilations);
void conv2d_(Tensor output,
             Tensor input,
             Tensor weight,
             Tensor bias,
             const std::vector<size_t> &pads,
             const std::vector<size_t> &strides,
             const std::vector<size_t> &dilations);
} // namespace infinicore::op
