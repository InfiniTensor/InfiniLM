#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/cat.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/linear.hpp"
#include "ops/matmul.hpp"
#include "ops/random_sample.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/silu_and_mul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_cat(m);
    bind_causal_softmax(m);
    bind_embedding(m);
    bind_linear(m);
    bind_matmul(m);
    bind_random_sample(m);
    bind_rms_norm(m);
    bind_rope(m);
    bind_silu(m);
    bind_silu_and_mul(m);
}

} // namespace infinicore::ops
