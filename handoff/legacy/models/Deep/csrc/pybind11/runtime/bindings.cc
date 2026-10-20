#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "runtime/utils.hpp"
#include "context.hpp"
#include "device.hpp"
#include "dtype.hpp"
#include "ops/add.hpp"
#include "ops/cat.hpp"
#include "tensor.hpp"

namespace infinilm::runtime {

void bind_runtime(pybind11::module_ &m) {
    context::bind(m);
    device::bind(m);
    dtype::bind(m);
    ops::bind_add(m);
    ops::bind_cat(m);
    tensor::bind(m);
}

} // namespace infinilm::runtime
