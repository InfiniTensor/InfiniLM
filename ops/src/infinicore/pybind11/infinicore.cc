#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../utils.hpp"
#include "context.hpp"
#include "device.hpp"
#include "device_event.hpp"
#include "dtype.hpp"
#include "graph.hpp"
#include "ops.hpp"
#include "tensor.hpp"

namespace infinicore {

PYBIND11_MODULE(_infinicore, m) {
    context::bind(m);
    device::bind(m);
    device_event::bind(m);
    dtype::bind(m);
    ops::bind(m);
    tensor::bind(m);
    graph::bind(m);
}

} // namespace infinicore
