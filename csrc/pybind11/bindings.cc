#include <pybind11/pybind11.h>

#include "cache/cache.hpp"
#include "engine/engine.hpp"
#include "infinicore/pybind11/context.hpp"
#include "infinicore/pybind11/device.hpp"
#include "infinicore/pybind11/device_event.hpp"
#include "infinicore/pybind11/dtype.hpp"
#include "infinicore/pybind11/graph.hpp"
#include "infinicore/pybind11/tensor.hpp"
#include "models/llama_legacy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_infinilm, m) {
    m.doc() = "InfiniLM Llama model Python bindings";

    infinicore::context::bind(m);
    infinicore::device::bind(m);
    infinicore::device_event::bind(m);
    infinicore::dtype::bind(m);
    infinicore::tensor::bind(m);
    infinicore::graph::bind(m);
    infinilm::cache::bind_cache(m);
    infinilm::models::llama_legacy::bind_llama(m);
    infinilm::engine::distributed::bind_dist_config(m);
    infinilm::engine::bind_infer_engine(m);
}
