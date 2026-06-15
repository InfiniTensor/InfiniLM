#include <pybind11/pybind11.h>

#include "cache/cache.hpp"
#include "engine/engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_infinilm, m) {
    m.doc() = "InfiniLM Python bindings";

    infinilm::cache::bind_cache(m);
    infinilm::engine::bind_hook_registry(m);
    infinilm::engine::distributed::bind_dist_config(m);
    infinilm::engine::bind_infer_engine(m);
}
