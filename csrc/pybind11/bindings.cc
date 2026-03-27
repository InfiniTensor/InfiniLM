#include <pybind11/pybind11.h>

#include "cache/cache.hpp"
#include "engine/engine.hpp"
#include "models/llama.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_infinilm, m) {
    m.doc() = "InfiniLM Llama model Python bindings";

    infinilm::cache::bind_cache(m);
    infinilm::models::llama::bind_llama(m);
    infinilm::engine::distributed::bind_dist_config(m);
    infinilm::engine::bind_infer_engine(m);
}
