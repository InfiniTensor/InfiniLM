#include <pybind11/pybind11.h>

#include "models/llama.hpp"

#include "engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_infinilm, m) {
    m.doc() = "InfiniLM Llama model Python bindings";

    infinilm::models::llama::bind_llama(m);
    infinilm::engine::distributed::bind_dist_config(m);
    infinilm::engine::bind_infer_engine(m);
}
