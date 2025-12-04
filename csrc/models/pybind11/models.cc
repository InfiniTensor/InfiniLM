#include "models/llama.hpp"
#include "wrapper/wrapper.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_infinilm_llama, m) {
    m.doc() = "InfiniLM Llama model Python bindings";

    infinilm::models::llama::bind_llama(m);
    infinilm::models::llama::bind_wrapper(m);
}
