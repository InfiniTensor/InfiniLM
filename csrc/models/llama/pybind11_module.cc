#include <pybind11/pybind11.h>
#include "pybind11_llama.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_infinilm_llama, m) {
    m.doc() = "InfiniLM Llama model Python bindings";

    infinilm::models::llama::bind_llama(m);
}
