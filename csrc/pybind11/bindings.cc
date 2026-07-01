#include <pybind11/pybind11.h>

#include "cache/cache.hpp"
#include "engine/engine.hpp"
#include "../models/backend_plugin_loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_infinilm, m) {
    m.doc() = "InfiniLM Python bindings";

    infinilm::cache::bind_cache(m);
    infinilm::engine::bind_hook_registry(m);
    infinilm::engine::distributed::bind_dist_config(m);
    infinilm::engine::bind_infer_engine(m);

    m.def("load_backend_plugin", &infinilm::models::load_backend_plugin,
          "Load one InfiniLM C++ backend plugin shared object.");
    m.def("load_backend_plugins_from_env", &infinilm::models::load_backend_plugins_from_env,
          "Load InfiniLM C++ backend plugins from INFINILM_BACKEND_PLUGINS.");
    m.def("loaded_backend_plugins", &infinilm::models::loaded_backend_plugins,
          "Return paths of loaded InfiniLM C++ backend plugins.");
}
