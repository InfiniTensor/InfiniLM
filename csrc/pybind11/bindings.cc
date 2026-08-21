#include "../layers/moe/kt_moe_callback.hpp"
#include <pybind11/pybind11.h>

#include "cache/cache.hpp"
#include "engine/engine.hpp"
#include <stdexcept>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_infinilm, m) {
    m.doc() = "InfiniLM Python bindings";

    infinilm::cache::bind_cache(m);
    infinilm::engine::bind_hook_registry(m);
    infinilm::engine::distributed::bind_dist_config(m);
    infinilm::engine::bind_infer_engine(m);

    // ---- KTransformers MoE offload integration ----
    // Register a Python callback (per layer) that receives
    // (hidden, routing_weights, topk_ids, layer_idx) as infinicore tensors
    // and must return the routed-expert output tensor (infinicore view).
    m.def(
        "set_kt_moe_callback", [](int layer_idx, py::function callback) {
            infinilm::layers::moe::KTMoECallbackRegistry::instance().set(layer_idx,
                                                                         [callback](const infinicore::Tensor &h, const infinicore::Tensor &w,
                                                                                    const infinicore::Tensor &i, int l) -> infinicore::Tensor {
                                                                             // Translate Python exceptions at the boundary while the GIL is
                                                                             // held: the C++ worker thread has no GIL and no Python frame to
                                                                             // unwind into, and py::error_already_set must not escape it.
                                                                             py::gil_scoped_acquire gil;
                                                                             try {
                                                                                 return callback(h, w, i, l).cast<infinicore::Tensor>();
                                                                             } catch (const py::error_already_set &e) {
                                                                                 throw std::runtime_error(
                                                                                     "KT MoE callback (layer " + std::to_string(l) + ") failed: " + e.what());
                                                                             }
                                                                         });
        },
        py::arg("layer_idx"), py::arg("callback"), "Register a KTransformers MoE callback for a given layer.");

    m.def(
        "clear_kt_moe_callbacks", []() {
            infinilm::layers::moe::KTMoECallbackRegistry::instance().clear();
        },
        "Clear all KT MoE callbacks.");

    // Release registered py::functions at module teardown (while the
    // interpreter is still alive) instead of process-exit static destruction.
    m.add_object("_kt_moe_cleanup",
                 py::capsule(reinterpret_cast<void *>(1), "_kt_moe_cleanup", [](void *) {
                     infinilm::layers::moe::KTMoECallbackRegistry::instance().clear();
                 }));
}
