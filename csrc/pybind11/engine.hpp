#include "../engine/infer_engine.hpp"
#include "infinicore/tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinilm::engine::distributed {

inline void bind_dist_config(py::module &m) {
    py::class_<DistConfig>(m, "DistConfig")
        .def(py::init<>(), "Default constructor, empty device list")
        .def(py::init<int>(), py::arg("tp_size"),
             "Constructor with tensor parallel size, auto-assigns device IDs 0..tp_size-1")
        .def(py::init<const std::vector<int> &>(), py::arg("tp_device_ids"),
             "Constructor with explicit device IDs")
        .def_readwrite("tp_device_ids", &DistConfig::tp_device_ids,
                       "List of device IDs used in tensor parallelism")
        .def("__repr__", [](const DistConfig &cfg) {
            return std::string(cfg);
        })
        .def("__str__", [](const DistConfig &cfg) {
            return std::string(cfg);
        });
}

} // namespace infinilm::engine::distributed

namespace infinilm::engine {

inline void bind_infer_engine(py::module &m) {

    py::class_<InferEngine, std::shared_ptr<InferEngine>>(m, "InferEngine")
        .def(py::init([](const infinilm::models::llama::LlamaConfig &cfg,
                         const infinilm::engine::distributed::DistConfig &dist,
                         infinicore::Device::Type dev) {
                 return new InferEngine(std::any(cfg), dist, dev);
             }),
             py::arg("config"),
             py::arg("distributed_config") = distributed::DistConfig(),
             py::arg("device_type") = infinicore::context::getDevice().getType())
        .def("load_param", &InferEngine::load_param,
             py::arg("name"), py::arg("param"),
             "Load a parameter tensor into all workers (each worker picks its shard)")
        .def(
            "generate", [](InferEngine &self, py::object input_ids, py::object position_ids) -> infinicore::Tensor {
                return self.generate(input_ids.cast<infinicore::Tensor>(), position_ids.cast<infinicore::Tensor>());
            },
            "Run inference on all ranks with arbitrary arguments")
        .def("reset_cache", &InferEngine::reset_cache,
             py::arg("full_reset") = true,
             "Reset the internal cache in all workers (clears state between generations)");

    // Optionally, you can add __repr__ for debugging
    m.attr("InferEngine").attr("__repr__") = py::cpp_function([](const InferEngine &self) {
        return "<InferEngine: " + std::string(self.get_dist_config()) + ">";
    });
}

} // namespace infinilm::engine
