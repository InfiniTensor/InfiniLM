#include "../../engine/infer_engine.hpp"
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
    py::class_<InferEngine, std::shared_ptr<InferEngine>> infer_engine(m, "InferEngine");
    infer_engine
        .def(py::init([](
                          const InfinilmModel::Config &cfg,
                          const distributed::DistConfig &dist,
                          infinicore::Device::Type dev,
                          std::shared_ptr<const infinilm::cache::CacheConfig> cache_cfg) {
                 return std::make_shared<InferEngine>(
                     cfg,
                     dist,
                     dev,
                     cache_cfg ? cache_cfg.get() : nullptr);
             }),
             py::arg("config"),
             py::arg("distributed_config") = distributed::DistConfig(),
             py::arg("device_type") = infinicore::context::getDevice().getType(),
             py::arg("cache_config") = py::none())
    .def("load_param", &InferEngine::load_param,
         py::arg("name"), py::arg("param"),
         "Load a parameter tensor into all workers (each worker picks its shard)")
        .def("state_dict", [](InferEngine &self) {
            py::list state_dict_tp_all;
            for (const auto &state_dict_tp : self.state_dict()) {
                py::dict result;
                for (const auto &[name, param] : state_dict_tp) {
                    result[py::cast(name)] = infinicore::Tensor(param);
                }
                state_dict_tp_all.append(result);
            }
            return state_dict_tp_all;
        })
        .def(
            "forward", [](InferEngine &self, const InferEngine::Input &input) -> InferEngine::Output { return self.forward(input); }, "Run inference on all ranks with arbitrary arguments")
        .def(
            "reset_cache", [](InferEngine &self, std::shared_ptr<const cache::CacheConfig> cfg) {
                self.reset_cache(cfg ? cfg.get() : nullptr);
            },
            py::arg("cache_config") = py::none())
        .def("get_cache_config", [](const InferEngine &self) {
            auto cfg = self.get_cache_config();
            return std::shared_ptr<cache::CacheConfig>(std::move(cfg->unique_copy()));
        })
        .def("__repr__", [](const InferEngine &self) {
            return "<InferEngine: " + std::string(self.get_dist_config()) + ">";
        });

    py::class_<InferEngine::Input>(infer_engine, "Input")
        .def(py::init([](const infinicore::Tensor &input_ids, const infinicore::Tensor &position_ids, const infinicore::Tensor &cache_positions) {
                 return new InferEngine::Input{input_ids, position_ids, cache_positions};
             }),
             py::arg("input_ids"), py::arg("position_ids"), py::arg("cache_positions"));

    py::class_<InferEngine::Output>(infer_engine, "Output")
        .def_readwrite("logits", &InferEngine::Output::logits, "Output tensor");
}

} // namespace infinilm::engine
