#include "../cache/cache_config.hpp"
#include "../engine/infer_engine.hpp"
#include "infinicore/tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinilm::cache {

inline void bind_cache_config(py::module &m) {
    // First bind the CacheType enum
    py::enum_<CacheType>(m, "CacheType")
        .value("DYNAMIC", CacheType::DYNAMIC)
        .value("PAGED", CacheType::PAGED)
        .export_values();

    // Then bind the CacheResetMode enum
    py::enum_<CacheResetMode>(m, "CacheResetMode")
        .value("PRESERVE", CacheResetMode::PRESERVE)
        .value("RECREATE", CacheResetMode::RECREATE)
        .export_values();

    // Finally bind the CacheConfig struct
    py::class_<CacheConfig>(m, "CacheConfig")
        .def(py::init<>(), "Default constructor")
        .def(py::init<CacheType, size_t, size_t>(),
             py::arg("type") = CacheType::DYNAMIC,
             py::arg("num_layers") = 32,
             py::arg("max_kv_cache_length") = 4096,
             "Constructor with parameters")
        .def_readwrite("type", &CacheConfig::type, "Cache type")
        .def_readwrite("num_layers", &CacheConfig::num_layers, "Number of layers")
        .def_readwrite("max_kv_cache_length", &CacheConfig::max_kv_cache_length,
                       "Maximum KV cache length")
        .def_readwrite("initial_capacity", &CacheConfig::initial_capacity,
                       "Initial cache capacity in tokens")
        .def_readwrite("initial_batch_size", &CacheConfig::initial_batch_size,
                       "Initial batch size for cache allocation")
        .def_readwrite("growth_factor", &CacheConfig::growth_factor,
                       "Cache growth factor when resizing (e.g., 2.0 for doubling)")
        .def_readwrite("allow_expand", &CacheConfig::allow_expand,
                       "Whether to allow cache expansion")
        .def_readwrite("reset_mode", &CacheConfig::reset_mode,
                       "Cache reset mode")
        .def("__eq__", &CacheConfig::operator==, py::is_operator(),
             "Check if two CacheConfig objects are equal")
        .def("__ne__", &CacheConfig::operator!=, py::is_operator(),
             "Check if two CacheConfig objects are not equal")
        .def("__repr__", [](const CacheConfig &cfg) {
            return fmt::format("CacheConfig(type={}, num_layers={}, max_kv_cache_length={}, "
                               "initial_capacity={}, initial_batch_size={}, growth_factor={}, "
                               "allow_expand={}, reset_mode={})",
                               static_cast<int>(cfg.type), cfg.num_layers,
                               cfg.max_kv_cache_length, cfg.initial_capacity,
                               cfg.initial_batch_size, cfg.growth_factor,
                               cfg.allow_expand, static_cast<int>(cfg.reset_mode));
        });
}

} // namespace infinilm::cache

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
        .def(py::init([](const InfinilmModel::Config &cfg,
                         const infinilm::engine::distributed::DistConfig &dist,
                         infinicore::Device::Type dev,
                         const infinilm::cache::CacheConfig &cache_config) {
                 return new InferEngine(cfg, dist, dev, cache_config);
             }),
             py::arg("config"),
             py::arg("distributed_config") = distributed::DistConfig(),
             py::arg("device_type") = infinicore::context::getDevice().getType(),
             py::arg("cache_config") = infinilm::cache::CacheConfig())
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
            "generate", [](InferEngine &self, py::object input_ids, py::object position_ids) -> infinicore::Tensor {
                return self.generate(input_ids.cast<infinicore::Tensor>(), position_ids.cast<infinicore::Tensor>());
            },
            "Run inference on all ranks with arbitrary arguments")
        .def("reset_cache", py::overload_cast<size_t>(&InferEngine::reset_cache), py::arg("pos") = 0, "Reset the internal cache in all workers to a specific position")
        .def("reset_cache", py::overload_cast<const cache::CacheConfig &, size_t>(&InferEngine::reset_cache), py::arg("cache_config"), py::arg("pos") = 0, "Reset cache with new KV configuration")
        .def("get_cache_config", &InferEngine::get_cache_config, "Get current KV configuration")
        .def("__repr__", [](const InferEngine &self) {
            return "<InferEngine: " + std::string(self.get_dist_config()) + ">";
        });
}

} // namespace infinilm::engine
