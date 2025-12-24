#include "../../cache/cache.hpp"
#include "infinicore/tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinilm::cache {

inline void bind_cache(py::module &m) {
    py::class_<infinilm::cache::CacheConfig,
               std::shared_ptr<infinilm::cache::CacheConfig>>(m, "CacheConfig")
        .def("__repr__", [](const infinilm::cache::CacheConfig &) {
            return "<CacheConfig (abstract)>";
        });

    py::class_<infinilm::cache::StaticKVCacheConfig,
               infinilm::cache::CacheConfig,
               std::shared_ptr<infinilm::cache::StaticKVCacheConfig>>(m, "StaticKVCacheConfig")
        .def(
            py::init<infinicore::Size, infinicore::Size>(),
            py::arg("max_batch_size") = 1,
            py::arg("max_cache_len") = std::numeric_limits<infinicore::Size>::max())
        .def(
            "max_batch_size",
            &infinilm::cache::StaticKVCacheConfig::max_batch_size)
        .def(
            "max_cache_len",
            &infinilm::cache::StaticKVCacheConfig::max_cache_len)
        .def("__repr__", [](const infinilm::cache::StaticKVCacheConfig &) {
            return "<StaticKVCacheConfig>";
        });
}

} // namespace infinilm::cache