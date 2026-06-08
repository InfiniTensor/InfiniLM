#include "../../engine/infer_engine.hpp"
#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"
#include <pybind11/gil.h>
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

namespace {

infinicore::Tensor tensor_from_py(py::handle param) {
    py::handle t = param;
    if (py::hasattr(param, "_underlying")) {
        t = param.attr("_underlying");
    }
    auto shape = t.attr("shape").cast<infinicore::Shape>();
    void *ptr = reinterpret_cast<void *>(t.attr("data_ptr")().cast<uintptr_t>());
    auto dtype = static_cast<infinicore::DataType>(py::cast<int>(py::int_(t.attr("dtype"))));
    py::object dev_obj = t.attr("device");
    infinicore::Device::Type dev_type = infinicore::Device::Type::CPU;
    infinicore::Device::Index dev_index = 0;
    if (py::isinstance<py::str>(dev_obj)) {
        const std::string s = py::str(dev_obj);
        const auto colon = s.find(':');
        const std::string kind = (colon == std::string::npos) ? s : s.substr(0, colon);
        if (colon != std::string::npos) {
            dev_index = static_cast<infinicore::Device::Index>(std::stoul(s.substr(colon + 1)));
        }
        if (kind == "CPU") {
            dev_type = infinicore::Device::Type::CPU;
        } else if (kind == "METAX" || kind == "cuda") {
            dev_type = infinicore::Device::Type::METAX;
        } else if (kind == "NVIDIA") {
            dev_type = infinicore::Device::Type::NVIDIA;
        } else {
            dev_type = infinicore::Device::Type::METAX;
        }
    } else {
        dev_type = static_cast<infinicore::Device::Type>(
            py::cast<int>(py::int_(dev_obj.attr("type"))));
        dev_index = dev_obj.attr("index").cast<infinicore::Device::Index>();
    }
    infinicore::Device device(dev_type, dev_index);
    if (t.attr("is_contiguous")().cast<bool>()) {
        return infinicore::Tensor::from_blob(ptr, shape, dtype, device);
    }
    auto strides = t.attr("strides").cast<infinicore::Strides>();
    return infinicore::Tensor::strided_from_blob(ptr, shape, strides, dtype, device);
}

std::optional<infinicore::Tensor> optional_tensor_from_py(py::handle param) {
    if (param.is_none()) {
        return std::nullopt;
    }
    return tensor_from_py(param);
}

py::dict tensor_to_blob_meta(const infinicore::Tensor &t) {
    py::dict meta;
    meta["data_ptr"] = reinterpret_cast<uintptr_t>(t->data());
    meta["shape"] = t->shape();
    meta["dtype"] = py::int_(static_cast<int>(t->dtype()));
    meta["device"] = py::str(static_cast<std::string>(t->device().toString()));
    if (t->is_contiguous()) {
        meta["strides"] = py::none();
    } else {
        meta["strides"] = t->strides();
    }
    return meta;
}

} // namespace

inline void bind_infer_engine(py::module &m) {
    py::class_<InferEngine, std::shared_ptr<InferEngine>> infer_engine(m, "InferEngine");
    infer_engine
        .def(py::init([](
                          const std::string &model_path,
                          const distributed::DistConfig &dist,
                          int dev_type,
                          std::shared_ptr<const infinilm::cache::CacheConfig> cache_cfg,
                          bool enable_graph_compiling,
                          const std::string &attention_backend,
                          std::optional<infinicore::DataType> kv_cache_dtype) {
                 auto dev = static_cast<infinicore::Device::Type>(dev_type);
                 return std::make_shared<InferEngine>(
                     model_path,
                     dist,
                     dev,
                     cache_cfg ? cache_cfg.get() : nullptr,
                     enable_graph_compiling,
                     infinilm::backends::parse_attention_backend(attention_backend),
                     kv_cache_dtype);
             }),
             py::arg("model_path") = "",
             py::arg("distributed_config") = distributed::DistConfig(),
             py::arg("device_type") = static_cast<int>(infinicore::context::getDevice().getType()),
             py::arg("cache_config") = py::none(),
             py::arg("enable_graph_compiling") = false,
             py::arg("attention_backend") = "default",
             py::arg("kv_cache_dtype") = py::none())
        .def(
            "load_param",
            [](InferEngine &self, const std::string &name, py::handle param) {
                self.load_param(name, tensor_from_py(param));
            },
            py::arg("name"),
            py::arg("param"),
            "Load a parameter tensor into all workers (each worker picks its shard)")
        .def("weight_blob", [](InferEngine &self, const std::string &name) -> py::dict {
            for (const auto &state_dict_tp : self.state_dict()) {
                for (const auto &[n, param] : state_dict_tp) {
                    if (n != name) {
                        continue;
                    }
                    const infinicore::Tensor &t = param;
                    py::dict meta;
                    meta["data_ptr"] = reinterpret_cast<uintptr_t>(t->data());
                    meta["shape"] = t->shape();
                    meta["dtype"] = py::int_(static_cast<int>(t->dtype()));
                    meta["device"] = py::str(static_cast<std::string>(t->device().toString()));
                    if (t->is_contiguous()) {
                        meta["strides"] = py::none();
                    } else {
                        meta["strides"] = t->strides();
                    }
                    return meta;
                }
            }
            throw std::runtime_error("weight not found: " + name);
        })
        .def("state_dict_keys", [](InferEngine &self) {
            py::list state_dict_keys_tp_all;
            for (const auto &state_dict_tp : self.state_dict()) {
                py::list keys;
                for (const auto &[name, param] : state_dict_tp) {
                    (void)param;
                    keys.append(py::cast(name));
                }
                state_dict_keys_tp_all.append(keys);
            }
            return state_dict_keys_tp_all;
        })
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
        .def("process_weights_after_loading", &InferEngine::process_weights_after_loading, "Process the weights after loading on all workers (e.g., for quantization)")
        .def(
            "forward",
            [](InferEngine &self, const InferEngine::Input &input) -> InferEngine::Output {
                return self.forward(input);
            },
            "Run inference on all ranks with arbitrary arguments",
            py::call_guard<py::gil_scoped_release>())
        .def(
            "reset_cache",
            [](InferEngine &self, std::shared_ptr<cache::CacheConfig> cfg) {
                self.reset_cache(cfg ? cfg.get() : nullptr);
            },
            py::arg("cache_config") = py::none(),
            py::call_guard<py::gil_scoped_release>())
        .def("get_cache_config", [](const InferEngine &self) -> std::shared_ptr<cache::CacheConfig> {
            auto cfg = self.get_cache_config();
            return cfg ? std::shared_ptr<cache::CacheConfig>(cfg->unique_copy()) : nullptr; })
        .def("paged_kv_blob_layers", [](InferEngine &self) -> py::list {
            py::list out;
            for (const auto &t : self.get_paged_kv_cache_tensors()) {
                out.append(tensor_to_blob_meta(t));
            }
            return out;
        })
        .def("prefill_graph_stats", [](const InferEngine &self) -> py::dict {
            const auto stats = self.prefill_graph_stats();
            py::dict out;
            out["prefill_graph_hits"] = stats.prefill_graph_hits;
            out["prefill_graph_misses"] = stats.prefill_graph_misses;
            out["decode_graph_hits"] = stats.decode_graph_hits;
            out["decode_graph_misses"] = stats.decode_graph_misses;
            out["piecewise_segment_replays"] = stats.piecewise_segment_replays;
            out["piecewise_prefill_hits"] = stats.piecewise_prefill_hits;
            out["piecewise_prefill_misses"] = stats.piecewise_prefill_misses;
            return out;
        })
        .def("native_capture_buckets", [](const InferEngine &self) -> py::list {
            py::list out;
            for (size_t bucket : self.native_capture_buckets()) {
                out.append(static_cast<int>(bucket));
            }
            return out;
        })
        .def("__repr__", [](const InferEngine &self) { return "<InferEngine: " + std::string(self.get_dist_config()) + ">"; });

    py::class_<InferEngine::Input>(infer_engine, "Input")
        .def(
            py::init([](
                         py::handle input_ids,
                         py::handle pixel_values,
                         py::handle position_ids,
                         py::handle past_sequence_lengths,
                         py::handle total_sequence_lengths,
                         py::handle input_offsets,
                         py::handle cu_seqlens,
                         py::handle block_tables,
                         py::handle slot_mapping,
                         py::handle image_bound,
                         py::handle tgt_sizes,
                         py::kwargs kwargs) {
                InferEngine::Input input{
                    optional_tensor_from_py(input_ids),
                    optional_tensor_from_py(pixel_values),
                    optional_tensor_from_py(position_ids),
                    optional_tensor_from_py(past_sequence_lengths),
                    optional_tensor_from_py(total_sequence_lengths),
                    optional_tensor_from_py(input_offsets),
                    optional_tensor_from_py(cu_seqlens),
                    optional_tensor_from_py(block_tables),
                    optional_tensor_from_py(slot_mapping),
                    optional_tensor_from_py(image_bound),
                    optional_tensor_from_py(tgt_sizes),
                };

                // Explicit defaults
                input.temperature = 1.0f;
                input.top_p = 1.0f;
                input.top_k = 1;

                // Allowed keyword arguments
                static const std::unordered_set<std::string> allowed_kwargs = {
                    "temperature",
                    "top_p",
                    "top_k",
                    "return_logits",
                    "is_final_prefill_chunk",
                };

                for (auto &item : kwargs) {
                    const std::string key = py::cast<std::string>(item.first);

                    if (allowed_kwargs.find(key) == allowed_kwargs.end()) {
                        throw py::value_error(
                            "InferEngine.Input got an unexpected keyword argument '" + key + "'");
                    }

                    if (key == "temperature") {
                        input.temperature = py::cast<float>(item.second);
                    } else if (key == "top_p") {
                        input.top_p = py::cast<float>(item.second);
                    } else if (key == "top_k") {
                        input.top_k = py::cast<int>(item.second);
                    } else if (key == "return_logits") {
                        input.return_logits = py::cast<bool>(item.second);
                    } else if (key == "is_final_prefill_chunk") {
                        if (py::isinstance<py::bool_>(item.second)) {
                            const bool flag = py::cast<bool>(item.second);
                            if (!flag) {
                                size_t n_req = 1;
                                if (input.block_tables.has_value()) {
                                    n_req = input.block_tables.value()->size(0);
                                } else if (input.input_offsets.has_value()) {
                                    n_req = input.input_offsets.value()->size(0) - 1;
                                }
                                input.is_final_prefill_chunk.assign(n_req, false);
                            }
                        } else {
                            input.is_final_prefill_chunk = py::cast<std::vector<bool>>(item.second);
                        }
                    }
                }

                return input;
            }),
            py::arg("input_ids") = std::nullopt,
            py::arg("pixel_values") = std::nullopt,
            py::arg("position_ids") = std::nullopt,
            py::arg("past_sequence_lengths") = std::nullopt,
            py::arg("total_sequence_lengths") = std::nullopt,
            py::arg("input_offsets") = std::nullopt,
            py::arg("cu_seqlens") = std::nullopt,
            py::arg("block_tables") = std::nullopt,
            py::arg("slot_mapping") = std::nullopt,
            py::arg("image_bound") = std::nullopt,
            py::arg("tgt_sizes") = std::nullopt)
        .def_readwrite("input_ids", &InferEngine::Input::input_ids)
        .def_readwrite("pixel_values", &InferEngine::Input::pixel_values)
        .def_readwrite("position_ids", &InferEngine::Input::position_ids)
        .def_readwrite("past_sequence_lengths", &InferEngine::Input::past_sequence_lengths)
        .def_readwrite("total_sequence_lengths", &InferEngine::Input::total_sequence_lengths)
        .def_readwrite("input_offsets", &InferEngine::Input::input_offsets)
        .def_readwrite("cu_seqlens", &InferEngine::Input::cu_seqlens)
        .def_readwrite("block_tables", &InferEngine::Input::block_tables)
        .def_readwrite("slot_mapping", &InferEngine::Input::slot_mapping)
        .def_readwrite("image_bound", &InferEngine::Input::image_bound)
        .def_readwrite("tgt_sizes", &InferEngine::Input::tgt_sizes)
        .def_readwrite("temperature", &InferEngine::Input::temperature)
        .def_readwrite("top_k", &InferEngine::Input::top_k)
        .def_readwrite("top_p", &InferEngine::Input::top_p)
        .def_readwrite("return_logits", &InferEngine::Input::return_logits)
        .def_readwrite("is_final_prefill_chunk", &InferEngine::Input::is_final_prefill_chunk);

    py::class_<InferEngine::Output>(infer_engine, "Output")
        .def_property_readonly("output_ids", [](const InferEngine::Output &self) {
            return tensor_to_blob_meta(self.output_ids);
        })
        .def_property_readonly("logits", [](const InferEngine::Output &self) -> py::object {
            if (!self.logits.has_value()) {
                return py::none();
            }
            return tensor_to_blob_meta(*self.logits);
        });
}

} // namespace infinilm::engine
