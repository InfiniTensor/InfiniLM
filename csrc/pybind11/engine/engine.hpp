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

inline infinicore::Tensor tensor_from_python(py::object obj) {
    auto data_ptr = obj.attr("data_ptr")().cast<std::uintptr_t>();
    auto shape = obj.attr("shape").cast<infinicore::Shape>();
    auto strides = obj.attr("stride")().cast<infinicore::Strides>();
    auto dtype = static_cast<infinicore::DataType>(py::int_(obj.attr("dtype").attr("_underlying")).cast<int>());
    auto py_device = obj.attr("device").attr("_underlying");
    auto device_type = static_cast<infinicore::Device::Type>(py::int_(py_device.attr("type")).cast<int>());
    auto device_index = py_device.attr("index").cast<int>();
    return infinicore::Tensor::strided_from_blob(
        reinterpret_cast<void *>(data_ptr),
        shape,
        strides,
        dtype,
        infinicore::Device(device_type, device_index));
}

inline std::optional<infinicore::Tensor> optional_tensor_from_python(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return tensor_from_python(obj);
}

inline py::object dtype_to_python(infinicore::DataType dtype) {
    auto inf = py::module_::import("infinicore");
    switch (dtype) {
    case infinicore::DataType::I64: return inf.attr("int64");
    case infinicore::DataType::I32: return inf.attr("int32");
    case infinicore::DataType::BF16: return inf.attr("bfloat16");
    case infinicore::DataType::F16: return inf.attr("float16");
    case infinicore::DataType::F32: return inf.attr("float32");
    default: throw std::runtime_error("Unsupported output dtype for Python wrapper");
    }
}

inline py::object device_to_python(const infinicore::Device &device) {
    auto inf = py::module_::import("infinicore");
    const char *type = device.getType() == infinicore::Device::Type::CPU ? "cpu" : "cuda";
    return inf.attr("device")(type, device.getIndex());
}

inline py::object tensor_to_python(const infinicore::Tensor &tensor) {
    auto inf = py::module_::import("infinicore");
    return inf.attr("strided_from_blob")(
        reinterpret_cast<std::uintptr_t>(tensor->data()),
        tensor->shape(),
        tensor->strides(),
        py::arg("dtype") = dtype_to_python(tensor->dtype()),
        py::arg("device") = device_to_python(tensor->device()));
}

inline void bind_infer_engine(py::module &m) {
    py::class_<InferEngine, std::shared_ptr<InferEngine>> infer_engine(m, "InferEngine");
    infer_engine
        .def(py::init([](
                          const std::string &model_path,
                          const distributed::DistConfig &dist,
                          int dev,
                          std::shared_ptr<const infinilm::cache::CacheConfig> cache_cfg,
                          bool enable_graph_compiling,
                          const std::string &attention_backend,
                          std::optional<infinicore::DataType> kv_cache_dtype) {
                 return std::make_shared<InferEngine>(
                     model_path,
                     dist,
                     static_cast<infinicore::Device::Type>(dev),
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
        .def("load_param", [](InferEngine &self, const std::string &name, py::object param) {
            self.load_param(name, tensor_from_python(param));
        },
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
        .def("state_dict_keyname", [](InferEngine &self) {
            py::list keys;
            auto all_state = self.state_dict();
            if (!all_state.empty()) {
                for (const auto &item : all_state[0]) {
                    keys.append(item.first);
                }
            }
            return keys;
        })
        .def("process_weights_after_loading", &InferEngine::process_weights_after_loading, "Process the weights after loading on all workers (e.g., for quantization)")
        .def("forward_py", [](InferEngine &self,
                              py::object input_ids,
                              py::object pixel_values,
                              py::object position_ids,
                              py::object past_sequence_lengths,
                              py::object total_sequence_lengths,
                              py::object input_offsets,
                              py::object cu_seqlens,
                              py::object block_tables,
                              py::object slot_mapping,
                              py::object image_bound,
                              py::object tgt_sizes,
                              py::kwargs kwargs) {
            InferEngine::Input input{
                optional_tensor_from_python(input_ids),
                optional_tensor_from_python(pixel_values),
                optional_tensor_from_python(position_ids),
                optional_tensor_from_python(past_sequence_lengths),
                optional_tensor_from_python(total_sequence_lengths),
                optional_tensor_from_python(input_offsets),
                optional_tensor_from_python(cu_seqlens),
                optional_tensor_from_python(block_tables),
                optional_tensor_from_python(slot_mapping),
                optional_tensor_from_python(image_bound),
                optional_tensor_from_python(tgt_sizes),
            };
            input.temperature = kwargs.contains("temperature") && !kwargs["temperature"].is_none() ? kwargs["temperature"].cast<float>() : 1.0f;
            input.top_p = kwargs.contains("top_p") && !kwargs["top_p"].is_none() ? kwargs["top_p"].cast<float>() : 1.0f;
            input.top_k = kwargs.contains("top_k") && !kwargs["top_k"].is_none() ? kwargs["top_k"].cast<int>() : 1;
            return tensor_to_python(self.forward(input).output_ids);
        },
             py::arg("input_ids") = py::none(),
             py::arg("pixel_values") = py::none(),
             py::arg("position_ids") = py::none(),
             py::arg("past_sequence_lengths") = py::none(),
             py::arg("total_sequence_lengths") = py::none(),
             py::arg("input_offsets") = py::none(),
             py::arg("cu_seqlens") = py::none(),
             py::arg("block_tables") = py::none(),
             py::arg("slot_mapping") = py::none(),
             py::arg("image_bound") = py::none(),
             py::arg("tgt_sizes") = py::none())
        .def("forward", [](InferEngine &self, const InferEngine::Input &input) -> InferEngine::Output { return self.forward(input); }, "Run inference on all ranks with arbitrary arguments")
        .def("reset_cache", [](InferEngine &self, std::shared_ptr<cache::CacheConfig> cfg) { self.reset_cache(cfg ? cfg.get() : nullptr); }, py::arg("cache_config") = py::none())
        .def("get_cache_config", [](const InferEngine &self) -> std::shared_ptr<cache::CacheConfig> {
            auto cfg = self.get_cache_config();
            return cfg ? std::shared_ptr<cache::CacheConfig>(cfg->unique_copy()) : nullptr; })
        .def("__repr__", [](const InferEngine &self) { return "<InferEngine: " + std::string(self.get_dist_config()) + ">"; });

    py::class_<InferEngine::Input>(infer_engine, "Input")
        .def(
            py::init([](
                         std::optional<infinicore::Tensor> input_ids,
                         std::optional<infinicore::Tensor> pixel_values,
                         std::optional<infinicore::Tensor> position_ids,
                         std::optional<infinicore::Tensor> past_sequence_lengths,
                         std::optional<infinicore::Tensor> total_sequence_lengths,
                         std::optional<infinicore::Tensor> input_offsets,
                         std::optional<infinicore::Tensor> cu_seqlens,
                         std::optional<infinicore::Tensor> block_tables,
                         std::optional<infinicore::Tensor> slot_mapping,
                         std::optional<infinicore::Tensor> image_bound,
                         std::optional<infinicore::Tensor> tgt_sizes,
                         py::kwargs kwargs) {
                InferEngine::Input input{
                    std::move(input_ids),
                    std::move(pixel_values),
                    std::move(position_ids),
                    std::move(past_sequence_lengths),
                    std::move(total_sequence_lengths),
                    std::move(input_offsets),
                    std::move(cu_seqlens),
                    std::move(block_tables),
                    std::move(slot_mapping),
                    std::move(image_bound),
                    std::move(tgt_sizes),
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
        .def_readwrite("top_p", &InferEngine::Input::top_p);

    py::class_<InferEngine::Output>(infer_engine, "Output")
        .def_readwrite("output_ids", &InferEngine::Output::output_ids, "Output tensor");
}

} // namespace infinilm::engine
