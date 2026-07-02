#include "../../debug_utils/hooks.hpp"
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
        .def_readwrite("moe_ep_backend", &DistConfig::moe_ep_backend,
                       "MoE expert-parallel backend")
        .def_readwrite("moe_ep_size", &DistConfig::moe_ep_size,
                       "MoE expert-parallel size")
        .def("__repr__", [](const DistConfig &cfg) {
            return std::string(cfg);
        })
        .def("__str__", [](const DistConfig &cfg) {
            return std::string(cfg);
        });
}

} // namespace infinilm::engine::distributed

namespace infinilm::engine {

inline void bind_hook_registry(py::module &m) {
    using infinilm::models::debug_utils::HookRegistry;

    // TODO: HookRegistry should be moved out from Llama-specific bindings to InfiniCore as common utils in future work
    // Bind HookRegistry
    py::class_<HookRegistry, std::shared_ptr<HookRegistry>>(m, "HookRegistry")
        .def(py::init<>())
        .def(
            "register_hook", [](HookRegistry &self, const std::string &name, py::object callback) {
                // Convert Python callable to C++ function
                self.register_hook(name, [callback](const std::string &hook_name, const infinicore::Tensor &tensor, int layer_idx) {
                    try {
                        // Call Python callback with hook name, tensor, and layer index
                        callback(hook_name, tensor, layer_idx);
                    } catch (const py::error_already_set &e) {
                        // Re-raise Python exception
                        throw;
                    }
                });
            },
            py::arg("name"), py::arg("callback"))
        .def("clear", &HookRegistry::clear)
        .def("has_hooks", &HookRegistry::has_hooks);
}

inline void bind_infer_engine(py::module &m) {
    py::class_<InferEngine, std::shared_ptr<InferEngine>> infer_engine(m, "InferEngine");

    infer_engine
        .def(py::init([](
                          const std::string &config_str,
                          const distributed::DistConfig &dist,
                          infinicore::Device::Type dev,
                          std::shared_ptr<const infinilm::cache::CacheConfig> cache_cfg,
                          bool enable_graph_compiling,
                          const std::string &attention_backend,
                          std::optional<infinicore::DataType> kv_cache_dtype,
                          bool use_mla,
                          const std::string &weight_load_mode) {
                 return std::make_shared<InferEngine>(
                     config_str,
                     dist,
                     dev,
                     cache_cfg ? cache_cfg.get() : nullptr,
                     enable_graph_compiling,
                     infinilm::backends::parse_attention_backend(attention_backend),
                     kv_cache_dtype,
                     use_mla,
                     weight_load_mode);
             }),
             py::arg("config_str") = "",
             py::arg("distributed_config") = distributed::DistConfig(),
             py::arg("device_type") = infinicore::context::getDevice().getType(),
             py::arg("cache_config") = py::none(),
             py::arg("enable_graph_compiling") = false,
             py::arg("attention_backend") = "default",
             py::arg("kv_cache_dtype") = py::none(),
             py::arg("use_mla") = false,
             py::arg("weight_load_mode") = "async")
        .def("load_param", &InferEngine::load_param,
             py::arg("name"), py::arg("param"),
             "Load a parameter tensor into all workers (each worker picks its shard)")
        .def("load_params", &InferEngine::load_params,
             py::arg("params"), py::arg("strict") = true,
             "Load a batch of parameter tensors into all workers, syncing once per worker")
        .def("state_dict_keyname", &InferEngine::state_dict_keys)
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
            "forward", [](InferEngine &self, const InferEngine::Input &input) -> InferEngine::Output {
                // IMPORTANT: Release the GIL before calling forward() to allow other Python threads
                // to run concurrently during inference (which may block for a long time).
                // Do NOT remove this — without it, the GIL is held throughout inference and will
                // deadlock or stall any other Python thread (e.g., request handling, scheduling).
                py::gil_scoped_release release;
                return self.forward(input);
            },
            "Run inference on all ranks with arbitrary arguments")
        .def(
            "reset_cache", [](InferEngine &self, std::shared_ptr<cache::CacheConfig> cfg) { self.reset_cache(cfg ? cfg.get() : nullptr); }, py::arg("cache_config") = py::none())
        .def("get_kv_cache", &InferEngine::get_kv_cache, "Get per-rank kv cache list")
        .def("get_cache_config", [](const InferEngine &self) -> std::shared_ptr<cache::CacheConfig> {
            auto cfg = self.get_cache_config();
            return cfg ? std::shared_ptr<cache::CacheConfig>(cfg->unique_copy()) : nullptr; })
        .def("__repr__", [](const InferEngine &self) { return "<InferEngine: " + std::string(self.get_dist_config()) + ">"; });

    py::class_<InferEngine::Input>(infer_engine, "Input")
        .def(
            py::init([](
                         std::optional<infinicore::Tensor> input_ids,
                         std::optional<infinicore::Tensor> position_ids,
                         std::optional<infinicore::Tensor> past_sequence_lengths,
                         std::optional<infinicore::Tensor> total_sequence_lengths,
                         std::optional<infinicore::Tensor> input_offsets,
                         std::optional<infinicore::Tensor> cu_seqlens,
                         std::optional<infinicore::Tensor> block_tables,
                         std::optional<infinicore::Tensor> slot_mapping,
                         std::optional<std::vector<infinicore::Tensor>> pixel_values,
                         std::optional<std::vector<infinicore::Tensor>> image_bound,
                         std::optional<std::vector<infinicore::Tensor>> tgt_sizes,
                         std::optional<std::vector<size_t>> image_req_ids,
                         std::optional<std::vector<size_t>> visual_token_ranges,
                         py::kwargs kwargs) {
                InferEngine::Input input{
                    std::move(input_ids),
                    std::move(position_ids),
                    std::move(past_sequence_lengths),
                    std::move(total_sequence_lengths),
                    std::move(input_offsets),
                    std::move(cu_seqlens),
                    std::move(block_tables),
                    std::move(slot_mapping),
                    std::move(pixel_values),
                    std::move(image_bound),
                    std::move(tgt_sizes),
                    std::move(image_req_ids),
                    std::move(visual_token_ranges),
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
            py::arg("position_ids") = std::nullopt,
            py::arg("past_sequence_lengths") = std::nullopt,
            py::arg("total_sequence_lengths") = std::nullopt,
            py::arg("input_offsets") = std::nullopt,
            py::arg("cu_seqlens") = std::nullopt,
            py::arg("block_tables") = std::nullopt,
            py::arg("slot_mapping") = std::nullopt,
            py::arg("pixel_values") = std::nullopt,
            py::arg("image_bound") = std::nullopt,
            py::arg("tgt_sizes") = std::nullopt,
            py::arg("image_req_ids") = std::nullopt,
            py::arg("visual_token_ranges") = std::nullopt)
        .def_readwrite("input_ids", &InferEngine::Input::input_ids)
        .def_readwrite("position_ids", &InferEngine::Input::position_ids)
        .def_readwrite("past_sequence_lengths", &InferEngine::Input::past_sequence_lengths)
        .def_readwrite("total_sequence_lengths", &InferEngine::Input::total_sequence_lengths)
        .def_readwrite("input_offsets", &InferEngine::Input::input_offsets)
        .def_readwrite("cu_seqlens", &InferEngine::Input::cu_seqlens)
        .def_readwrite("block_tables", &InferEngine::Input::block_tables)
        .def_readwrite("slot_mapping", &InferEngine::Input::slot_mapping)
        .def_readwrite("pixel_values", &InferEngine::Input::pixel_values)
        .def_readwrite("image_bound", &InferEngine::Input::image_bound)
        .def_readwrite("tgt_sizes", &InferEngine::Input::tgt_sizes)
        .def_readwrite("image_req_ids", &InferEngine::Input::image_req_ids)
        .def_readwrite("visual_token_ranges", &InferEngine::Input::visual_token_ranges)
        .def_readwrite("temperature", &InferEngine::Input::temperature)
        .def_readwrite("top_k", &InferEngine::Input::top_k)
        .def_readwrite("top_p", &InferEngine::Input::top_p);

    py::class_<InferEngine::Output>(infer_engine, "Output")
        .def_readwrite("output_ids", &InferEngine::Output::output_ids, "Output tensor");
}

} // namespace infinilm::engine
