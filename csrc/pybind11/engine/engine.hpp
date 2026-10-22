#include "../../debug_utils/hooks.hpp"
#include "../../engine/infer_engine.hpp"
#include "runtime/tensor.hpp"
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
        .def_readwrite("pp_size", &DistConfig::pp_size,
                       "Pipeline parallel size")
        .def_readwrite("pp_stage", &DistConfig::pp_stage,
                       "Pipeline parallel stage index for this engine")
        .def_readwrite("master_addr", &DistConfig::master_addr,
                       "Address used to bootstrap distributed communication")
        .def_readwrite("master_port", &DistConfig::master_port,
                       "TCP port used to bootstrap distributed communication")
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
                self.register_hook(name, [callback](const std::string &hook_name, const infinilm::runtime::Tensor &tensor, int layer_idx) {
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
    py::class_<InferEngine, std::shared_ptr<InferEngine>> infer_engine(
        m, "InferEngine", py::release_gil_before_calling_cpp_dtor());

    infer_engine
        .def(py::init([](
                          const std::string &config_str,
                          const distributed::DistConfig &dist,
                          infinilm::runtime::Device::Type dev,
                          std::shared_ptr<const infinilm::cache::CacheConfig> cache_cfg,
                          bool enable_graph_compiling,
                          const std::string &attention_backend,
                          std::optional<infinilm::runtime::DataType> kv_cache_dtype,
                          bool use_mla,
                          const std::string &weight_load_mode,
                          bool pre_transpose,
                          std::optional<std::vector<std::pair<size_t, size_t>>> graph_capture_profiles,
                          bool graph_metadata_upload) {
                 const auto frozen_dist = dist;
                 py::gil_scoped_release release;
                 return std::make_shared<InferEngine>(
                     config_str,
                     frozen_dist,
                     dev,
                     cache_cfg ? cache_cfg.get() : nullptr,
                     enable_graph_compiling,
                     infinilm::backends::parse_attention_backend(attention_backend),
                     kv_cache_dtype,
                     use_mla,
                     weight_load_mode,
                     pre_transpose, graph_capture_profiles, graph_metadata_upload);
             }),
             py::arg("config_str"),
             py::arg("distributed_config"),
             py::arg("device_type"),
             py::arg("cache_config") = py::none(),
             py::arg("enable_graph_compiling") = false,
             py::arg("attention_backend") = "default",
             py::arg("kv_cache_dtype") = py::none(),
             py::arg("use_mla") = false,
             py::arg("weight_load_mode") = "async",
             py::arg("pre_transpose") = false,
             py::arg("graph_capture_profiles") = py::none(),
             py::arg("graph_metadata_upload") = false)
        .def("load_param", &InferEngine::load_param,
             py::arg("name"), py::arg("param"),
             py::call_guard<py::gil_scoped_release>(),
             "Load a parameter tensor into all workers (each worker picks its shard)")
        .def("load_params", &InferEngine::load_params,
             py::arg("params"), py::arg("strict") = true,
             py::call_guard<py::gil_scoped_release>(),
             "Load a batch of parameter tensors into all workers, syncing once per worker")
        .def("state_dict_keyname", &InferEngine::state_dict_keys)
        .def("state_dict", [](InferEngine &self) {
            const auto tensors = [&self] {
                py::gil_scoped_release release;
                return self.state_dict();
            }();
            py::list state_dict_tp_all;
            for (const auto &state_dict_tp : tensors) {
                py::dict result;
                for (const auto &[name, param] : state_dict_tp) {
                    result[py::cast(name)] = infinilm::runtime::Tensor(param);
                }
                state_dict_tp_all.append(result);
            }
            return state_dict_tp_all;
        })
        .def("process_weights_after_loading", &InferEngine::process_weights_after_loading, py::call_guard<py::gil_scoped_release>(), "Process the weights after loading on all workers (e.g., for quantization)")
        .def("forward", [](InferEngine &self, const InferEngine::Input &input) -> InferEngine::Output {
                // IMPORTANT: Release the GIL before calling forward() to allow other Python threads
                // to run concurrently during inference (which may block for a long time).
                // Do NOT remove this — without it, the GIL is held throughout inference and will
                // deadlock or stall any other Python thread (e.g., request handling, scheduling).
                const auto frozen_input = input; // copy native values/properties while GIL still held
                py::gil_scoped_release release;
                return self.forward(frozen_input); }, "Run inference on all ranks with arbitrary arguments")
        .def("reset_cache", [](InferEngine &self, std::shared_ptr<cache::CacheConfig> cfg) { self.reset_cache(cfg ? cfg.get() : nullptr); }, py::arg("cache_config") = py::none(), py::call_guard<py::gil_scoped_release>())
        .def("get_kv_cache", &InferEngine::get_kv_cache, py::call_guard<py::gil_scoped_release>(), "Get per-rank kv cache list")
        .def("get_nsa_pool_epoch", [](InferEngine &self) {
            py::gil_scoped_release release;
            return self.get_nsa_pool_epoch(); }, "Readonly native VideoNSA raw-pool lifetime stamp")
        .def("get_cache_config", [](const InferEngine &self) -> std::shared_ptr<cache::CacheConfig> {
            py::gil_scoped_release release;
            return self.get_cache_config(); })
        .def("__repr__", [](const InferEngine &self) { return "<InferEngine: " + std::string(self.get_dist_config()) + ">"; });

    using NsaRowValue = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                   std::vector<nsa::Page>, std::vector<std::int64_t>>;
    py::class_<nsa::Snapshot>(infer_engine, "NsaOwnerSnapshot")
        .def(py::init([](std::uint64_t epoch, std::uint64_t page_size, std::uint64_t page_count,
                         const std::vector<NsaRowValue> &rows,
                         const std::vector<std::int32_t> &past, const std::vector<std::int32_t> &total,
                         const std::vector<std::int32_t> &offsets, const std::vector<std::int32_t> &cu,
                         const std::vector<std::vector<std::int32_t>> &tables,
                         const std::vector<std::int64_t> &slots) {
            nsa::Snapshot s{epoch, page_size, page_count, {}, {past, total, offsets, cu, {}, 0, slots}};
            for (const auto &r : rows) {
                s.rows.push_back({std::get<0>(r), std::get<1>(r), std::get<2>(r),
                                  std::get<3>(r), std::get<4>(r), std::get<5>(r)});
            }
            if (tables.empty() || tables.size() != rows.size()) {
                throw py::value_error("NSA table rows missing");
            }
            s.upload.table_width = tables.front().size();
            for (const auto &t : tables) {
                if (t.size() != s.upload.table_width) {
                    throw py::value_error("NSA table is not rectangular");
                }
                s.upload.tables.insert(s.upload.tables.end(), t.begin(), t.end());
            }
            nsa::validate(s, s.upload);
            return s; // no py::object, Tensor or borrowed CPU pointer survives
        }));

    py::class_<InferEngine::Input>(infer_engine, "Input")
        .def(
            py::init([](
                         std::optional<infinilm::runtime::Tensor> input_ids,
                         std::optional<infinilm::runtime::Tensor> position_ids,
                         std::optional<infinilm::runtime::Tensor> past_sequence_lengths,
                         std::optional<infinilm::runtime::Tensor> total_sequence_lengths,
                         std::optional<infinilm::runtime::Tensor> input_offsets,
                         std::optional<infinilm::runtime::Tensor> cu_seqlens,
                         std::optional<infinilm::runtime::Tensor> block_tables,
                         std::optional<infinilm::runtime::Tensor> slot_mapping,
                         std::optional<infinilm::runtime::Tensor> mamba_init_state_indices,
                         std::optional<infinilm::runtime::Tensor> mamba_final_state_indices,
                         std::optional<std::vector<infinilm::runtime::Tensor>> pixel_values,
                         std::optional<std::vector<infinilm::runtime::Tensor>> image_bound,
                         std::optional<std::vector<infinilm::runtime::Tensor>> tgt_sizes,
                         std::optional<std::vector<infinilm::runtime::Tensor>> image_grid_thw,
                         std::optional<std::vector<size_t>> image_req_ids,
                         std::optional<std::vector<size_t>> visual_token_ranges,
                         std::optional<infinilm::runtime::Tensor> target_hidden_states,
                         bool sample_all_positions,
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
                    std::move(mamba_init_state_indices),
                    std::move(mamba_final_state_indices),
                    std::move(pixel_values),
                    std::move(image_bound),
                    std::move(tgt_sizes),
                    std::move(image_grid_thw),
                    std::move(image_req_ids),
                    std::move(visual_token_ranges),
                    std::move(target_hidden_states),
                    sample_all_positions,
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
                    "random_values",
                    "sampling_rows",
                    "nsa_owner_snapshot",
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
                    } else if (key == "random_values") {
                        if (!item.second.is_none()) {
                            input.random_values = py::cast<std::vector<float>>(item.second);
                        }

                    } else if (key == "nsa_owner_snapshot") {
                        if (!item.second.is_none()) {
                            input.nsa_owner_snapshot = py::cast<nsa::Snapshot>(item.second);
                        }
                    } else if (key == "sampling_rows") {
                        if (!item.second.is_none()) {
                            input.sampling_rows = py::cast<std::vector<std::tuple<float, float, int>>>(item.second);
                        }
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
            py::arg("mamba_init_state_indices") = std::nullopt,
            py::arg("mamba_final_state_indices") = std::nullopt,
            py::arg("pixel_values") = std::nullopt,
            py::arg("image_bound") = std::nullopt,
            py::arg("tgt_sizes") = std::nullopt,
            py::arg("image_grid_thw") = std::nullopt,
            py::arg("image_req_ids") = std::nullopt,
            py::arg("visual_token_ranges") = std::nullopt,
            py::arg("target_hidden_states") = std::nullopt,
            py::arg("sample_all_positions") = false)
        .def_readwrite("input_ids", &InferEngine::Input::input_ids)
        .def_readwrite("position_ids", &InferEngine::Input::position_ids)
        .def_readwrite("past_sequence_lengths", &InferEngine::Input::past_sequence_lengths)
        .def_readwrite("total_sequence_lengths", &InferEngine::Input::total_sequence_lengths)
        .def_readwrite("input_offsets", &InferEngine::Input::input_offsets)
        .def_readwrite("cu_seqlens", &InferEngine::Input::cu_seqlens)
        .def_readwrite("block_tables", &InferEngine::Input::block_tables)
        .def_readwrite("slot_mapping", &InferEngine::Input::slot_mapping)
        .def_readwrite("mamba_init_state_indices", &InferEngine::Input::mamba_init_state_indices)
        .def_readwrite("mamba_final_state_indices", &InferEngine::Input::mamba_final_state_indices)
        .def_readwrite("pixel_values", &InferEngine::Input::pixel_values)
        .def_readwrite("image_bound", &InferEngine::Input::image_bound)
        .def_readwrite("tgt_sizes", &InferEngine::Input::tgt_sizes)
        .def_readwrite("image_grid_thw", &InferEngine::Input::image_grid_thw)
        .def_readwrite("image_req_ids", &InferEngine::Input::image_req_ids)
        .def_readwrite("visual_token_ranges", &InferEngine::Input::visual_token_ranges)
        .def_readwrite("target_hidden_states", &InferEngine::Input::target_hidden_states)
        .def_readwrite("sample_all_positions", &InferEngine::Input::sample_all_positions)
        .def_readwrite("temperature", &InferEngine::Input::temperature)
        .def_readwrite("top_k", &InferEngine::Input::top_k)
        .def_readwrite("top_p", &InferEngine::Input::top_p)
        .def_readwrite("random_values", &InferEngine::Input::random_values)
        .def_readwrite("sampling_rows", &InferEngine::Input::sampling_rows);

    py::class_<InferEngine::Output>(infer_engine, "Output")
        .def_readwrite("output_ids", &InferEngine::Output::output_ids, "Sampled token IDs")
        .def_readwrite("logits", &InferEngine::Output::logits, "Raw logits tensor")
        .def_readwrite("hidden_states", &InferEngine::Output::hidden_states, "Raw hidden states tensor")
        .def_property_readonly("execution_report", [](const InferEngine::Output &output) {
            py::dict result;
            const auto *graph = output.graph_execution ? &*output.graph_execution : nullptr;
            const size_t device_segments = graph ? graph->device_graph_segments : 0;
            const size_t host_segments = graph ? graph->host_segments : 0;
            result["mode"] = graph == nullptr     ? "eager"
                           : device_segments == 0 ? "host_replay"
                           : host_segments == 0   ? "device_graph"
                                                  : "mixed_graph";
            result["device_graph_segments"] = device_segments;
            result["host_segments"] = host_segments;
            result["operator_count"] = graph ? graph->operator_count : 0;
            result["fallback_reason"] = graph ? graph->fallback_reason : output.eager_reason;
            return result;
        });
}

} // namespace infinilm::engine
