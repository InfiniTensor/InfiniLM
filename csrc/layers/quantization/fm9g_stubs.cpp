#include "../moe/ep/ep_config.hpp"
#include "awq.hpp"
#include "compressed_tensors.hpp"
#include "gptq.hpp"
#include "kv_quant.hpp"
#include "w4_execution.hpp"
#include "w4_marlin_adapter.hpp"
#include <stdexcept>

namespace infinilm::quantization {
namespace {
[[noreturn]] void unsupported(const char *name) {
    throw std::runtime_error(std::string(name) + " is unavailable in the FM9G-only build");
}
} // namespace

std::vector<ParamDescriptor> CompressedTensors::get_param_layout(
    size_t, size_t, int, int, int, int, const infinilm::runtime::DataType &, bool) const {
    unsupported("CompressedTensors");
}

infinilm::runtime::Tensor CompressedTensors::forward(
    infinilm::runtime::Runtime &, const ParamsMap &, const infinilm::runtime::Tensor &, bool, float) const {
    unsupported("CompressedTensors");
}

std::vector<SplitParam> CompressedTensors::split_params(
    const std::unordered_map<std::string, infinilm::weights::Weight> &,
    const std::vector<SplitInfo> &, int, int, int, int) const {
    unsupported("CompressedTensors");
}

std::vector<ParamDescriptor> AWQ::get_param_layout(
    size_t, size_t, int, int, int, int, const infinilm::runtime::DataType &, bool) const {
    unsupported("AWQ");
}

infinilm::runtime::Tensor AWQ::forward(
    infinilm::runtime::Runtime &, const ParamsMap &, const infinilm::runtime::Tensor &, bool, float) const {
    unsupported("AWQ");
}

std::vector<SplitParam> AWQ::split_params(
    const std::unordered_map<std::string, infinilm::weights::Weight> &,
    const std::vector<SplitInfo> &, int, int, int, int) const {
    unsupported("AWQ");
}

std::shared_ptr<const W4Execution> AWQ::prepare_w4_after_loading(
    infinilm::runtime::Runtime &, const ParamsMap &, int, bool,
    std::optional<int64_t>) const {
    unsupported("AWQ");
}

std::shared_ptr<BaseQuantization> AWQ::process_weights_after_loading(
    infinilm::runtime::Runtime &, ParamsMap &, bool, int) const {
    unsupported("AWQ");
}

std::vector<ParamDescriptor> GPTQ::get_param_layout(
    size_t, size_t, int, int, int, int, const infinilm::runtime::DataType &, bool) const {
    unsupported("GPTQ");
}

infinilm::runtime::Tensor GPTQ::forward(
    infinilm::runtime::Runtime &, const ParamsMap &, const infinilm::runtime::Tensor &, bool, float) const {
    unsupported("GPTQ");
}

std::vector<SplitParam> GPTQ::split_params(
    const std::unordered_map<std::string, infinilm::weights::Weight> &,
    const std::vector<SplitInfo> &, int, int, int, int) const {
    unsupported("GPTQ");
}

std::shared_ptr<const W4Execution> GPTQ::prepare_w4_after_loading(
    infinilm::runtime::Runtime &, const ParamsMap &, int, bool,
    std::optional<int64_t>) const {
    unsupported("GPTQ");
}

std::shared_ptr<BaseQuantization> GPTQ::process_weights_after_loading(
    infinilm::runtime::Runtime &, ParamsMap &, bool, int) const {
    unsupported("GPTQ");
}

infinilm::runtime::Tensor W4Execution::forward(
    infinilm::runtime::Runtime &, const infinilm::runtime::Tensor &, bool, float) const {
    unsupported("W4 execution");
}

infinilm::runtime::Tensor W4Execution::forward_allreduce(
    infinilm::runtime::Runtime &, const infinilm::runtime::Tensor &, bool,
    infinicclComm_t, float) const {
    unsupported("W4 execution");
}
} // namespace infinilm::quantization

namespace infinilm {
namespace layers::moe {
EPConfig make_ep_config(const std::string &backend, size_t configured_ep_size,
                        size_t tp_size, size_t tp_rank) {
    if (!backend.empty() && backend != "disabled") {
        throw std::runtime_error("MoE expert parallelism is unavailable in the FM9G-only build");
    }
    (void)tp_size;
    EPConfig config;
    config.backend = EPBackend::Disabled;
    config.ep_rank = tp_rank;
    config.ep_size = configured_ep_size == 0 ? 1 : configured_ep_size;
    return config;
}
} // namespace layers::moe

void KVQuantUtils::quantize(infinilm::runtime::Runtime &,
                            infinilm::runtime::Tensor &,
                            infinilm::runtime::Tensor &,
                            infinilm::quantization::KVQuantAlgo,
                            const infinilm::runtime::Tensor &,
                            const infinilm::runtime::Tensor &) {
    throw std::runtime_error("INT8 KV cache quantization is unavailable in the FM9G-only build");
}

namespace quantization::w4_detail {
Metadata validate_metadata(infinilm::runtime::Runtime &, const ParamsMap &,
                           const nlohmann::json &, int, bool) {
    throw std::runtime_error("W4 metadata validation is unavailable in the FM9G-only build");
}

void drain_or_terminate(infinilm::runtime::Runtime &) noexcept {}
} // namespace quantization::w4_detail
} // namespace infinilm
