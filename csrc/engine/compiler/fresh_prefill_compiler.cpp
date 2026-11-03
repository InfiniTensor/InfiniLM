#include "fresh_prefill_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include <cstdlib>
#include <cstring>
#include <unordered_set>
namespace infinilm::engine {
namespace {
bool enabled() {
    const auto *flag = std::getenv("INFINILM_ASCEND_FRESH_PREFILL_GRAPH");
    return flag && std::strcmp(flag, "1") == 0;
}
bool cpu_tensor(const std::optional<infinicore::Tensor> &t,
                infinicore::DataType dtype, const std::vector<size_t> &shape) {
    return t && t.value()->device().getType() == infinicore::Device::Type::CPU
        && t.value()->dtype() == dtype && t.value()->shape() == shape
        && t.value()->is_contiguous();
}
} // namespace
void FreshPrefillCompiler::compile() {
    if (entry_) {
        infinicore::context::syncStream();
    }
    entry_.reset();
}
FreshPrefillCompiler::Compiled FreshPrefillCompiler::get_compiled(const InfinilmModel::Input &input) {
    using namespace infinicore;
    const Compiled fallback{nullptr, nullptr};
    if (!enabled() || context::getDevice().getType() != Device::Type::ASCEND
        || input.sample_all_positions || input.pixel_values || input.image_bound
        || input.tgt_sizes || input.image_grid_thw || input.image_req_ids
        || input.visual_token_ranges || input.target_hidden_states
        || input.mamba_init_state_indices || input.mamba_final_state_indices) {
        return fallback;
    }
    const auto *cfg = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    if (!cfg || !input.input_ids || input.input_ids.value()->ndim() != 2) {
        return fallback;
    }
    const auto T = input.input_ids.value()->size(1);
    // A bounded first-shape cache avoids retaining graphs for arbitrary lengths.
    // Short prompts use eager submission; they rarely amortize graph creation.
    if (T < 1024 || T > 2048
        || !cpu_tensor(input.input_ids, DataType::I64, {1, T})
        || !cpu_tensor(input.total_sequence_lengths, DataType::I32, {1})
        || !cpu_tensor(input.input_offsets, DataType::I32, {2})
        || !cpu_tensor(input.cu_seqlens, DataType::I32, {2})
        || !cpu_tensor(input.slot_mapping, DataType::I64, {T})
        || !input.position_ids || !input.block_tables) {
        return fallback;
    }
    const auto &model_cfg = model_->get_model_config();
    if (model_cfg) {
        const auto type = model_cfg->get<std::string>("model_type");
        if (type != "fm9g7b" && type != "llama" && type != "qwen2") {
            return fallback;
        }
    }
    const auto &ctx = global_state::get_forward_context();
    if (global_state::get_infinilm_config().attention_backend != backends::AttentionBackend::FLASH_ATTN) {
        return fallback;
    }
    for (const auto &state : ctx.conv_state_vec) {
        if (state) {
            return fallback;
        }
    }
    for (const auto &state : ctx.ssm_state_vec) {
        if (state) {
            return fallback;
        }
    }
    const auto pos_shape = input.position_ids.value()->shape();
    if (!(cpu_tensor(input.position_ids, DataType::I64, {T})
          || cpu_tensor(input.position_ids, DataType::I64, {1, T}))) {
        return fallback;
    }
    const auto blocks_shape = input.block_tables.value()->shape();
    if (blocks_shape.size() != 2 || blocks_shape[0] != 1 || blocks_shape[1] == 0
        || !cpu_tensor(input.block_tables, DataType::I32, blocks_shape)) {
        return fallback;
    }
    const auto *offsets = reinterpret_cast<const int32_t *>(input.input_offsets.value()->data());
    const auto *cu = reinterpret_cast<const int32_t *>(input.cu_seqlens.value()->data());
    const auto *lengths = reinterpret_cast<const int32_t *>(input.total_sequence_lengths.value()->data());
    if (offsets[0] != 0 || offsets[1] != static_cast<int32_t>(T)
        || cu[0] != 0 || cu[1] != offsets[1] || lengths[0] != offsets[1]) {
        return fallback;
    }
    if (input.past_sequence_lengths) {
        if (!cpu_tensor(input.past_sequence_lengths, DataType::I32, {1})
            || *reinterpret_cast<const int32_t *>(input.past_sequence_lengths.value()->data()) != 0) {
            return fallback;
        }
    }
    const auto *slots = reinterpret_cast<const int64_t *>(input.slot_mapping.value()->data());
    const size_t capacity = cfg->num_blocks() * cfg->block_size();
    std::unordered_set<int64_t> seen;
    seen.reserve(T);
    for (size_t i = 0; i < T; ++i) {
        if (slots[i] < 0 || static_cast<uint64_t>(slots[i]) >= capacity || !seen.insert(slots[i]).second) {
            return fallback;
        }
    }
    if (entry_) {
        auto &dst = entry_->input;
        if (dst.input_ids.value()->shape() != input.input_ids.value()->shape()
            || dst.position_ids.value()->shape() != pos_shape
            || dst.block_tables.value()->shape() != blocks_shape
            || dst.past_sequence_lengths.has_value() != input.past_sequence_lengths.has_value()) {
            return fallback;
        }
        // Refresh every graph input, including shape-constant metadata: decode
        // graphs share the allocator arena and can overwrite prefill storage.
        dst.input_ids.value()->copy_from(input.input_ids.value());
        dst.position_ids.value()->copy_from(input.position_ids.value());
        if (dst.past_sequence_lengths) {
            dst.past_sequence_lengths.value()->copy_from(input.past_sequence_lengths.value());
        }
        dst.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
        dst.input_offsets.value()->copy_from(input.input_offsets.value());
        dst.cu_seqlens.value()->copy_from(input.cu_seqlens.value());
        dst.block_tables.value()->copy_from(input.block_tables.value());
        dst.slot_mapping.value()->copy_from(input.slot_mapping.value());
        graph::bind_eager_host_int_array(dst.slot_mapping.value(), slots, T);
        model_->reset_runtime_state();
        auto output = std::make_shared<InfinilmModel::Output>(InfinilmModel::Output{
            std::get<1>(entry_->compiled)->logits->resume_from_blob_()});
        return {std::get<0>(entry_->compiled), output};
    }
    auto device_input = input;
    auto move = [](std::optional<Tensor> &t) {
        if (t) {
            t = t.value()->to(context::getDevice());
        }
    };
    move(device_input.input_ids);
    move(device_input.position_ids);
    move(device_input.past_sequence_lengths);
    move(device_input.total_sequence_lengths);
    move(device_input.input_offsets);
    move(device_input.cu_seqlens);
    move(device_input.block_tables);
    move(device_input.slot_mapping);
    auto &forward = global_state::get_forward_context();
    const auto old_metadata = forward.attn_metadata;
    struct Restore {
        global_state::AttentionMetadata &target;
        global_state::AttentionMetadata saved;
        ~Restore() { target = saved; }
    } restore{forward.attn_metadata, old_metadata};
    forward.attn_metadata = {device_input.past_sequence_lengths,
                             device_input.total_sequence_lengths, device_input.input_offsets,
                             device_input.cu_seqlens, device_input.block_tables, device_input.slot_mapping, T, T};
    graph::bind_eager_host_int_array(device_input.input_offsets.value(), offsets, 2);
    graph::bind_eager_host_int_array(device_input.cu_seqlens.value(), cu, 2);
    graph::bind_eager_host_int_array(device_input.total_sequence_lengths.value(), lengths, 1);
    graph::bind_eager_host_int_array(device_input.slot_mapping.value(), slots, T);
    context::syncStream();
    barrier_->wait();
    context::startGraphRecording();
    try {
        auto output = model_->forward(device_input);
        auto graph = context::stopGraphRecording();
        // The per-forward eager bindings are cleared by to_model_input() on
        // the next token. These shape-constant arrays must belong to the graph.
        graph->bind_host_int_array(device_input.input_offsets.value(), offsets, 2);
        graph->bind_host_int_array(device_input.cu_seqlens.value(), cu, 2);
        graph->bind_host_int_array(device_input.total_sequence_lengths.value(), lengths, 1);
        barrier_->wait();
        auto shared_output = std::make_shared<InfinilmModel::Output>(InfinilmModel::Output{
            graph::GraphTensor(output.logits)});
        entry_ = Entry{std::move(device_input), {graph, shared_output}};
        spdlog::info("Ascend fresh prefill graph cached: tokens={} (one entry)", T);
        // Return persistent output ownership; RankWorker launches the graph.
        return {graph, std::make_shared<InfinilmModel::Output>(InfinilmModel::Output{
                           shared_output->logits->resume_from_blob_()})};
    } catch (...) {
        if (context::isGraphRecording()) {
            context::cancelGraphRecording();
        }
        throw;
    }
}
} // namespace infinilm::engine
