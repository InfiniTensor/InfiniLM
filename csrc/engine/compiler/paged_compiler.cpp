#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <spdlog/spdlog.h>

namespace infinilm::engine {

PagedCompiler::PagedCompiler(
    const std::shared_ptr<InfinilmModel> &model,
    RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
}

void PagedCompiler::compile() {
    compiled_map_decode_.clear();
    graph_disabled_batches_.clear();
    initialized_ = false;
    num_blocks_ = 0;
    block_size_ = 0;

    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        model_->get_cache_config());
    if (paged_config == nullptr) {
        return;
    }
    num_blocks_ = paged_config->num_blocks();
    block_size_ = paged_config->block_size();
    initialized_ = num_blocks_ > 0 && block_size_ > 0;
}

InfinilmModel::Input PagedCompiler::make_decode_input(
    size_t b,
    size_t block_per_req) const {
    InfinilmModel::Input input;
    const auto device = infinicore::context::getDevice();
    input.input_ids = infinicore::Tensor::empty(
        {1, b}, infinicore::DataType::I64, device);
    input.position_ids = infinicore::Tensor::empty(
        {b}, infinicore::DataType::I64, device);
    input.total_sequence_lengths = infinicore::Tensor::empty(
        {b}, infinicore::DataType::I32, device);
    set_zeros(input.input_ids.value());
    set_zeros(input.position_ids.value());
    std::vector<int32_t> total_sequence_lengths_vec(b, 1);
    infinicore::context::memcpyH2D(
        input.total_sequence_lengths.value()->data(),
        total_sequence_lengths_vec.data(),
        b * sizeof(int32_t),
        false);

    input.input_offsets = infinicore::Tensor::empty(
        {b + 1}, infinicore::DataType::I32, device);
    std::vector<int32_t> input_offsets_vec(b + 1, 0);
    for (size_t i = 0; i <= b; ++i) {
        input_offsets_vec[i] = static_cast<int32_t>(i);
    }
    infinicore::context::memcpyH2D(
        input.input_offsets.value()->data(),
        input_offsets_vec.data(),
        (b + 1) * sizeof(int32_t),
        false);
    input.request_ids = infinicore::Tensor::empty(
        {b}, infinicore::DataType::I32, device);
    infinicore::context::memcpyH2D(
        input.request_ids.value()->data(),
        input_offsets_vec.data(),
        b * sizeof(int32_t),
        false);
    input.cu_seqlens = infinicore::Tensor::empty(
        {b + 1}, infinicore::DataType::I32, device);
    infinicore::context::memcpyH2D(
        input.cu_seqlens.value()->data(),
        input_offsets_vec.data(),
        (b + 1) * sizeof(int32_t),
        false);
    input.block_tables = infinicore::Tensor::empty(
        {b, block_per_req}, infinicore::DataType::I32, device);
    set_zeros(input.block_tables.value());
    input.slot_mapping = infinicore::Tensor::empty(
        {b}, infinicore::DataType::I64, device);
    // Graph::instantiate runs several warmups. Padding slots prevent those
    // dummy forwards from modifying a live request's MLA/indexer KV caches.
    set_minus_one_device_async(input.slot_mapping.value());

    infinilm::global_state::get_forward_context().attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.request_ids,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping,
        static_cast<int64_t>(block_per_req * block_size_),
    };
    return input;
}

void PagedCompiler::compile_decode(
    size_t batch_size,
    size_t block_per_req) {
    if (!initialized_ || batch_size == 0
        || batch_size > model_->max_decode_graph_batch_size()
        || block_per_req == 0 || block_per_req > num_blocks_
        || compiled_map_decode_.find(batch_size) != compiled_map_decode_.end()
        || graph_disabled_batches_.find(batch_size)
               != graph_disabled_batches_.end()) {
        return;
    }

    auto input = make_decode_input(batch_size, block_per_req);

    // Warm the exact static decode shape, then capture it once. All ranks
    // enter warmup/capture in lock-step so TP collectives are instantiated in
    // the same order.
    barrier_->wait();
    auto fail_collectively = [&](bool local_ok,
                                 const char *stage,
                                 const std::string &local_error) {
        const bool all_ok = barrier_->wait(local_ok);
        if (all_ok) {
            return false;
        }
        if (infinicore::context::isGraphRecording()) {
            infinicore::context::cancelGraphRecording();
        }
        graph_disabled_batches_.insert(batch_size);
        const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
        if (!local_ok) {
            spdlog::warn(
                "[{}] disabling decode graph batch {} after {} failure: {}",
                rank.tp_rank,
                batch_size,
                stage,
                local_error);
        } else if (rank.tp_rank == 0) {
            spdlog::warn(
                "disabling decode graph batch {} after a peer failed during {}",
                batch_size,
                stage);
        }
        return true;
    };

    bool local_ok = true;
    std::string local_error;
    try {
        (void)model_->forward(input);
        infinicore::context::syncStream();
        model_->reset_runtime_state();
        infinicore::context::syncStream();
    } catch (const std::exception &e) {
        local_ok = false;
        local_error = e.what();
    } catch (...) {
        local_ok = false;
        local_error = "unknown exception";
    }
    if (fail_collectively(local_ok, "warmup", local_error)) {
        return;
    }

    InfinilmModel::Output output;
    local_ok = true;
    local_error.clear();
    try {
        infinicore::context::startGraphRecording();
        // Graph-aware memsets put Marlin lock resets inside every replay instead
        // of paying separate eager launches from get_compiled().
        model_->reset_runtime_state();
        output = model_->forward(input);
    } catch (const std::exception &e) {
        local_ok = false;
        local_error = e.what();
    } catch (...) {
        local_ok = false;
        local_error = "unknown exception";
    }
    if (fail_collectively(local_ok, "recording", local_error)) {
        return;
    }

    std::shared_ptr<infinicore::graph::Graph> graph;
    local_ok = true;
    local_error.clear();
    try {
        graph = infinicore::context::stopGraphRecording();
        if (graph == nullptr) {
            throw std::runtime_error("graph recording returned no graph");
        }
    } catch (const std::exception &e) {
        local_ok = false;
        local_error = e.what();
    } catch (...) {
        local_ok = false;
        local_error = "unknown exception";
    }
    if (fail_collectively(local_ok, "instantiation", local_error)) {
        return;
    }

    auto shared_output = std::make_shared<InfinilmModel::Output>(
        InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});
    auto padding_total_sequence_lengths = infinicore::Tensor::empty(
        {batch_size}, infinicore::DataType::I32,
        infinicore::context::getDevice());
    padding_total_sequence_lengths->copy_from(
        input.total_sequence_lengths.value());
    auto padding_request_ids = infinicore::Tensor::empty(
        {batch_size}, infinicore::DataType::I32, infinicore::context::getDevice());
    padding_request_ids->copy_from(input.request_ids.value());
    compiled_map_decode_[batch_size] = CompiledResult{
        std::move(input),
        std::make_tuple(std::move(graph), std::move(shared_output)),
        std::move(padding_total_sequence_lengths),
        std::move(padding_request_ids),
        {},
    };
    if (global_state::get_tensor_model_parallel_rank_info().tp_rank == 0) {
        spdlog::info("compiled paged decode graph for batch {}", batch_size);
    }
}

PagedCompiler::Compiled PagedCompiler::get_compiled(
    const InfinilmModel::Input &input) {
    if (!initialized_ || !input.block_tables.has_value()
        || !input.input_ids.has_value()) {
        return {nullptr, nullptr};
    }
    const size_t batch_size = input.block_tables.value()->size(0);
    const size_t graph_batch_size = model_->decode_graph_batch_size(batch_size);
    const size_t block_per_req = input.block_tables.value()->size(1);

    // One input token per active request is the decode-only graph contract.
    // Prefill, mixed batches, oversized batches, and dynamic widths stay eager.
    if (batch_size == 0 || batch_size > model_->max_decode_graph_batch_size()
        || batch_size != input.input_ids.value()->size(1)) {
        return {nullptr, nullptr};
    }
    if (graph_batch_size < batch_size
        || graph_batch_size > model_->max_decode_graph_batch_size()) {
        return {nullptr, nullptr};
    }
    if (block_per_req == 0 || block_per_req > num_blocks_) {
        return {nullptr, nullptr};
    }
    if (graph_disabled_batches_.find(graph_batch_size)
        != graph_disabled_batches_.end()) {
        return {nullptr, nullptr};
    }
    if (compiled_map_decode_.find(graph_batch_size)
        == compiled_map_decode_.end()) {
        if (batch_size != graph_batch_size
            && global_state::get_tensor_model_parallel_rank_info().tp_rank == 0) {
            spdlog::info(
                "padding paged decode graph batch {} to bucket {}",
                batch_size,
                graph_batch_size);
        }
        // Match vLLM's persistent block table: one fixed-width graph input is
        // reused as requests grow across cache blocks. Capturing the first
        // observed width would force all later, longer decode steps eager.
        compile_decode(graph_batch_size, num_blocks_);
    }
    auto result = compiled_map_decode_.find(graph_batch_size);
    if (result == compiled_map_decode_.end()) {
        return {nullptr, nullptr};
    }
    auto &graph_input = result->second.input;

    const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
    if (block_per_req > compiled_block_per_req) {
        return {nullptr, nullptr};
    }

    const size_t padding_size = graph_batch_size - batch_size;
    auto copy_prefix = [batch_size](
                           infinicore::Tensor &dst,
                           const infinicore::Tensor &src,
                           size_t dim) {
        dst->narrow({{dim, 0, batch_size}})->copy_from(src);
    };
    copy_prefix(graph_input.input_ids.value(), input.input_ids.value(), 1);
    copy_prefix(graph_input.position_ids.value(), input.position_ids.value(), 0);
    copy_prefix(
        graph_input.total_sequence_lengths.value(),
        input.total_sequence_lengths.value(),
        0);
    copy_prefix(graph_input.request_ids.value(), input.request_ids.value(), 0);

    // Decode-only offsets are canonical [0, 1, ..., batch]. Keep the graph
    // bucket's preinitialized dummy suffix and update only the active prefix.
    graph_input.input_offsets.value()
        ->narrow({{0, 0, batch_size + 1}})
        ->copy_from(input.input_offsets.value());
    graph_input.cu_seqlens.value()
        ->narrow({{0, 0, batch_size + 1}})
        ->copy_from(input.cu_seqlens.value());

    if (padding_size > 0) {
        auto input_ids_tail = graph_input.input_ids.value()->narrow(
            {{1, batch_size, padding_size}});
        auto position_ids_tail = graph_input.position_ids.value()->narrow(
            {{0, batch_size, padding_size}});
        set_zeros_device_async(input_ids_tail);
        set_zeros_device_async(position_ids_tail);
        graph_input.total_sequence_lengths.value()
            ->narrow({{0, batch_size, padding_size}})
            ->copy_from(result->second.padding_total_sequence_lengths->narrow(
                {{0, batch_size, padding_size}}));
        graph_input.request_ids.value()
            ->narrow({{0, batch_size, padding_size}})
            ->copy_from(result->second.padding_request_ids->narrow(
                {{0, batch_size, padding_size}}));
    }

    auto &graph_block_tables = graph_input.block_tables.value();
    const size_t staging_key = batch_size * (compiled_block_per_req + 1) + block_per_req;
    auto staging_it = result->second.block_tables_staging.find(staging_key);
    if (staging_it == result->second.block_tables_staging.end()) {
        staging_it = result->second.block_tables_staging.emplace(
                                                            staging_key,
                                                            infinicore::Tensor::empty(
                                                                {batch_size, block_per_req},
                                                                infinicore::DataType::I32,
                                                                infinicore::context::getDevice()))
                         .first;
    }
    auto &block_tables_staging = staging_it->second;
    block_tables_staging->copy_from(input.block_tables.value());

    // The destination prefix is strided when the runtime width is smaller
    // than the graph width. Keep a persistent device source so the queued
    // rearrange never observes a freed temporary allocation.
    set_minus_one_device_async(graph_block_tables);
    graph_block_tables
        ->narrow(
            {{0, 0, batch_size},
             {1, 0, block_per_req}})
        ->copy_from(block_tables_staging);
    graph_input.slot_mapping.value()
        ->narrow({{0, 0, batch_size}})
        ->copy_from(input.slot_mapping.value());
    if (padding_size > 0) {
        // Dummy attention may read block 0, but must never write any cache.
        auto block_tables_tail = graph_block_tables->narrow(
            {{0, batch_size, padding_size}});
        auto slot_mapping_tail = graph_input.slot_mapping.value()->narrow(
            {{0, batch_size, padding_size}});
        set_zeros_device_async(block_tables_tail);
        set_minus_one_device_async(slot_mapping_tail);
    }

    auto graph = std::get<0>(result->second.compiled);
    auto shared_output = std::make_shared<InfinilmModel::Output>(
        InfinilmModel::Output{
            std::get<1>(result->second.compiled)->logits->resume_from_blob_()});
    return std::make_tuple(std::move(graph), std::move(shared_output));
}

} // namespace infinilm::engine
