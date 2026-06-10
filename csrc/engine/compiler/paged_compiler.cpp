#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../compiled_prefill_flags.hpp"
#include "../../utils.hpp"

namespace infinilm::engine {
PagedCompiler::PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
    for (size_t b = 1; b < 64; ++b) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 64; b < 128; b += 16) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 128; b < 256; b += 32) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 256; b <= 512; b += 64) {
        decode_batch_sizes_.push_back(b);
    }
}

void PagedCompiler::compile() {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
        size_t max_batch_size = *std::max_element(decode_batch_sizes_.begin(), decode_batch_sizes_.end());
        compiled_map_decode_.clear();
        block_tables_holder_ = infinicore::Tensor::empty(
            {nblocks * max_batch_size}, infinicore::DataType::I32, infinicore::context::getDevice());
        set_zeros(block_tables_holder_);
        for (size_t b : decode_batch_sizes_) {
            InfinilmModel::Input input;
            input.input_ids = infinicore::Tensor::empty({1, b}, infinicore::DataType::I64, infinicore::context::getDevice());
            input.position_ids = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
            input.total_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::I32, infinicore::context::getDevice());
            set_zeros(input.input_ids.value());
            set_zeros(input.position_ids.value());
            set_zeros(input.total_sequence_lengths.value());
            std::vector<int32_t> total_sequence_lengths_vec(b, 1);
            infinicore::context::memcpyH2D(input.total_sequence_lengths.value()->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t), false);
            input.input_offsets = infinicore::Tensor::empty({b + 1}, infinicore::DataType::I32, infinicore::context::getDevice());
            std::vector<int32_t> input_offsets_vec(b + 1, 0);
            for (size_t i = 0; i <= b; i++) {
                input_offsets_vec[i] = i;
            }
            infinicore::context::memcpyH2D(input.input_offsets.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
            input.cu_seqlens = infinicore::Tensor::empty({b + 1}, infinicore::DataType::I32, infinicore::context::getDevice());
            infinicore::context::memcpyH2D(input.cu_seqlens.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
            const size_t block_per_req = nblocks;
            input.block_tables = block_tables_holder_->as_strided({b, block_per_req}, {(ptrdiff_t)block_per_req, 1});
            input.slot_mapping = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
            set_zeros(input.slot_mapping.value());

            // Attention reads attn_metadata from thread-local forward context.
            infinilm::global_state::get_forward_context().attn_metadata = {
                input.past_sequence_lengths,
                input.total_sequence_lengths,
                input.input_offsets,
                input.cu_seqlens,
                input.block_tables,
                input.slot_mapping,
            };

            barrier_->wait();
            compiled_map_decode_[b] = capture_forward_graph_(std::move(input));
        }

        // Prefill graphs: one capture per bucket (MVP: 4096 full prefill, batch_size == 1).
        compiled_map_prefill_.clear();
        if (!skip_cpp_prefill_graph() && !native_piecewise_prefill_enabled()) {
        for (size_t seq_bucket : prefill_seq_buckets_) {
            const size_t S = seq_bucket;
            InfinilmModel::Input input;
            input.input_ids = infinicore::Tensor::empty({1, S}, infinicore::DataType::I64, infinicore::context::getDevice());
            input.position_ids = infinicore::Tensor::empty({S}, infinicore::DataType::I64, infinicore::context::getDevice());
            input.past_sequence_lengths = infinicore::Tensor::empty({1}, infinicore::DataType::I32, infinicore::context::getDevice());
            input.total_sequence_lengths = infinicore::Tensor::empty({1}, infinicore::DataType::I32, infinicore::context::getDevice());
            set_zeros(input.input_ids.value());
            set_zeros(input.position_ids.value());
            set_zeros(input.past_sequence_lengths.value());
            set_zeros(input.total_sequence_lengths.value());

            std::vector<int32_t> past_lengths_vec(1, 0);
            std::vector<int32_t> total_lengths_vec(1, static_cast<int32_t>(S));
            infinicore::context::memcpyH2D(
                input.past_sequence_lengths.value()->data(), past_lengths_vec.data(), sizeof(int32_t), false);
            infinicore::context::memcpyH2D(
                input.total_sequence_lengths.value()->data(), total_lengths_vec.data(), sizeof(int32_t), false);

            input.input_offsets = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::context::getDevice());
            std::vector<int32_t> input_offsets_vec{0, static_cast<int32_t>(S)};
            infinicore::context::memcpyH2D(
                input.input_offsets.value()->data(), input_offsets_vec.data(), 2 * sizeof(int32_t), false);

            input.cu_seqlens = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::context::getDevice());
            infinicore::context::memcpyH2D(
                input.cu_seqlens.value()->data(), input_offsets_vec.data(), 2 * sizeof(int32_t), false);

            const size_t block_per_req = nblocks;
            input.block_tables = block_tables_holder_->as_strided({1, block_per_req}, {(ptrdiff_t)block_per_req, 1});
            input.slot_mapping = infinicore::Tensor::empty({S}, infinicore::DataType::I64, infinicore::context::getDevice());
            set_zeros(input.slot_mapping.value());

            infinilm::global_state::get_forward_context().attn_metadata = {
                input.past_sequence_lengths,
                input.total_sequence_lengths,
                input.input_offsets,
                input.cu_seqlens,
                input.block_tables,
                input.slot_mapping,
            };

            barrier_->wait();
            compiled_map_prefill_[seq_bucket] = capture_forward_graph_(std::move(input));
        }
        }
    }
}

PagedCompiler::CompiledResult PagedCompiler::capture_forward_graph_(InfinilmModel::Input input) {
    auto &ctx = infinilm::global_state::get_forward_context();
    ctx.defer_row_parallel_allreduce = true;
    ctx.deferred_allreduces.clear();

    barrier_->wait();
    infinicore::context::startGraphRecording();
    auto output = model_->forward(input);
    auto graph = infinicore::context::stopGraphRecording();
    barrier_->wait();

    ctx.defer_row_parallel_allreduce = false;
    auto post_graph_allreduces = std::make_shared<std::vector<global_state::DeferredAllreduce>>(
        std::move(ctx.deferred_allreduces));
    ctx.deferred_allreduces.clear();
    global_state::run_deferred_allreduces(*post_graph_allreduces);

    auto shared_output = std::shared_ptr<InfinilmModel::Output>(
        new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

    return CompiledResult{
        std::move(input),
        std::make_tuple(graph, shared_output),
        std::move(post_graph_allreduces),
    };
}

void PagedCompiler::record_graph_hit(bool is_prefill) {
    if (is_prefill) {
        ++prefill_graph_hits_;
    } else {
        ++decode_graph_hits_;
    }
}

void PagedCompiler::record_graph_miss(bool is_prefill) {
    if (is_prefill) {
        ++prefill_graph_misses_;
    } else {
        ++decode_graph_misses_;
    }
}

PagedCompiler::GraphStats PagedCompiler::graph_stats() const {
    return GraphStats{
        prefill_graph_hits_,
        prefill_graph_misses_,
        decode_graph_hits_,
        decode_graph_misses_,
    };
}

static size_t compute_prefill_len(const InfinilmModel::Input &input) {
    if (input.input_offsets.has_value()) {
        const auto &offsets = input.input_offsets.value();
        const size_t n = offsets->size(0);
        if (n >= 2) {
            auto cpu_offsets = offsets->to(infinicore::Device::cpu());
            const auto *data = reinterpret_cast<const int32_t *>(cpu_offsets->data());
            return static_cast<size_t>(data[n - 1] - data[0]);
        }
    }
    return input.input_ids.value()->size(1);
}

PagedCompiler::Compiled PagedCompiler::get_compiled(const InfinilmModel::Input &input) {
    auto &forward_ctx = infinilm::global_state::get_forward_context();
    auto attach_post_graph_allreduces = [&forward_ctx](const CompiledResult &cr) {
        if (cr.post_graph_allreduces) {
            forward_ctx.post_graph_allreduces = *cr.post_graph_allreduces;
        } else {
            forward_ctx.post_graph_allreduces.clear();
        }
    };
    auto miss = [&forward_ctx]() {
        forward_ctx.post_graph_allreduces.clear();
        return Compiled{nullptr, nullptr};
    };

    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t batch_size = input.block_tables.value()->size(0);
        size_t block_per_req = input.block_tables.value()->size(1);

        if (batch_size != input.input_ids.value()->size(1)) {
            if (skip_cpp_prefill_graph() || native_piecewise_prefill_enabled()) {
                return miss();
            }
            const size_t compute_len = compute_prefill_len(input);
            auto result = compiled_map_prefill_.find(compute_len);
            if (result == compiled_map_prefill_.end()) {
                return miss();
            }

            auto &graph_input = result->second.input;

            graph_input.input_ids.value()->copy_from(input.input_ids.value());
            graph_input.position_ids.value()->copy_from(input.position_ids.value());
            if (graph_input.past_sequence_lengths.has_value() && input.past_sequence_lengths.has_value()) {
                graph_input.past_sequence_lengths.value()->copy_from(input.past_sequence_lengths.value());
            }
            graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
            graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
            graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());

            const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
            if (block_per_req > compiled_block_per_req) {
                return miss();
            }

            auto &graph_block_tables = graph_input.block_tables.value();
            set_minus_one(graph_block_tables);
            graph_block_tables->narrow({{1, 0, block_per_req}})->copy_from(input.block_tables.value());
            graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());

            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});
            attach_post_graph_allreduces(result->second);

            return std::make_tuple(graph, shared_output);
        } else {
            auto result = compiled_map_decode_.find(batch_size);
            if (result == compiled_map_decode_.end()) {
                return miss();
            }
            auto &graph_input = result->second.input;

            graph_input.input_ids.value()->copy_from(input.input_ids.value());
            graph_input.position_ids.value()->copy_from(input.position_ids.value());
            graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
            graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
            graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());

            const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
            if (block_per_req > compiled_block_per_req) {
                // Runtime width exceeds compiled graph slot; fall back to eager path.
                return miss();
            }

            // Initialize full padding to -1, then overwrite the narrowed logical region.
            // This matches scheduler padding semantics without risking -1 access during graph recording.
            auto &graph_block_tables = graph_input.block_tables.value();
            set_minus_one(graph_block_tables);
            graph_input.block_tables.value()->narrow({{1, 0, block_per_req}})->copy_from(input.block_tables.value());
            graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());

            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});
            attach_post_graph_allreduces(result->second);

            return std::make_tuple(graph, shared_output);
        }
    } else {
        return miss();
    }
}

} // namespace infinilm::engine
