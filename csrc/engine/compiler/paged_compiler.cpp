#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../compiled_prefill_flags.hpp"
#include "../../utils.hpp"
#include "attn_metadata_utils.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace infinilm::engine {
namespace {

std::vector<size_t> parse_decode_cg_batches_() {
    std::vector<size_t> batches;
    if (const char *raw = std::getenv("INFINI_DECODE_CG_BATCHES")) {
        std::string spec(raw);
        size_t start = 0;
        while (start < spec.size()) {
            const size_t comma = spec.find(',', start);
            const std::string token =
                spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!token.empty()) {
                batches.push_back(static_cast<size_t>(std::stoul(token)));
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }
    if (batches.empty()) {
        // Default ladder (was 76 sizes up to 512). Decode graphs are not
        // replayed under TP>1 (see get_compiled); smaller ladder saves compile VRAM on TP=1.
        for (size_t b = 1; b <= 16; ++b) {
            batches.push_back(b);
        }
        batches.push_back(32);
        batches.push_back(64);
    }
    std::sort(batches.begin(), batches.end());
    batches.erase(std::unique(batches.begin(), batches.end()), batches.end());
    return batches;
}

} // namespace

PagedCompiler::PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
    decode_batch_sizes_ = parse_decode_cg_batches_();
    spdlog::info(
        "paged decode CG: capture batches=[{}]",
        [&]() {
            std::string s;
            for (size_t i = 0; i < decode_batch_sizes_.size(); ++i) {
                if (i) {
                    s += ',';
                }
                s += std::to_string(decode_batch_sizes_[i]);
            }
            return s;
        }());
}

void PagedCompiler::compile() {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
        size_t max_batch_size = *std::max_element(decode_batch_sizes_.begin(), decode_batch_sizes_.end());
        compiled_map_decode_.clear();
        block_tables_holder_ = infinicore::Tensor::empty(
            {nblocks * max_batch_size}, infinicore::DataType::I32, infinicore::context::getDevice());
        set_zeros(block_tables_holder_);
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        if (rank_info.tp_size > 1 && !decode_cg_tp_enabled()) {
            spdlog::info(
                "paged decode CG: skip capture (tp_size={} > 1; decode graphs are eager-only under TP; "
                "set INFINI_DECODE_CG_TP=1 to opt in)",
                rank_info.tp_size);
        } else if (skip_monolithic_decode_graph()) {
            spdlog::info(
                "paged decode CG: skip monolithic capture (INFINI_DECODE_GRAPH_ONLY / "
                "INFINI_SKIP_MONOLITHIC_DECODE_CG); MoE+FA host-break ready — use "
                "PiecewiseDecodeCompiler when INFINI_DECODE_PIECEWISE is on");
        } else {
            if (rank_info.tp_size > 1) {
                spdlog::info(
                    "paged decode CG: capturing under TP (tp_size={}, INFINI_DECODE_CG_TP=1)",
                    rank_info.tp_size);
            }
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
    ctx.deferred_allreduces.clear();

    const size_t batch_size = input.block_tables.value()->size(0);
    const size_t input_width = input.input_ids.value()->size(1);
    const bool is_decode_capture = (batch_size == input_width);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    // RC-3: under TP, deferred row-parallel AR runs after graph->run(), but lm_head already
    // consumed non-allreduced hidden states inside the graph. Keep AR inline for decode CG.
    const bool defer_ar =
        !(is_decode_capture && rank_info.tp_size > 1 && decode_cg_tp_enabled());
    ctx.defer_row_parallel_allreduce = defer_ar;

    // Phase-scoped FA/MoE host_break under INFINI_CUDAGRAPH_POLICY=full_and_piecewise.
    const auto phase = is_decode_capture ? infinicore::context::InferencePhase::Decode
                                         : infinicore::context::InferencePhase::Prefill;
    infinicore::context::InferencePhaseGuard phase_guard(phase);

    barrier_->wait();
    infinicore::context::startGraphRecording();
    auto output = model_->forward(input);
    auto graph = infinicore::context::stopGraphRecording();
    barrier_->wait();

    ctx.defer_row_parallel_allreduce = false;
    auto post_graph_allreduces = std::make_shared<std::vector<global_state::DeferredAllreduce>>(
        std::move(ctx.deferred_allreduces));
    ctx.deferred_allreduces.clear();
    if (!post_graph_allreduces->empty()) {
        global_state::run_deferred_allreduces(*post_graph_allreduces);
    }

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
            const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
            if (rank_info.tp_size > 1 && !decode_cg_tp_enabled()) {
                // Decode CUDAGraph replay under TP requires INFINI_DECODE_CG_TP=1.
                return miss();
            }
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

            // RC-2 analog: refresh attn_metadata from graph_input narrowed to runtime shapes.
            attn_metadata_utils::set_attn_metadata_for_decode_batch(graph_input, input);


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
