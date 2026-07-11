#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../workspace/workspace_context.hpp"

#include <string>

namespace infinilm::engine {

namespace {
bool same_shape_and_dtype(const infinicore::Tensor &lhs, const infinicore::Tensor &rhs) {
    return lhs->shape() == rhs->shape() && lhs->dtype() == rhs->dtype();
}

bool is_mrope_decode_position_ids(const infinicore::Tensor &position_ids, size_t batch_size) {
    const auto shape = position_ids->shape();
    return shape.size() == 2 && shape[0] == batch_size && shape[1] == 3;
}
} // namespace

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
        compiled_map_decode_mrope_.clear();
        block_tables_holder_ = infinicore::Tensor::empty(
            {nblocks * max_batch_size}, infinicore::DataType::I32, infinicore::context::getDevice());
        set_zeros(block_tables_holder_);

        auto make_decode_input = [&](size_t b, bool mrope_position_ids) {
            InfinilmModel::Input input;
            input.input_ids = infinicore::Tensor::empty({1, b}, infinicore::DataType::I32, infinicore::context::getDevice());
            if (mrope_position_ids) {
                input.position_ids = infinicore::Tensor::empty({b, 3}, infinicore::DataType::I64, infinicore::context::getDevice());
            } else {
                input.position_ids = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
            }
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
            return input;
        };

        {
            const size_t warmup_batch_size = std::min(max_batch_size, static_cast<size_t>(64));
            auto input = make_decode_input(warmup_batch_size, false);
            {
                WorkspaceForwardGuard forward_guard(maybe_current_workspace());
                model_->forward(input);
            }
            infinicore::context::syncStream();
            // Warmup runs the eager Marlin path and may leave per-layer lock
            // workspaces dirty. Reset before CUDA graph capture so capture
            // starts from the same all-zero lock state as normal execution.
            model_->reset_runtime_state();
            infinicore::context::syncStream();
        }

        auto capture_decode_graph = [&](size_t b,
                                        bool mrope_position_ids,
                                        std::unordered_map<size_t, CompiledResult> &compiled_map,
                                        const std::string &scope_name) {
            auto input = make_decode_input(b, mrope_position_ids);

            barrier_->wait();
            {
                WorkspaceForwardGuard forward_guard(maybe_current_workspace());
                (void)model_->forward(input);
            }
            infinicore::context::syncStream();
            // Capture must not start with stale Marlin locks from previous
            // warmup/capture attempts. This reset is intentionally outside
            // graph capture; the current implementation still pays a memset
            // before every graph replay in get_compiled().
            model_->reset_runtime_state();
            infinicore::context::syncStream();
            barrier_->wait();
            infinicore::context::startGraphRecording();
            WorkspaceCollectiveScopeGuard collective_scope_guard(
                maybe_current_workspace(),
                scope_name + std::to_string(b));
            WorkspaceForwardGuard forward_guard(maybe_current_workspace());
            // Capture runtime workspace resets so graph replay does not need
            // an extra host-launched reset phase before every decode token.
            model_->reset_runtime_state();
            auto output = model_->forward(input);
            auto graph = infinicore::context::stopGraphRecording();
            barrier_->wait();

            auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

            compiled_map[b] = CompiledResult{std::move(input), std::make_tuple(graph, shared_output)};
        };

        const bool compile_mrope_decode = model_->supports_mrope_position_ids();
        for (size_t b : decode_batch_sizes_) {
            capture_decode_graph(b, false, compiled_map_decode_, "paged_decode.");
            if (compile_mrope_decode) {
                capture_decode_graph(b, true, compiled_map_decode_mrope_, "paged_decode_mrope.");
            }
        }
    }
}

PagedCompiler::Compiled PagedCompiler::get_compiled(const InfinilmModel::Input &input) {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t batch_size = input.block_tables.value()->size(0);
        size_t block_per_req = input.block_tables.value()->size(1);

        // only support decode only batch
        if (batch_size != input.input_ids.value()->size(1)) {
            return {nullptr, nullptr};
        } else {
            const bool mrope_position_ids = is_mrope_decode_position_ids(input.position_ids.value(), batch_size);
            auto &compiled_map = mrope_position_ids ? compiled_map_decode_mrope_ : compiled_map_decode_;
            auto result = compiled_map.find(batch_size);
            if (result == compiled_map.end()) {
                return {nullptr, nullptr};
            }
            auto &graph_input = result->second.input;

            if (!same_shape_and_dtype(graph_input.input_ids.value(), input.input_ids.value()) ||
                !same_shape_and_dtype(graph_input.position_ids.value(), input.position_ids.value()) ||
                !same_shape_and_dtype(graph_input.total_sequence_lengths.value(), input.total_sequence_lengths.value()) ||
                !same_shape_and_dtype(graph_input.input_offsets.value(), input.input_offsets.value()) ||
                !same_shape_and_dtype(graph_input.cu_seqlens.value(), input.cu_seqlens.value()) ||
                !same_shape_and_dtype(graph_input.slot_mapping.value(), input.slot_mapping.value())) {
                return {nullptr, nullptr};
            }

            graph_input.input_ids.value()->copy_from(input.input_ids.value());
            graph_input.position_ids.value()->copy_from(input.position_ids.value());
            graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
            graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
            graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());

            const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
            if (block_per_req > compiled_block_per_req) {
                // Runtime width exceeds compiled graph slot; fall back to eager path.
                return {nullptr, nullptr};
            }

            // Initialize only the active graph rows to -1, then overwrite the
            // runtime logical region. Avoid clearing the full preallocated
            // holder on every decode token.
            auto &graph_block_tables = graph_input.block_tables.value();
            if (graph_block_tables->dtype() != input.block_tables.value()->dtype()) {
                return {nullptr, nullptr};
            }
            set_minus_one_device_async(graph_block_tables);
            graph_block_tables->narrow({{1, 0, block_per_req}})->copy_from(input.block_tables.value());
            graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());
            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});

            return std::make_tuple(graph, shared_output);
        }
    } else {
        return {nullptr, nullptr};
    }
}

} // namespace infinilm::engine
