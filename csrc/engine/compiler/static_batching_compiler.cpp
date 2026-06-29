#include "static_batching_compiler.hpp"
#include "../../cache/cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

namespace infinilm::engine {
StaticBatchingCompiler::StaticBatchingCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
}

void StaticBatchingCompiler::compile() {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())) {
        size_t b = dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())->max_batch_size();
        InfinilmModel::Input input;
        input.input_ids = infinicore::Tensor::empty({b, 1}, infinicore::DataType::I64, infinicore::context::getDevice());
        input.position_ids = infinicore::Tensor::empty({b, 1}, infinicore::DataType::I64, infinicore::context::getDevice());
        input.past_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::I32, infinicore::context::getDevice());
        input.total_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::I32, infinicore::context::getDevice());
        input.block_tables = infinicore::Tensor::empty({b, 1}, infinicore::DataType::I32, infinicore::context::getDevice());

        set_zeros(input.input_ids.value());
        set_zeros(input.position_ids.value());
        set_zeros(input.past_sequence_lengths.value());

        std::vector<int32_t> total_sequence_lengths_vec(b, 1);
        infinicore::context::memcpyH2D(input.total_sequence_lengths.value()->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t), false);

        std::vector<int32_t> block_tables_vec(b);
        for (size_t i = 0; i < b; ++i) {
            block_tables_vec[i] = static_cast<int32_t>(i);
        }
        infinicore::context::memcpyH2D(input.block_tables.value()->data(), block_tables_vec.data(), b * sizeof(int32_t), false);

        // Attention reads attn_metadata from thread-local forward context.
        infinilm::global_state::get_forward_context().attn_metadata = {
            input.past_sequence_lengths,
            input.total_sequence_lengths,
            input.input_offsets,
            input.cu_seqlens,
            input.block_tables,
            input.slot_mapping,
        };

        model_->forward(input);
        infinicore::context::syncStream();
        model_->reset_runtime_state();
        infinicore::context::syncStream();

        barrier_->wait();
        model_->reset_runtime_state();
        infinicore::context::syncStream();
        infinicore::context::startGraphRecording();
        auto output = model_->forward(input);
        auto graph = infinicore::context::stopGraphRecording();
        barrier_->wait();

        auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

        compiled_map_[std::make_tuple(b, 1)] = CompiledResult{std::move(input), std::make_tuple(graph, shared_output)};
    }
}

StaticBatchingCompiler::Compiled StaticBatchingCompiler::get_compiled(
    const InfinilmModel::Input &input) {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())) {
        size_t batch_size = input.input_ids.value()->size(0);
        size_t seqlen = input.input_ids.value()->size(1);
        auto result = compiled_map_.find(std::make_tuple(batch_size, seqlen));
        if (result == compiled_map_.end()) {
            return std::make_tuple(nullptr, nullptr);
        } else {
            auto &graph_input = result->second.input;
            graph_input.input_ids.value()->copy_from(input.input_ids.value());
            graph_input.position_ids.value()->copy_from(input.position_ids.value());
            graph_input.past_sequence_lengths.value()->copy_from(input.past_sequence_lengths.value());
            graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
            model_->reset_runtime_state();

            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});
            return std::make_tuple(graph, shared_output);
        }
    } else {
        return std::make_tuple(nullptr, nullptr);
    }
}
} // namespace infinilm::engine
