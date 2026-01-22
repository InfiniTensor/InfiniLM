#include "static_batching_compiler.hpp"

#include "../../cache/cache.hpp"

namespace infinilm::engine {
StaticBatchingCompiler::StaticBatchingCompiler(const std::shared_ptr<InfinilmModel> &model)
    : GraphCompiler(model) {
}

void StaticBatchingCompiler::compile() {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())) {
        size_t b = dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())->max_batch_size();
        InfinilmModel::Input input;
        input.input_ids = infinicore::Tensor::empty({b, 1}, infinicore::DataType::I64, infinicore::context::getDevice());
        input.position_ids = infinicore::Tensor::empty({b, 1}, infinicore::DataType::I64, infinicore::context::getDevice());
        input.past_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
        input.total_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
        infinicore::context::startGraphRecording();
        auto output = model_->forward(input);
        auto graph = infinicore::context::stopGraphRecording();

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

            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});
            return std::make_tuple(graph, shared_output);
        }
    } else {
        return std::make_tuple(nullptr, nullptr);
    }
}
} // namespace infinilm::engine
