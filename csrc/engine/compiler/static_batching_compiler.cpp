#include "static_batching_compiler.hpp"
#include "../../cache/cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace {
bool supports_static_graph_kv_cache(const infinicore::Tensor &kv_cache) {
    if (kv_cache.empty() || kv_cache->ndim() != 5 || kv_cache->size(0) != 2 || !kv_cache->is_contiguous()) {
        return false;
    }

    const auto dtype = kv_cache->dtype();
    const auto head_dim = kv_cache->size(4);
    return (dtype == infinicore::DataType::kFloat16 || dtype == infinicore::DataType::kBFloat16)
        && (head_dim == 64 || head_dim == 128);
}

std::optional<size_t> static_graph_cache_page_size(size_t batch_size) {
    const auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    std::optional<size_t> page_size;
    for (const auto &kv_cache : kv_cache_vec) {
        if (!supports_static_graph_kv_cache(kv_cache)
            || kv_cache->size(1) != batch_size
            || kv_cache->size(3) == 0
            || kv_cache->size(3) % 256 != 0
            || (page_size && kv_cache->size(3) != *page_size)) {
            return std::nullopt;
        }
        page_size = kv_cache->size(3);
    }
    return page_size;
}

bool supports_static_graph_attention(size_t batch_size) {
    const auto &config = infinilm::global_state::get_infinilm_config();
    if (!config.model_config
        || infinicore::context::getDevice().type() != infinicore::Device::Type::kNvidia
        || config.model_config->get_kv_quant_scheme() != infinilm::quantization::KVQuantAlgo::NONE) {
        return false;
    }

    return static_graph_cache_page_size(batch_size).has_value();
}
} // namespace

namespace infinilm::engine {
StaticBatchingCompiler::StaticBatchingCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
}

void StaticBatchingCompiler::compile() {
    compiled_map_.clear();
    const auto *static_config = dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config());
    if (static_config == nullptr) {
        return;
    }

    const size_t b = static_config->max_batch_size();
    if (!supports_static_graph_attention(b)) {
        return;
    }
    const size_t cache_page_size = *static_graph_cache_page_size(b);
    {
        InfinilmModel::Input input;
        input.input_ids = infinicore::Tensor::empty({b, 1}, infinicore::DataType::kInt64, infinicore::context::getDevice());
        input.position_ids = infinicore::Tensor::empty({b, 1}, infinicore::DataType::kInt64, infinicore::context::getDevice());
        input.past_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::kInt32, infinicore::context::getDevice());
        input.total_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::kInt32, infinicore::context::getDevice());
        set_zeros(input.input_ids.value());
        set_zeros(input.position_ids.value());
        set_zeros(input.past_sequence_lengths.value());
        std::vector<int32_t> total_sequence_lengths_vec(b, 1);
        infinicore::context::memcpyH2D(input.total_sequence_lengths.value()->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t), false);
        input.block_tables = infinicore::Tensor::empty({b, 1}, infinicore::DataType::kInt32, infinicore::context::getDevice());
        std::vector<int32_t> block_tables_vec(b);
        for (size_t i = 0; i < b; ++i) {
            block_tables_vec[i] = static_cast<int32_t>(i);
        }
        infinicore::context::memcpyH2D(input.block_tables.value()->data(), block_tables_vec.data(), b * sizeof(int32_t), false);
        input.slot_mapping = infinicore::Tensor::empty({b}, infinicore::DataType::kInt64, infinicore::context::getDevice());
        std::vector<int64_t> slot_mapping_vec(b);
        for (size_t i = 0; i < b; ++i) {
            slot_mapping_vec[i] = static_cast<int64_t>(i * cache_page_size);
        }
        infinicore::context::memcpyH2D(input.slot_mapping.value()->data(), slot_mapping_vec.data(), b * sizeof(int64_t), false);

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
        (void)model_->forward(input);
        infinicore::context::syncStream();

        GraphRecordingGuard recording;
        auto output = model_->forward(input);
        auto graph = recording.finish();
        barrier_->wait();

        auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

        compiled_map_[std::make_tuple(b, 1)] = CompiledResult{
            std::move(input), std::make_tuple(graph, shared_output), cache_page_size};
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

            ASSERT(input.past_sequence_lengths.value()->device().type() == infinicore::Device::Type::kCpu);
            ASSERT(input.past_sequence_lengths.value()->dtype() == infinicore::DataType::kInt32);
            ASSERT(input.past_sequence_lengths.value()->numel() == batch_size);
            const auto *past_sequence_lengths = reinterpret_cast<const int32_t *>(
                input.past_sequence_lengths.value()->data());
            std::vector<int64_t> slot_mapping(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                ASSERT(past_sequence_lengths[i] >= 0);
                ASSERT(static_cast<size_t>(past_sequence_lengths[i]) < result->second.cache_page_size);
                slot_mapping[i] = static_cast<int64_t>(
                    i * result->second.cache_page_size + static_cast<size_t>(past_sequence_lengths[i]));
            }
            infinicore::context::memcpyH2D(
                graph_input.slot_mapping.value()->data(),
                slot_mapping.data(),
                batch_size * sizeof(int64_t),
                false);

            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});
            return std::make_tuple(graph, shared_output);
        }
    } else {
        return std::make_tuple(nullptr, nullptr);
    }
}
} // namespace infinilm::engine
