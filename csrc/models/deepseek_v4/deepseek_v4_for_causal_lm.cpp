#include "deepseek_v4_for_causal_lm.hpp"
#include "deepseek_v4_cache.hpp"

#include "../../layers/quantization/none_quantization.hpp"
#include "../../models/models_registry.hpp"
#include "../../execution/submit.hpp"
#include "../../execution/paged_cache_checks.hpp"
#include "../../engine/distributed/communication_group.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
bool v4_fused_w13_enabled() {
    const char *disabled = std::getenv("INFINILM_DISABLE_FUSED_W13");
    return !(disabled && disabled[0] == '1' && disabled[1] == '\0');
}

infinilm::runtime::Tensor v4_cat_alloc(infinilm::runtime::Runtime &runtime,
                                          const std::vector<infinilm::runtime::Tensor> &inputs,
                                          size_t axis, const infinilm::runtime::Shape &shape) {
    auto output = infinilm::runtime::Tensor::empty(runtime, shape, inputs.front()->dtype());
    if (output->numel()) {
        infinilm::execution::submit<infini::ops::Cat>(
            runtime, infini::ops::Cat::DefaultConfig(runtime.device().type()),
            inputs, static_cast<int64_t>(axis), output);
    }
    return output;
}

infinilm::runtime::Tensor v4_broadcast_materialized(infinilm::runtime::Runtime &runtime,
                                                    const infinilm::runtime::Tensor &input,
                                                    const std::vector<int64_t> &shape) {
    infinilm::runtime::Shape unsigned_shape;
    unsigned_shape.reserve(shape.size());
    for (auto dim : shape) {
        if (dim < 0) throw std::invalid_argument("V4 broadcast shape must be nonnegative");
        unsigned_shape.push_back(static_cast<size_t>(dim));
    }
    auto output = infinilm::runtime::Tensor::empty(runtime, unsigned_shape, input->dtype());
    if (output->numel()) {
        infinilm::execution::submit<infini::ops::Copy>(
            runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()), input, false, output);
    }
    return output;
}
} // namespace
#include "../../layers/mlp/fused_activation.hpp"
#include <infini/ops.h>
#include "../../global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include "../../layers/linear/dense.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../layers/rotary_embedding/rotary_embedding_factory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {
using NoneQuant = infinilm::quantization::NoneQuantization;

float v4_bfloat16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}


void v4_debug_stats(infinilm::runtime::Runtime &runtime,
                    const infinilm::runtime::Tensor &tensor,
                    const std::string &label) {
    if (std::getenv("INFINILM_V4_DEBUG_STATS") == nullptr ||
        runtime.device().index() != 0 || !tensor) {
        return;
    }
    auto host = tensor->contiguous(runtime)->to(
        runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    const size_t count = host->numel();
    double sum = 0.0;
    double square_sum = 0.0;
    double max_abs = 0.0;
    std::array<float, 8> first{};
    std::array<float, 8> last_token_first{};
    const size_t hidden = host->shape().back();
    const size_t streams = host->ndim() >= 2 && host->size(host->ndim() - 2) == 4 ? 4 : 1;
    const size_t last_token_offset = count >= streams * hidden
        ? count - streams * hidden : 0;
    const auto accumulate = [&](size_t index, float value) {
        sum += value;
        square_sum += static_cast<double>(value) * value;
        max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
        if (index < first.size()) first[index] = value;
        if (index >= last_token_offset &&
            index < last_token_offset + last_token_first.size()) {
            last_token_first[index - last_token_offset] = value;
        }
    };
    if (host->dtype() == infinilm::runtime::DataType::kBFloat16) {
        const auto *data = reinterpret_cast<const uint16_t *>(host->data());
        for (size_t i = 0; i < count; ++i) accumulate(i, v4_bfloat16_to_float(data[i]));
    } else if (host->dtype() == infinilm::runtime::DataType::kFloat32) {
        const auto *data = reinterpret_cast<const float *>(host->data());
        for (size_t i = 0; i < count; ++i) accumulate(i, data[i]);
    } else {
        std::cout << "V4STAT " << label << " unsupported_dtype="
                  << static_cast<int>(host->dtype()) << std::endl;
        return;
    }
    std::cout << std::setprecision(10) << "V4STAT " << label
              << " count=" << count << " sum=" << sum
              << " rms=" << std::sqrt(square_sum / std::max<size_t>(count, 1))
              << " maxabs=" << max_abs << " first=";
    for (size_t i = 0; i < first.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << first[i];
    }
    std::cout << " last_token_first=";
    for (size_t i = 0; i < last_token_first.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << last_token_first[i];
    }
    std::cout << std::endl;
}

void v4_debug_indices(infinilm::runtime::Runtime &runtime,
                      const infinilm::runtime::Tensor &tensor,
                      const std::string &label) {
    if (std::getenv("INFINILM_V4_DEBUG_STATS") == nullptr ||
        runtime.device().index() != 0 || !tensor) {
        return;
    }
    auto host = tensor->contiguous(runtime)->to(
        runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    std::cout << "V4META " << label << " shape=";
    for (size_t i = 0; i < host->shape().size(); ++i) {
        if (i) std::cout << "x";
        std::cout << host->shape()[i];
    }
    std::cout << " values=";
    const size_t count = std::min<size_t>(host->numel(), 32);
    if (host->dtype() == infinilm::runtime::DataType::kInt32) {
        const auto *data = reinterpret_cast<const int32_t *>(host->data());
        for (size_t i = 0; i < count; ++i) {
            if (i) std::cout << ",";
            std::cout << data[i];
        }
    } else if (host->dtype() == infinilm::runtime::DataType::kInt64) {
        const auto *data = reinterpret_cast<const int64_t *>(host->data());
        for (size_t i = 0; i < count; ++i) {
            if (i) std::cout << ",";
            std::cout << data[i];
        }
    } else {
        std::cout << "unsupported_dtype=" << static_cast<int>(host->dtype());
    }
    std::cout << std::endl;
}

int64_t v4_debug_first_index(infinilm::runtime::Runtime &runtime,
                             const infinilm::runtime::Tensor &tensor) {
    auto host = tensor->contiguous(runtime)->to(
        runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    if (host->numel() == 0) return -1;
    if (host->dtype() == infinilm::runtime::DataType::kInt32) {
        return reinterpret_cast<const int32_t *>(host->data())[0];
    }
    if (host->dtype() == infinilm::runtime::DataType::kInt64) {
        return reinterpret_cast<const int64_t *>(host->data())[0];
    }
    return -1;
}

void v4_debug_topk(infinilm::runtime::Runtime &runtime,
                   const infinilm::runtime::Tensor &logits,
                   const std::string &label) {
    if (std::getenv("INFINILM_V4_DEBUG_STATS") == nullptr ||
        runtime.device().index() != 0 || !logits || logits->numel() == 0) {
        return;
    }
    auto host = logits->contiguous(runtime)->to(
        runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    const size_t width = host->shape().back();
    const size_t offset = host->numel() - width;
    std::array<std::pair<float, size_t>, 10> top;
    for (auto &item : top) item = {-std::numeric_limits<float>::infinity(), 0};
    for (size_t i = 0; i < width; ++i) {
        float value;
        if (host->dtype() == infinilm::runtime::DataType::kBFloat16) {
            value = v4_bfloat16_to_float(
                reinterpret_cast<const uint16_t *>(host->data())[offset + i]);
        } else if (host->dtype() == infinilm::runtime::DataType::kFloat32) {
            value = reinterpret_cast<const float *>(host->data())[offset + i];
        } else {
            return;
        }
        if (value <= top.back().first) continue;
        size_t position = top.size() - 1;
        while (position > 0 && value > top[position - 1].first) {
            top[position] = top[position - 1];
            --position;
        }
        top[position] = {value, i};
    }
    std::cout << "V4TOPK " << label << " ids=";
    for (size_t i = 0; i < top.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << top[i].second;
    }
    std::cout << " values=";
    for (size_t i = 0; i < top.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << top[i].first;
    }
    std::cout << std::endl;
}

std::shared_ptr<infinilm::quantization::BaseQuantization> dense_quant() {
    return std::make_shared<NoneQuant>(nullptr);
}

std::shared_ptr<infinilm::quantization::BaseQuantization> w8a8_quant(
    const std::shared_ptr<infinilm::config::ModelConfig> &config) {
    return config->get_quantization_method();
}

struct QliRowSchedule {
    int64_t cost{0};
    int64_t last_block_cost{0};
};

int64_t qli_block_cost(size_t query_rows, size_t key_rows) {
    const size_t aligned_key_rows = key_rows <= 1024
        ? (1536 + key_rows) / 2
        : (2560 + key_rows) / 2;
    return static_cast<int64_t>(query_rows * aligned_key_rows);
}

std::array<int32_t, 1024> make_qli_metadata_host(
    size_t query_length, size_t key_length, size_t requests = 1) {
    constexpr size_t kGroupSize = 64;
    constexpr size_t kQueryBlock = 256;
    constexpr size_t kKeyBlock = 2048;
    constexpr size_t kCompressRatio = 4;
    constexpr size_t kAicCoreCount = 20;  // Ascend 910C
    constexpr size_t kLiMetadataSize = 8;

    if (query_length == 0 || key_length == 0) {
        throw std::invalid_argument("DeepseekV4 QLI metadata requires nonzero lengths");
    }
    // Packed decode: one query token per request. The vendor metadata kernel
    // emits a single enabled core with bn2_end equal to the request count;
    // per-request key ranges come from the length tensors, not the schedule.
    if (query_length == requests) {
        std::array<int32_t, 1024> decode_metadata{};
        decode_metadata[0] = 1;
        decode_metadata[4] = static_cast<int32_t>(requests);
        return decode_metadata;
    }

    std::array<int32_t, 1024> metadata{};
    const size_t query_blocks =
        (query_length * kGroupSize + kQueryBlock - 1) / kQueryBlock;
    std::vector<QliRowSchedule> rows;
    rows.reserve(query_blocks);
    int64_t total_cost = 0;
    int64_t max_row_cost = 0;
    int64_t last_block_cost = 0;

    for (size_t row = 0; row < query_blocks; ++row) {
        const size_t query_tail = (query_length * kGroupSize) % kQueryBlock;
        const size_t query_width = row + 1 == query_blocks && query_tail != 0
            ? query_tail
            : kQueryBlock;
        const size_t first_query_token = row * kQueryBlock / kGroupSize;
        const size_t last_query_token =
            (std::min((row + 1) * kQueryBlock, query_length * kGroupSize) - 1)
            / kGroupSize;
        const int64_t causal_key_end = std::clamp<int64_t>(
            static_cast<int64_t>(last_query_token)
                + static_cast<int64_t>(key_length)
                - static_cast<int64_t>(query_length),
            0, static_cast<int64_t>(key_length - 1));
        (void)first_query_token;
        const size_t compressed_key_length =
            static_cast<size_t>(causal_key_end + 1) / kCompressRatio;
        const size_t key_blocks = compressed_key_length == 0
            ? 0
            : (compressed_key_length + kKeyBlock - 1) / kKeyBlock;
        const size_t key_tail = compressed_key_length % kKeyBlock;
        const size_t normal_blocks = key_blocks - (key_tail != 0 ? 1 : 0);
        const size_t query_rows = query_width / kGroupSize;
        const int64_t normal_cost = qli_block_cost(query_rows, kKeyBlock);
        const int64_t tail_cost = key_tail == 0
            ? 0
            : qli_block_cost(query_rows, key_tail);
        const int64_t row_cost =
            static_cast<int64_t>(normal_blocks) * normal_cost + tail_cost;
        const int64_t row_last_cost = key_blocks == 0
            ? 0
            : (key_tail == 0 ? normal_cost : tail_cost);
        rows.push_back({row_cost, row_last_cost});
        total_cost += row_cost;
        max_row_cost = std::max(max_row_cost, row_cost);
        if (key_blocks != 0) {
            last_block_cost = row_last_cost;
        }
    }

    // The vendor metadata kernel emits one enabled empty core for an all-empty
    // compressed causal range. QLI uses this to advance the single batch.
    if (total_cost == 0) {
        metadata[0] = 1;
        metadata[4] = 1;
        return metadata;
    }

    size_t row_cursor = 0;
    int64_t remaining_cost = total_cost;
    int32_t previous_batch_end = 0;
    int32_t previous_row_end = 0;
    for (size_t core = 0; core < kAicCoreCount && remaining_cost > 0; ++core) {
        const int64_t cost_limit = std::max(
            remaining_cost / static_cast<int64_t>(kAicCoreCount - core),
            max_row_cost);
        int64_t core_cost = 0;
        int64_t batch_remaining_cost = 0;
        for (size_t row = row_cursor; row < rows.size(); ++row) {
            batch_remaining_cost += rows[row].cost;
        }

        int32_t batch_end = 0;
        int32_t row_end = 0;
        if (cost_limit + last_block_cost / 2 >= batch_remaining_cost) {
            core_cost = batch_remaining_cost;
            row_cursor = rows.size();
            batch_end = 1;
        } else {
            while (row_cursor < rows.size()
                   && cost_limit + rows[row_cursor].last_block_cost / 2
                          >= core_cost + rows[row_cursor].cost) {
                core_cost += rows[row_cursor].cost;
                ++row_cursor;
            }
            if (row_cursor == rows.size()) {
                batch_end = 1;
            } else {
                row_end = static_cast<int32_t>(row_cursor);
            }
        }
        if (core_cost <= 0) {
            throw std::logic_error("DeepseekV4 QLI host scheduler made no progress");
        }
        remaining_cost -= core_cost;

        const size_t offset = core * kLiMetadataSize;
        metadata[offset] = 1;
        metadata[offset + 1] = previous_batch_end;
        metadata[offset + 2] = previous_row_end;
        metadata[offset + 4] = batch_end;
        metadata[offset + 5] = row_end;
        previous_batch_end = batch_end;
        previous_row_end = row_end;
    }
    if (remaining_cost != 0) {
        throw std::logic_error("DeepseekV4 QLI metadata exceeds Ascend 910C core capacity");
    }
    metadata[19 * kLiMetadataSize] = static_cast<int32_t>(requests);
    return metadata;
}

static std::mutex qli_h2d_mutex;

void refresh_qli_metadata(infinilm::runtime::Runtime &runtime,
                          infinilm::runtime::Tensor destination,
                          size_t query_length,
                          size_t key_length,
                          size_t requests = 1) {
    auto host_storage = std::make_shared<std::array<int32_t, 1024>>(
        make_qli_metadata_host(query_length, key_length, requests));
    auto host_tensor = infinilm::runtime::Tensor::from_blob(
        host_storage->data(), {host_storage->size()},
        infinilm::runtime::DataType::kInt32,
        infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu},
        host_storage);
    // Enqueue without syncStream: concurrent syncStream calls on different
    // rank threads trigger ACL 507015. Stream ordering ensures the copy
    // completes before subsequent kernels read the metadata.
    // Async H2D: concurrent synchronous copies from multiple rank workers
    // trigger ACL 507015. Stream ordering ensures correctness without sync.
    runtime.memcpyH2D(
        destination->data(), host_tensor->data(),
        destination->nbytes(), /*async=*/true);
}

infinilm::runtime::Tensor make_qli_metadata(
    infinilm::runtime::Runtime &runtime,
    size_t query_length,
    size_t key_length,
    size_t requests = 1) {
    auto device_tensor = infinilm::runtime::Tensor::empty(
        runtime, {size_t{1024}}, infinilm::runtime::DataType::kInt32);
    refresh_qli_metadata(
        runtime, device_tensor, query_length, key_length, requests);
    return device_tensor;
}


void prepare_v4_qli_metadata_before_kernels(
    const infinilm::InfinilmModel::Input &input,
    infinilm::global_state::ForwardContext &forward_context,
    const std::shared_ptr<infinilm::config::ModelConfig> &config) {
    auto &metadata = forward_context.attn_metadata;
    if (!input.input_ids || !input.input_ids.value()
        || !metadata.input_offsets || !metadata.input_offsets.value()
        || !metadata.total_sequence_lengths
        || !metadata.total_sequence_lengths.value()) {
        return;
    }
    auto &runtime = forward_context.require_runtime();
    const size_t query_rows = input.input_ids.value()->numel();
    const size_t requests = metadata.total_sequence_lengths.value()->size(0);
    if (query_rows == 0 || query_rows == requests) {
        return;  // Decode schedules were pre-uploaded with the cache.
    }

    auto offsets_cpu = metadata.input_offsets.value()->to(
        runtime, infinilm::runtime::Device{
                     infinilm::runtime::Device::Type::kCpu});
    const auto *offsets = reinterpret_cast<const int32_t *>(offsets_cpu->data());
    auto indices_storage = std::make_shared<std::vector<int64_t>>();
    indices_storage->reserve(requests);
    for (size_t row = 0; row < requests; ++row) {
        indices_storage->push_back(
            static_cast<int64_t>(offsets[row + 1] - 1));
    }
    auto indices_device = infinilm::runtime::Tensor::empty(
        runtime, {requests}, infinilm::runtime::DataType::kInt64);
    runtime.memcpyH2D(
        indices_device->data(), indices_storage->data(),
        requests * sizeof(int64_t), /*async=*/true);
    metadata.last_token_ids = std::move(indices_device);

    const size_t max_query = metadata.max_query_length != 0
        ? metadata.max_query_length
        : (query_rows + requests - 1) / requests;
    const size_t max_kv = metadata.max_sequence_length != 0
        ? metadata.max_sequence_length : max_query;
    auto &states = forward_context.deepseek_v4_qli_metadata;
    auto state = std::make_shared<
        infinilm::global_state::DeepseekV4QliMetadataState>();
    // Vendor metadata for a packed B x q prefill matches the single-sequence
    // q-row schedule, with its final batch field replaced by B.
    state->tensor = make_qli_metadata(
        runtime, max_query, max_kv, requests);
    state->query_length = query_rows;
    state->last_key_length = metadata.first_total_sequence_length.value_or(
        max_kv);
    state->pre_uploaded = true;
    states.insert_or_assign(query_rows, std::move(state));
}

std::pair<infinilm::runtime::Tensor, infinilm::runtime::Tensor>
rms_norm_dynamic_quant(infinilm::runtime::Runtime &runtime,
                       const infinilm::runtime::Tensor &input,
                       const infinilm::runtime::Tensor &gamma,
                       double epsilon) {
    if (!input || input->ndim() == 0 || !gamma ||
        gamma->shape() != infinilm::runtime::Shape{input->shape().back()}) {
        throw std::invalid_argument(
            "DeepSeek V4 fused RMSNorm quantization geometry is invalid");
    }
    size_t rows = 1;
    for (size_t dim = 0; dim + 1 < input->ndim(); ++dim) {
        rows *= input->shape()[dim];
    }
    const size_t columns = input->shape().back();
    auto flat = input->view({rows, columns});
    auto packed = infinilm::runtime::Tensor::empty(
        runtime, flat->shape(), infinilm::runtime::DataType::kInt8);
    auto packed2 = infinilm::runtime::Tensor::empty(
        runtime, {1}, infinilm::runtime::DataType::kInt8);
    auto scale = infinilm::runtime::Tensor::empty(
        runtime, {rows}, infinilm::runtime::DataType::kFloat32);
    auto scale2 = infinilm::runtime::Tensor::empty(
        runtime, {rows}, infinilm::runtime::DataType::kFloat32);
    infinilm::execution::submit<infini::ops::RmsNormDynamicQuant>(
        runtime,
        infini::ops::RmsNormDynamicQuant::DefaultConfig(runtime.device().type()),
        packed, packed2, scale, scale2, flat, gamma, epsilon);
    return {packed, scale->view({rows, 1})};
}

std::shared_ptr<infinilm::layers::RoPE> make_deepseek_v4_rope(
    infinilm::runtime::Runtime &runtime,
    const std::shared_ptr<infinilm::config::ModelConfig> &config,
    bool compressed) {
    const size_t head_dim = config->get<size_t>("head_dim");
    const size_t rotary_dim = config->get_or<size_t>("qk_rope_head_dim", 64);
    size_t max_pos = config->get_or<size_t>("max_position_embeddings", 4096);
    if (const char *override_value = std::getenv("INFINILM_V4_ROPE_CACHE_LEN")) {
        const auto requested = std::stoull(override_value);
        if (requested == 0 || requested > max_pos) {
            throw std::invalid_argument("INFINILM_V4_ROPE_CACHE_LEN is out of range");
        }
        max_pos = static_cast<size_t>(requested);
    }
    const double theta = compressed
        ? config->get_or<double>("compress_rope_theta", 160000.0)
        : config->get_or<double>("rope_theta", 10000.0);
    auto scaling = infinilm::layers::rotary_embedding::make_scaling_config(config);
    return std::make_shared<infinilm::layers::RoPE>(
        runtime, head_dim, rotary_dim, max_pos, theta,
        infinilm::layers::RoPE::Algo::GPT_J, config->get_dtype(),
        runtime.device(), std::move(scaling), std::nullopt, false, true);
}

void apply_v4_partial_rope(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    const std::shared_ptr<infinilm::layers::RoPE> &rope,
    infinilm::global_state::ForwardContext &forward_context,
    size_t start_position,
    const infinilm::runtime::Tensor &positions,
    bool inverse) {
    if (!x || x->ndim() != 3 || !rope) {
        throw std::invalid_argument(
            "DeepseekV4 partial RoPE expects [tokens, heads, head_dim]");
    }
    const size_t tokens = x->size(0);
    const size_t head_dim = x->size(2);
    const size_t rotary_dim = rope->rotary_dim();
    if (rope->interleaved_cos_cache()->size(0) < tokens
        || rotary_dim > head_dim
        || start_position + tokens
               > rope->interleaved_cos_cache()->size(0)) {
        throw std::invalid_argument(
            "DeepseekV4 partial RoPE cache range is invalid");
    }
    // Graph replay cannot rebind the address of a cache narrow view, so route
    // RoPE tables through stable holders refreshed at the current position.
    auto &states = forward_context.deepseek_v4_rope;
    auto [state_it, inserted] = states.try_emplace(
        std::make_pair(reinterpret_cast<size_t>(rope.get()), tokens),
        std::make_shared<infinilm::global_state::DeepseekV4RopeState>());
    auto state = state_it->second;
    if (inserted || !state->cos || !state->sin || !state->inverse_sin) {
        state->cos = infinilm::runtime::Tensor::empty(
            runtime, {tokens, 1, 1, rotary_dim}, rope->dtype());
        state->sin = infinilm::runtime::Tensor::empty(
            runtime, state->cos->shape(), rope->dtype());
        state->inverse_sin = infinilm::runtime::Tensor::empty(
            runtime, state->cos->shape(), rope->dtype());
        state->refresh =
            [runtime_ptr = &runtime,
             rope,
             cos = state->cos,
             sin = state->sin,
             inverse_sin = state->inverse_sin,
             tokens](size_t start) mutable {
                const auto &cos_cache = rope->interleaved_cos_cache();
                const auto &sin_cache = rope->interleaved_sin_cache();
                const auto &inverse_sin_cache =
                    rope->interleaved_inverse_sin_cache();
                if (!cos_cache || !sin_cache || !inverse_sin_cache
                    || start + tokens > cos_cache->size(0)) {
                    throw std::invalid_argument(
                        "DeepseekV4 RoPE replay cache range is invalid");
                }
                const size_t rotary_dim = rope->rotary_dim();
                if (std::getenv("INFINILM_LOG_ROPE") != nullptr) {
                    std::fprintf(stderr,
                                 "V4_ROPE_REFRESH rows=%zu start=%zu\n",
                                 tokens, start);
                }
                cos->copy_from(
                    *runtime_ptr,
                    cos_cache->narrow({{0, start, tokens}})
                        ->view({{tokens, 1, 1, rotary_dim}}));
                sin->copy_from(
                    *runtime_ptr,
                    sin_cache->narrow({{0, start, tokens}})
                        ->view({{tokens, 1, 1, rotary_dim}}));
                inverse_sin->copy_from(
                    *runtime_ptr,
                    inverse_sin_cache->narrow({{0, start, tokens}})
                        ->view({{tokens, 1, 1, rotary_dim}}));
            };
    }
    const bool gather_positions = positions
        && positions->numel() == tokens && tokens > 1
        && positions->dtype() == infinilm::runtime::DataType::kInt64
        && std::getenv("INFINILM_DISABLE_ROPE_GATHER") == nullptr;
    state->device_gathered = gather_positions;
    if (gather_positions) {
        // Batch decode places each row at an independent sequence position;
        // gather once per rope type and reuse the rows across all layers.
        if (!state->gathered_this_forward) {
        infinilm::execution::submit<infini::ops::Embedding>(
            runtime, infini::ops::Embedding::DefaultConfig(runtime.device().type()),
            positions, rope->interleaved_cos_cache(),
            state->cos->view({tokens, rotary_dim}));
        infinilm::execution::submit<infini::ops::Embedding>(
            runtime, infini::ops::Embedding::DefaultConfig(runtime.device().type()),
            positions, rope->interleaved_sin_cache(),
            state->sin->view({tokens, rotary_dim}));
        infinilm::execution::submit<infini::ops::Embedding>(
            runtime, infini::ops::Embedding::DefaultConfig(runtime.device().type()),
            positions, rope->interleaved_inverse_sin_cache(),
            state->inverse_sin->view({tokens, rotary_dim}));
        state->gathered_this_forward = true;
        }
    } else if (runtime.graphCaptureState()
               == infinilm::execution::graph::GraphManager::CaptureState::
                   kInactive) {
        // EAGER: use cache views directly. Zero-copy, zero-H2D.
        // Concurrent H2D from multiple rank workers triggers ACL 507015.
        const auto &cache = inverse
            ? rope->interleaved_inverse_sin_cache()
            : rope->interleaved_sin_cache();
        auto eager_cos = rope->interleaved_cos_cache()
            ->narrow({{0, start_position, tokens}})
            ->view({tokens, 1, 1, rotary_dim});
        auto eager_sin = cache
            ->narrow({{0, start_position, tokens}})
            ->view({tokens, 1, 1, rotary_dim});
        auto eager_x4 = x->view({tokens, 1, x->size(1), head_dim});
        infinilm::execution::submit<infini::ops::PartialRotaryMul>(
            runtime, infini::ops::PartialRotaryMul::DefaultConfig(
                         runtime.device().type()),
            eager_x4, eager_cos, eager_sin, int64_t{1},
            static_cast<int64_t>(head_dim - rotary_dim),
            static_cast<int64_t>(head_dim));
        return;
    }
    const auto &cos = state->cos;
    const auto &sin = inverse ? state->inverse_sin : state->sin;
    auto x4 = x->view({tokens, 1, x->size(1), head_dim});
    infinilm::execution::submit<infini::ops::PartialRotaryMul>(
        runtime, infini::ops::PartialRotaryMul::DefaultConfig(
                     runtime.device().type()),
        x4, cos, sin, int64_t{1},
        static_cast<int64_t>(head_dim - rotary_dim),
        static_cast<int64_t>(head_dim));
}

void update_shared_kv_cache(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &kv,
    infinilm::runtime::Tensor &cache,
    const infinilm::runtime::Tensor &slot_mapping) {
    if (!cache || cache->ndim() != 5 || cache->size(0) != 2) {
        throw std::invalid_argument("DeepseekV4 sliding-window cache is invalid");
    }
    auto k_cache = cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache = cache->narrow({{0, 1, 1}})->squeeze(0);
    auto config = infinilm::execution::paged_cache_checks::validate(
        runtime, k_cache, v_cache, kv, kv, slot_mapping);
    if (!config) {
        return;
    }
    const infini::ops::Tensor scale{
        nullptr, {1}, infinilm::runtime::DataType::kFloat32, runtime.device()};
    infinilm::execution::submit<infini::ops::ReshapeAndCacheFlash>(
        runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, *config,
        kv, kv, slot_mapping, scale, scale, std::string{"auto"},
        k_cache, v_cache);
    INFINILM_TRACE_OP(PagedCaching, k_cache);
}

infinilm::runtime::Tensor sparse_shared_kv_attention(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &query,
    const infinilm::runtime::Tensor &kv,
    size_t layer_idx,
    size_t local_heads,
    size_t head_dim,
    const infinilm::runtime::Tensor &sinks,
    infinilm::global_state::ForwardContext &forward_context) {
    auto &metadata = forward_context.attn_metadata;
    if (!metadata.input_offsets || !metadata.cu_seqlens
        || !metadata.total_sequence_lengths || !metadata.block_tables
        || !metadata.slot_mapping) {
        throw std::invalid_argument(
            "DeepseekV4 sparse attention requires paged sequence metadata");
    }
    const size_t index = cache_index(layer_idx, CacheSlot::SlidingWindow);
    if (index >= forward_context.kv_cache_vec.size()
        || !forward_context.kv_cache_vec[index]) {
        throw std::invalid_argument("DeepseekV4 sliding-window cache is missing");
    }
    auto &cache = forward_context.kv_cache_vec[index];
    update_shared_kv_cache(runtime, kv, cache, *metadata.slot_mapping);
    auto ori_kv = cache->narrow({{0, 0, 1}})->squeeze(0);
    const size_t past = metadata.first_past_sequence_length.value_or(0);
    if (layer_idx == 0 && std::getenv("INFINILM_V4_DEBUG_STATS") != nullptr) {
        const std::string suffix = ".past" + std::to_string(past);
        v4_debug_indices(runtime, *metadata.input_offsets,
                         "layer0.input_offsets" + suffix);
        v4_debug_indices(runtime, *metadata.total_sequence_lengths,
                         "layer0.total_sequence_lengths" + suffix);
        v4_debug_indices(runtime, *metadata.block_tables,
                         "layer0.block_tables" + suffix);
        v4_debug_indices(runtime, *metadata.slot_mapping,
                         "layer0.slot_mapping" + suffix);
        const int64_t block = v4_debug_first_index(runtime, *metadata.block_tables);
        const int64_t total = v4_debug_first_index(
            runtime, *metadata.total_sequence_lengths);
        if (block >= 0 && static_cast<size_t>(block) < ori_kv->size(0) && total > 0) {
            const size_t rows = std::min<size_t>(
                static_cast<size_t>(total), ori_kv->size(1));
            auto active = ori_kv->narrow({
                {0, static_cast<size_t>(block), 1}, {1, 0, rows}})
                ->view({rows, ori_kv->size(3)});
            v4_debug_stats(runtime, active, "layer0.k_cache_active" + suffix);
        }
    }
    const size_t block_size = ori_kv->size(1);
    const size_t max_query = metadata.max_query_length != 0
        ? metadata.max_query_length : query->size(0);
    const size_t table_capacity = metadata.block_tables.value()->size(1) * block_size;
    const size_t max_kv = metadata.max_sequence_length != 0
        ? metadata.max_sequence_length : table_capacity;

    auto empty_i32 = infinilm::runtime::Tensor::empty(
        runtime, {0}, infinilm::runtime::DataType::kInt32);
    auto empty_f32 = infinilm::runtime::Tensor::empty(
        runtime, {0}, infinilm::runtime::DataType::kFloat32);
    auto empty_bf16 = infinilm::runtime::Tensor::empty(
        runtime, {0}, infinilm::runtime::DataType::kBFloat16);
    const size_t requests = metadata.total_sequence_lengths.value()->size(0);
    auto query_lengths = metadata.input_offsets.value()
        ->narrow({{0, 1, requests}})->contiguous(runtime);
    const bool packed_decode = requests > 1 && query->size(0) == requests;
    const bool prefill_cu_seqlens =
        requests > 1 ? !packed_decode : past == 0;
    auto cu_seqlens_ori_kv = prefill_cu_seqlens
        ? *metadata.input_offsets : empty_i32;
    // The vendor decode contract derives query lengths from cu_seqlens_q and
    // requires seqused_q to stay empty; non-empty seqused_q marks varlen
    // prefill and rejects batched decode metadata.
    const auto &seqused_q = static_cast<const infinilm::runtime::Tensor &>(empty_i32);
    auto &sas_states = forward_context.deepseek_v4_sas_metadata;
    auto [sas_state_it, sas_inserted] = sas_states.try_emplace(
        size_t{1},
        std::make_shared<
            infinilm::global_state::DeepseekV4SasMetadataState>());
    auto sas_state = sas_state_it->second;
    if (sas_inserted || !sas_state->tensor) {
        sas_state->tensor = infinilm::runtime::Tensor::empty(
            runtime, {1024}, infinilm::runtime::DataType::kInt32);
    }
    if (!sas_state->ready) {
    infinilm::execution::submit<infini::ops::SparseAttnSharedkvMetadata>(
        runtime, infini::ops::SparseAttnSharedkvMetadata::DefaultConfig(runtime.device().type()),
        sas_state->tensor, *metadata.input_offsets, cu_seqlens_ori_kv,
        empty_i32, seqused_q, *metadata.total_sequence_lengths,
        static_cast<int64_t>(local_heads), int64_t{1},
        static_cast<int64_t>(head_dim),
        static_cast<int64_t>(requests),
        static_cast<int64_t>(max_query), static_cast<int64_t>(max_kv),
        int64_t{0}, int64_t{0}, int64_t{1}, int64_t{4}, int64_t{3},
        int64_t{127}, int64_t{0}, std::string{"TND"},
        std::string{"PA_ND"}, true, false);
        sas_state->ready = true;
    }
    auto sas_metadata = sas_state->tensor;

    auto output = infinilm::runtime::Tensor::empty(
        runtime, query->shape(), query->dtype());
    infinilm::execution::submit<infini::ops::SparseAttnSharedkv>(
        runtime, infini::ops::SparseAttnSharedkv::DefaultConfig(runtime.device().type()),
        output, empty_f32, query, ori_kv, empty_bf16, empty_i32, empty_i32,
        *metadata.block_tables, empty_i32,
        *metadata.input_offsets, cu_seqlens_ori_kv, empty_i32, empty_i32,
        *metadata.total_sequence_lengths, sinks, sas_metadata,
        1.0 / std::sqrt(static_cast<double>(head_dim)), int64_t{1},
        int64_t{4}, int64_t{3}, static_cast<int64_t>(ori_kv->stride(0)),
        int64_t{0}, int64_t{127}, int64_t{0},
        std::string{"TND"}, std::string{"PA_ND"}, false);
    INFINILM_TRACE_OP(MultiheadAttentionVarlen, output);
    return output;
}

infinilm::runtime::Tensor compressed_sparse_shared_kv_attention(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &query,
    const infinilm::runtime::Tensor &kv,
    const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &indexer_query,
    const infinilm::runtime::Tensor &indexer_weights,
    size_t layer_idx,
    size_t local_heads,
    size_t head_dim,
    size_t compress_ratio,
    size_t indexer_heads,
    size_t indexer_head_dim,
    size_t index_topk,
    const infinilm::runtime::Tensor &sinks,
    const infinilm::runtime::Tensor &compressor_ape,
    const infinilm::runtime::Tensor &compressor_wkv,
    const infinilm::runtime::Tensor &compressor_wgate,
    const infinilm::runtime::Tensor &compressor_norm,
    const infinilm::runtime::Tensor &indexer_ape,
    const infinilm::runtime::Tensor &indexer_wkv,
    const infinilm::runtime::Tensor &indexer_wgate,
    const infinilm::runtime::Tensor &indexer_norm,
    const std::shared_ptr<infinilm::layers::RoPE> &rope,
    infinilm::global_state::ForwardContext &forward_context) {
    if (compress_ratio != 4 && compress_ratio != 128) {
        throw std::invalid_argument("DeepseekV4 compressed attention requires ratio 4 or 128");
    }
    auto &metadata = forward_context.attn_metadata;
    if (!metadata.input_offsets || !metadata.past_sequence_lengths
        || !metadata.total_sequence_lengths || !metadata.block_tables
        || !metadata.slot_mapping) {
        throw std::invalid_argument(
            "DeepseekV4 compressed attention requires paged sequence metadata");
    }
    const size_t requests = metadata.total_sequence_lengths.value()->size(0);
    const auto &compressed_block_table =
        compress_ratio == 4
            ? (metadata.ratio4_block_tables
                   ? *metadata.ratio4_block_tables
                   : metadata.block_tables.value())
            : (metadata.ratio128_block_tables
                   ? *metadata.ratio128_block_tables
                   : metadata.block_tables.value());
    const auto &state_block_table =
        compress_ratio == 4
            ? (metadata.state4_block_tables
                   ? *metadata.state4_block_tables
                   : metadata.block_tables.value())
            : (metadata.state128_block_tables
                   ? *metadata.state128_block_tables
                   : metadata.block_tables.value());

    auto &ori_cache_storage = forward_context.kv_cache_vec.at(
        cache_index(layer_idx, CacheSlot::SlidingWindow));
    auto &cmp_cache_storage = forward_context.kv_cache_vec.at(
        cache_index(layer_idx, CacheSlot::CompressedKv));
    auto &state_cache = forward_context.kv_cache_vec.at(
        cache_index(layer_idx, CacheSlot::CompressorState));
    if (!ori_cache_storage || !cmp_cache_storage || !state_cache) {
        throw std::invalid_argument("DeepseekV4 compressed attention cache is missing");
    }
    update_shared_kv_cache(
        runtime, kv, ori_cache_storage, *metadata.slot_mapping);
    auto ori_kv = ori_cache_storage->narrow({{0, 0, 1}})->squeeze(0);
    auto cmp_kv = cmp_cache_storage->narrow({{0, 0, 1}})->squeeze(0);
    const size_t block_size = cmp_kv->size(1);
    const auto &exact_rows =
        compress_ratio == 4
            ? metadata.compressed_rows_ratio4
            : metadata.compressed_rows_ratio128;
    size_t rows;
    if (exact_rows.has_value()) {
        rows = exact_rows.value();
        if (rows == 0) {
            rows = 1;  // Minimal allocation for compressor state updates.
        }
    } else {
        rows = std::min(
            query->size(0), query->size(0) / compress_ratio + requests);
        if (rows == 0) {
            rows = 1;
        }
    }
    // The vendor metadata emits [-1,31] sentinel slots for every
    // over-allocated row. Exact sizing eliminates all sentinels; when the
    // exact count is zero there are no complete groups so skip the scatter.
    // The compressor still runs with 1 row to maintain streaming state.
    const bool skip_compressed_scatter =
        exact_rows.has_value() && exact_rows.value() == 0;
    if (!rope || !rope->interleaved_cos_cache()
        || !rope->interleaved_sin_cache()) {
        throw std::invalid_argument("DeepseekV4 compressor requires interleaved RoPE caches");
    }
    auto &compressor_states =
        forward_context.deepseek_v4_compressor_metadata;
    auto [compressor_state_it, compressor_inserted] =
        compressor_states.try_emplace(
            compress_ratio,
            std::make_shared<infinilm::global_state::
                                 DeepseekV4CompressorMetadataState>());
    auto compressor_state = compressor_state_it->second;
    const bool compressor_shape_valid =
        compressor_state->cos
        && compressor_state->cos->shape()
               == infinilm::runtime::Shape{
                   rows, 1, 1, rope->rotary_dim()}
        && compressor_state->sin
        && compressor_state->sin->shape() == compressor_state->cos->shape()
        && compressor_state->slots
        && compressor_state->slots->shape()
               == infinilm::runtime::Shape{rows, 2};
    if (compressor_inserted || !compressor_shape_valid) {
        compressor_state->cos = infinilm::runtime::Tensor::empty(
            runtime, {rows, 1, 1, rope->rotary_dim()}, rope->dtype());
        compressor_state->sin = infinilm::runtime::Tensor::empty(
            runtime, compressor_state->cos->shape(), rope->dtype());
        compressor_state->slots = infinilm::runtime::Tensor::empty(
            runtime, {rows, 2}, infinilm::runtime::DataType::kInt32);
    }
    const auto &compress_cos = compressor_state->cos;
    const auto &compress_sin = compressor_state->sin;
    const auto &compress_slots = compressor_state->slots;
    if (!skip_compressed_scatter && !compressor_state->ready) {
    infinilm::execution::submit<infini::ops::CompressorMetadata>(
        runtime, infini::ops::CompressorMetadata::DefaultConfig(runtime.device().type()),
        compress_cos, compress_sin, compress_slots,
        rope->interleaved_cos_cache(), rope->interleaved_sin_cache(),
        *metadata.input_offsets, *metadata.past_sequence_lengths,
        compressed_block_table, static_cast<int64_t>(block_size), int64_t{2},
        static_cast<int64_t>(compress_ratio), static_cast<int64_t>(requests));
        compressor_state->ready = true;
    }

    const size_t coff = compressor_wkv->size(0) / head_dim;
    if ((coff != 1 && coff != 2)
        || compressor_wkv->size(0) != compressor_wgate->size(0)) {
        throw std::invalid_argument("DeepseekV4 compressor projection shape is invalid");
    }
    auto compressed_kv = infinilm::runtime::Tensor::empty(
        runtime, {rows, head_dim}, query->dtype());
    auto flat_hidden = hidden_states->view(
        {hidden_states->numel() / hidden_states->shape().back(),
         hidden_states->shape().back()});
    if (!skip_compressed_scatter)
    infinilm::execution::submit<infini::ops::Compressor>(
        runtime, infini::ops::Compressor::DefaultConfig(runtime.device().type()),
        compressed_kv, state_cache, flat_hidden, compressor_wkv,
        compressor_wgate, compressor_ape, compressor_norm,
        compress_sin->view({rows, rope->rotary_dim()}),
        compress_cos->view({rows, rope->rotary_dim()}),
        state_block_table, *metadata.input_offsets,
        *metadata.past_sequence_lengths,
        static_cast<int64_t>(rope->rotary_dim()),
        static_cast<int64_t>(compress_ratio), static_cast<int64_t>(coff),
        1e-6, int64_t{2}, int64_t{1});
    if (!skip_compressed_scatter)
    infinilm::execution::submit<infini::ops::ScatterNdUpdateV2>(
        runtime, infini::ops::ScatterNdUpdateV2::DefaultConfig(runtime.device().type()),
        cmp_kv, compress_slots, compressed_kv);

    auto empty_i32 = infinilm::runtime::Tensor::empty(
        runtime, {0}, infinilm::runtime::DataType::kInt32);
    auto empty_f32 = infinilm::runtime::Tensor::empty(
        runtime, {0}, infinilm::runtime::DataType::kFloat32);
    auto query_lengths = metadata.input_offsets.value()
        ->narrow({{0, 1, requests}})->contiguous(runtime);
    const bool packed_decode = requests > 1 && query->size(0) == requests;
    const bool prefill_cu_seqlens = requests > 1
        ? !packed_decode
        : metadata.first_past_sequence_length.value_or(0) == 0;
    auto cu_seqlens_ori_kv = prefill_cu_seqlens
        ? *metadata.input_offsets : empty_i32;
    const auto &seqused_q = static_cast<const infinilm::runtime::Tensor &>(empty_i32);
    const size_t max_query = metadata.max_query_length != 0
        ? metadata.max_query_length : query->size(0);
    const size_t table_capacity = compressed_block_table->size(1) * block_size;
    const size_t max_kv = metadata.max_sequence_length != 0
        ? metadata.max_sequence_length : table_capacity;
    infinilm::runtime::Tensor compressed_sparse_indices = empty_i32;
    if (compress_ratio == 4) {
        auto &indexer_state_cache = forward_context.kv_cache_vec.at(
            cache_index(layer_idx, CacheSlot::IndexerState));
        auto &indexer_key_cache = forward_context.kv_cache_vec.at(
            cache_index(layer_idx, CacheSlot::IndexerKey));
        auto &indexer_scale_cache = forward_context.kv_cache_vec.at(
            cache_index(layer_idx, CacheSlot::IndexerScale));
        if (!indexer_query || !indexer_weights || !indexer_state_cache
            || !indexer_key_cache || !indexer_scale_cache
            || !indexer_ape || !indexer_wkv || !indexer_wgate || !indexer_norm) {
            throw std::invalid_argument("DeepseekV4 ratio-4 indexer state is incomplete");
        }
        if (indexer_query->shape() != infinilm::runtime::Shape{
                query->size(0), indexer_heads, indexer_head_dim}
            || indexer_weights->shape() != infinilm::runtime::Shape{
                query->size(0), indexer_heads}) {
            throw std::invalid_argument("DeepseekV4 indexer query shape is invalid");
        }
        const size_t indexer_coff = indexer_wkv->size(0) / indexer_head_dim;
        if ((indexer_coff != 1 && indexer_coff != 2)
            || indexer_wkv->size(0) != indexer_wgate->size(0)) {
            throw std::invalid_argument("DeepseekV4 indexer compressor shape is invalid");
        }
        auto indexer_kv = infinilm::runtime::Tensor::empty(
            runtime, {rows, indexer_head_dim}, query->dtype());
        if (!skip_compressed_scatter)
        infinilm::execution::submit<infini::ops::Compressor>(
            runtime, infini::ops::Compressor::DefaultConfig(runtime.device().type()),
            indexer_kv, indexer_state_cache, flat_hidden, indexer_wkv,
            indexer_wgate, indexer_ape, indexer_norm,
            compress_sin->view({rows, rope->rotary_dim()}),
            compress_cos->view({rows, rope->rotary_dim()}),
            state_block_table, *metadata.input_offsets,
            *metadata.past_sequence_lengths,
            static_cast<int64_t>(rope->rotary_dim()),
            static_cast<int64_t>(compress_ratio), static_cast<int64_t>(indexer_coff),
            1e-6, int64_t{2}, int64_t{1});

        auto indexer_query_flat = indexer_query->view(
            {query->size(0) * indexer_heads, indexer_head_dim});
        auto indexer_query_i8_flat = infinilm::runtime::Tensor::empty(
            runtime, indexer_query_flat->shape(), infinilm::runtime::DataType::kInt8);
        auto indexer_query_scale_f32 = infinilm::runtime::Tensor::empty(
            runtime, {query->size(0) * indexer_heads, 1},
            infinilm::runtime::DataType::kFloat32);
        infinilm::execution::submit<infini::ops::PerChannelQuantI8>(
            runtime, infini::ops::PerChannelQuantI8::DefaultConfig(runtime.device().type()),
            indexer_query_flat, indexer_query_i8_flat, indexer_query_scale_f32);
        auto indexer_query_scale_f16 = infinilm::runtime::Tensor::empty(
            runtime, indexer_query_scale_f32->shape(),
            infinilm::runtime::DataType::kFloat16);
        infinilm::execution::submit<infini::ops::Cast>(
            runtime, infini::ops::Cast::DefaultConfig(runtime.device().type()),
            indexer_query_scale_f32, indexer_query_scale_f16);

        auto indexer_key_i8 = infinilm::runtime::Tensor::empty(
            runtime, indexer_kv->shape(), infinilm::runtime::DataType::kInt8);
        auto indexer_key_scale_f32 = infinilm::runtime::Tensor::empty(
            runtime, {rows, 1}, infinilm::runtime::DataType::kFloat32);
        if (!skip_compressed_scatter)
        infinilm::execution::submit<infini::ops::PerChannelQuantI8>(
            runtime, infini::ops::PerChannelQuantI8::DefaultConfig(runtime.device().type()),
            indexer_kv, indexer_key_i8, indexer_key_scale_f32);
        auto indexer_key_scale_f16 = infinilm::runtime::Tensor::empty(
            runtime, indexer_key_scale_f32->shape(),
            infinilm::runtime::DataType::kFloat16);
        if (!skip_compressed_scatter)
        infinilm::execution::submit<infini::ops::Cast>(
            runtime, infini::ops::Cast::DefaultConfig(runtime.device().type()),
            indexer_key_scale_f32, indexer_key_scale_f16);
        if (!skip_compressed_scatter)
        infinilm::execution::submit<infini::ops::ScatterNdUpdateV2>(
            runtime, infini::ops::ScatterNdUpdateV2::DefaultConfig(runtime.device().type()),
            indexer_key_cache, compress_slots, indexer_key_i8);
        if (!skip_compressed_scatter)
        infinilm::execution::submit<infini::ops::ScatterNdUpdateV2>(
            runtime, infini::ops::ScatterNdUpdateV2::DefaultConfig(runtime.device().type()),
            indexer_scale_cache, compress_slots, indexer_key_scale_f16);

        auto indexer_weights_f16 = infinilm::runtime::Tensor::empty(
            runtime, indexer_weights->shape(), infinilm::runtime::DataType::kFloat16);
        infinilm::execution::submit<infini::ops::Cast>(
            runtime, infini::ops::Cast::DefaultConfig(runtime.device().type()),
            indexer_weights, indexer_weights_f16);
        infinilm::runtime::Tensor qli_metadata;
        if (requests == 1) {
            const size_t qli_key_length =
                metadata.first_total_sequence_length.value_or(max_kv);
            auto &qli_states = forward_context.deepseek_v4_qli_metadata;
            // All ratio-4 layers share identical QLI scheduling geometry. A
            // single stable buffer also lets graph replay refresh it once pe
            // decode step.
            // Key the shared schedule by query rows: each captured decode
            // batch size needs its own host-side scheduling geometry.
            auto [qli_state_it, inserted] = qli_states.try_emplace(
                query->size(0),
                std::make_shared<
                    infinilm::global_state::DeepseekV4QliMetadataState>());
            auto qli_state = qli_state_it->second;
            qli_state->query_length = query->size(0);
            const size_t scheduled_query_length =
                std::getenv("INFINILM_QLI_QUERY1") != nullptr
                    ? 1
                    : qli_state->query_length;
            if (inserted) {
                qli_state->tensor = make_qli_metadata(
                    runtime, scheduled_query_length, qli_key_length,
                    requests);
                qli_state->refresh =
                    [runtime_ptr = &runtime,
                     tensor = qli_state->tensor,
                     query_length = scheduled_query_length,
                     requests](
                        size_t key_length) {
                        refresh_qli_metadata(
                            *runtime_ptr, tensor, query_length, key_length,
                            requests);
                    };
            } else if (!qli_state->pre_uploaded
                       && qli_state->last_key_length != qli_key_length
                       && runtime.graphCaptureState()
                              == infinilm::execution::graph::GraphManager::
                                  CaptureState::kInactive) {
                // Only refresh when geometry actually changes. Repeated H2D
                // copies from concurrent rank workers trigger ACL 507015.
                qli_state->refresh(qli_key_length);
                qli_state->last_key_length = qli_key_length;
            }
            qli_metadata = qli_state->tensor;
        } else if (std::getenv("INFINILM_ENABLE_VENDOR_QLI") == nullptr) {
            // Batched prefill uses the host schedule; decode took the branch above.
            const size_t qli_key_length =
                metadata.first_total_sequence_length.value_or(max_kv);
            auto &qli_states = forward_context.deepseek_v4_qli_metadata;
            // Key the shared schedule by query rows: each captured decode
            // batch size needs its own host-side scheduling geometry.
            auto [qli_state_it, inserted] = qli_states.try_emplace(
                query->size(0),
                std::make_shared<
                    infinilm::global_state::DeepseekV4QliMetadataState>());
            auto qli_state = qli_state_it->second;
            qli_state->query_length = query->size(0);
            const size_t scheduled_query_length =
                std::getenv("INFINILM_QLI_QUERY1") != nullptr
                    ? 1
                    : qli_state->query_length;
            if (inserted) {
                qli_state->tensor = make_qli_metadata(
                    runtime, scheduled_query_length, qli_key_length,
                    requests);
                qli_state->refresh =
                    [runtime_ptr = &runtime,
                     tensor = qli_state->tensor,
                     query_length = scheduled_query_length,
                     requests](
                        size_t key_length) {
                        refresh_qli_metadata(
                            *runtime_ptr, tensor, query_length, key_length,
                            requests);
                    };
            }
            qli_metadata = qli_state->tensor;
        } else {
            // Batched sequences have independent causal ranges; use the vendo
            // metadata kernel, which recomputes per replay as a host segment.
            qli_metadata = infinilm::runtime::Tensor::empty(
                runtime, {size_t{1024}},
                infinilm::runtime::DataType::kInt32);
            infinilm::execution::submit<
                infini::ops::VllmQuantLightningIndexerMetadata>(
                runtime,
                infini::ops::VllmQuantLightningIndexerMetadata::DefaultConfig(
                    runtime.device().type()),
                qli_metadata, query_lengths,
                *metadata.total_sequence_lengths,
                static_cast<int64_t>(indexer_heads), int64_t{1},
                static_cast<int64_t>(indexer_head_dim), int64_t{0}, int64_t{0},
                static_cast<int64_t>(requests),
                static_cast<int64_t>(packed_decode ? 1 : max_query),
                static_cast<int64_t>(max_kv), std::string{"TND"},
                std::string{"PA_BSND"}, static_cast<int64_t>(index_topk),
                int64_t{3}, std::numeric_limits<int64_t>::max(),
                std::numeric_limits<int64_t>::max(), int64_t{4});
        }
        compressed_sparse_indices = infinilm::runtime::Tensor::empty(
            runtime, {query->size(0), 1, index_topk},
            infinilm::runtime::DataType::kInt32);
        infinilm::execution::submit<infini::ops::VllmQuantLightningIndexer>(
            runtime,
            infini::ops::VllmQuantLightningIndexer::DefaultConfig(runtime.device().type()),
            compressed_sparse_indices, empty_f32,
            indexer_query_i8_flat->view(
                {query->size(0), indexer_heads, indexer_head_dim}),
            indexer_key_cache, indexer_weights_f16,
            indexer_query_scale_f16->view({query->size(0), indexer_heads}),
            indexer_scale_cache, query_lengths, *metadata.total_sequence_lengths,
            compressed_block_table, qli_metadata, int64_t{0}, int64_t{0},
            std::string{"TND"}, std::string{"PA_BSND"},
            static_cast<int64_t>(index_topk), int64_t{3},
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<int64_t>::max(), int64_t{4}, false);
    }

    auto &sas_states = forward_context.deepseek_v4_sas_metadata;
    auto [sas_state_it, sas_inserted] = sas_states.try_emplace(
        compress_ratio,
        std::make_shared<
            infinilm::global_state::DeepseekV4SasMetadataState>());
    auto sas_state = sas_state_it->second;
    if (sas_inserted || !sas_state->tensor) {
        sas_state->tensor = infinilm::runtime::Tensor::empty(
            runtime, {1024}, infinilm::runtime::DataType::kInt32);
    }
    if (!sas_state->ready) {
    infinilm::execution::submit<infini::ops::SparseAttnSharedkvMetadata>(
        runtime, infini::ops::SparseAttnSharedkvMetadata::DefaultConfig(runtime.device().type()),
        sas_state->tensor, *metadata.input_offsets, cu_seqlens_ori_kv,
        empty_i32, seqused_q, *metadata.total_sequence_lengths,
        static_cast<int64_t>(local_heads), int64_t{1},
        static_cast<int64_t>(head_dim),
        static_cast<int64_t>(requests),
        static_cast<int64_t>(max_query), static_cast<int64_t>(max_kv),
        int64_t{0}, compress_ratio == 4 ? static_cast<int64_t>(index_topk) : int64_t{0},
        static_cast<int64_t>(compress_ratio), int64_t{4}, int64_t{3},
        int64_t{127}, int64_t{0}, std::string{"TND"},
        std::string{"PA_ND"}, true, true);
        sas_state->ready = true;
    }
    auto sas_metadata = sas_state->tensor;

    auto output = infinilm::runtime::Tensor::empty(
        runtime, query->shape(), query->dtype());
    infinilm::execution::submit<infini::ops::SparseAttnSharedkv>(
        runtime, infini::ops::SparseAttnSharedkv::DefaultConfig(runtime.device().type()),
        output, empty_f32, query, ori_kv, cmp_kv, empty_i32,
        compressed_sparse_indices, *metadata.block_tables,
        compressed_block_table, *metadata.input_offsets, cu_seqlens_ori_kv,
        empty_i32, empty_i32, *metadata.total_sequence_lengths, sinks,
        sas_metadata, 1.0 / std::sqrt(static_cast<double>(head_dim)),
        static_cast<int64_t>(compress_ratio), int64_t{4}, int64_t{3},
        static_cast<int64_t>(ori_kv->stride(0)),
        static_cast<int64_t>(cmp_kv->stride(0)), int64_t{127}, int64_t{0},
        std::string{"TND"}, std::string{"PA_ND"}, false);
    INFINILM_TRACE_OP(MultiheadAttentionVarlen, output);
    return output;
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor, infinilm::runtime::Tensor>
hc_pre(infinilm::runtime::Runtime &runtime,
       const infinilm::runtime::Tensor &x,
       const infinilm::runtime::Tensor &hc_fn,
       const infinilm::runtime::Tensor &hc_scale,
       const infinilm::runtime::Tensor &hc_base,
       int64_t hc_mult, int64_t sinkhorn_iters, double norm_eps, double hc_eps) {
    if (!x || (x->ndim() != 3 && x->ndim() != 4)) {
        throw std::invalid_argument("DeepseekV4 HC input must be rank 3 or 4");
    }
    const auto shape = x->shape();
    const size_t d = shape.back();
    const bool stream_input = x->ndim() == 4;
    const size_t batch = shape[0];
    const size_t seq = shape[1];
    infinilm::runtime::Tensor y, post, comb;
    if (stream_input) {
        if (shape[2] != static_cast<size_t>(hc_mult)) {
            throw std::invalid_argument("DeepseekV4 HC stream dimension mismatch");
        }
        y = infinilm::runtime::Tensor::empty(runtime, {batch, seq, d}, x->dtype());
    } else {
        y = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    }
    post = infinilm::runtime::Tensor::empty(runtime, {batch, seq, static_cast<size_t>(hc_mult)}, infinilm::runtime::DataType::kFloat32);
    comb = infinilm::runtime::Tensor::empty(runtime, {batch, seq, static_cast<size_t>(hc_mult), static_cast<size_t>(hc_mult)}, infinilm::runtime::DataType::kFloat32);
    if (std::getenv("INFINILM_V4_HC_FALLBACK") != nullptr) {
        if (stream_input) {
            auto first_stream = x->narrow({{2, 0, 1}});
            first_stream = first_stream->squeeze(2);
            first_stream = first_stream->contiguous(runtime);
            infinilm::execution::submit<infini::ops::Copy>(
                runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                first_stream, false, y);
        } else if (x->numel()) {
            infinilm::execution::submit<infini::ops::Copy>(
                runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                x, false, y);
        }
        infinilm::execution::submit<infini::ops::Fill>(
            runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
            post, 0.0, post);
        infinilm::execution::submit<infini::ops::Fill>(
            runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
            comb, 0.0, comb);
        return {y, post, comb};
    }
    infinilm::execution::submit<infini::ops::HcPre>(
        runtime, infini::ops::HcPre::DefaultConfig(runtime.device().type()),
        y, post, comb, x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, norm_eps, hc_eps);
    return {y, post, comb};
}

infinilm::runtime::Tensor hc_post(infinilm::runtime::Runtime &runtime,
                                  const infinilm::runtime::Tensor &x,
                                  const infinilm::runtime::Tensor &residual,
                                  const infinilm::runtime::Tensor &post,
                                  const infinilm::runtime::Tensor &comb) {
    auto out = infinilm::runtime::Tensor::empty(runtime, residual->shape(), residual->dtype());
    if (std::getenv("INFINILM_V4_HC_FALLBACK") != nullptr) {
        const auto &shape = residual->shape();
        auto broadcast = v4_broadcast_materialized(
            runtime, x, {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1]), 4, static_cast<int64_t>(shape[3])});
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()),
            residual, broadcast, out);
        return out;
    }
    infinilm::execution::submit<infini::ops::HcPost>(
        runtime, infini::ops::HcPost::DefaultConfig(runtime.device().type()),
        out, x, residual, post, comb);
    return out;
}

infinilm::runtime::Tensor apply_hc_head(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    const infinilm::runtime::Tensor &hc_fn,
    const infinilm::runtime::Tensor &hc_scale,
    const infinilm::runtime::Tensor &hc_base,
    const infinilm::runtime::Tensor &rms_weight,
    double eps) {
    if (!x || (x->ndim() != 3 && x->ndim() != 4)) {
        throw std::invalid_argument(
            "DeepseekV4 HC head input must be [tokens, streams, hidden] or "
            "[batch, seq, streams, hidden]");
    }
    const auto shape = x->shape();
    const size_t hidden = shape.back();
    const size_t streams = shape[shape.size() - 2];
    const size_t rows = x->numel() / (streams * hidden);
    auto flat = x->view({rows, streams, hidden});
    auto output = infinilm::runtime::Tensor::empty(
        runtime, {rows, hidden}, x->dtype());
    infinilm::execution::submit<infini::ops::DeepseekV4HcHead>(
        runtime,
        infini::ops::DeepseekV4HcHead::DefaultConfig(runtime.device().type()),
        output, flat, hc_fn, hc_scale, hc_base, rms_weight, eps);
    if (x->ndim() == 4) {
        return output->view({shape[0], shape[1], hidden});
    }
    return output;
}

} // namespace

DeepseekV4MLA::DeepseekV4MLA(infinilm::runtime::Runtime &runtime,
                             std::shared_ptr<infinilm::config::ModelConfig> config,
                             size_t layer_idx,
                             const engine::distributed::RankInfo &rank_info,
                             backends::AttentionBackend attention_backend,
                             const layers::moe::EPConfig &,
                             std::shared_ptr<infinilm::layers::RoPE> rope)
    : layer_idx_(layer_idx) {
    models::validate_runtime_construction(config, runtime, rank_info);
    const auto dtype = config->get_dtype();
    const auto q = w8a8_quant(config);
    const size_t hidden = config->get<size_t>("hidden_size");
    const size_t q_lora = config->get<size_t>("q_lora_rank");
    const size_t heads = config->get<size_t>("num_attention_heads");
    const size_t head_dim = config->get<size_t>("head_dim");
    const size_t o_groups = config->get<size_t>("o_groups");
    const size_t o_lora = config->get<size_t>("o_lora_rank");
    if (rank_info.tp_size <= 0 || heads % static_cast<size_t>(rank_info.tp_size) != 0
        || o_groups % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::invalid_argument("DeepseekV4 MLA heads and output groups must divide TP size");
    }
    const size_t local_heads = heads / static_cast<size_t>(rank_info.tp_size);
    const auto &config_json = config->get_config_json();
    if (config_json.contains("compress_ratios") && layer_idx < config_json["compress_ratios"].size()) {
        compress_ratio_ = config_json["compress_ratios"][layer_idx].get<size_t>();
    }
    wq_a_ = bind_child("wq_a", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden, q_lora, q, false, dtype));
    q_norm_ = bind_child("q_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, q_lora, config->get<double>("rms_norm_eps"), dtype));
    wq_b_ = bind_child("wq_b", std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
        runtime, q_lora, heads * head_dim, q, false, dtype,
        rank_info.tp_rank, rank_info.tp_size, static_cast<int>(heads)));
    wkv_ = bind_child("wkv", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden, head_dim, q, false, dtype));
    kv_norm_ = bind_child("kv_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim, config->get<double>("rms_norm_eps"), infinilm::runtime::DataType::kFloat32));
    wo_a_ = bind_child("wo_a", std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
        runtime, local_heads * head_dim, o_groups * o_lora, dense_quant(), false, dtype,
        rank_info.tp_rank, rank_info.tp_size));
    wo_b_ = bind_child("wo_b", std::make_shared<infinilm::layers::linear::RowParallelLinear>(
        runtime, o_groups * o_lora, hidden, dense_quant(), false, dtype,
        rank_info.tp_rank, rank_info.tp_size, rank_info.comm));
    if (compress_ratio_ > 1) {
        // Ratio 4 retains the full latent rank and carries the sparse-attention indexer.
        // Ratio 128 uses the half-rank compressed stream and has no indexer tensors.
        const size_t compressed_rank = compress_ratio_ == 128 ? q_lora / 2 : q_lora;
        compressor_ape_ = infinilm::weights::Weight(runtime, {compress_ratio_, compressed_rank}, infinilm::runtime::DataType::kFloat32);
        compressor_wkv_ = infinilm::weights::Weight(runtime, {compressed_rank, hidden}, dtype);
        compressor_wgate_ = infinilm::weights::Weight(runtime, {compressed_rank, hidden}, dtype);
        compressor_norm_ = infinilm::weights::Weight(runtime, {head_dim}, dtype);
        bind_weight("compressor.ape", compressor_ape_);
        bind_weight("compressor.wkv.weight", compressor_wkv_);
        bind_weight("compressor.wgate.weight", compressor_wgate_);
        bind_weight("compressor.norm.weight", compressor_norm_);
        if (compress_ratio_ == 4) {
            indexer_heads_ = config->get_or<size_t>("index_n_heads", heads);
            indexer_head_dim_ = config->get_or<size_t>("index_head_dim", head_dim / 4);
            index_topk_ = config->get_or<size_t>("index_topk", 512);
            const size_t indexer_coff = head_dim / indexer_head_dim_;
            indexer_wq_b_ = bind_child("indexer.wq_b", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, q_lora, indexer_heads_ * indexer_head_dim_, q, false, dtype));
            indexer_weights_proj_ = bind_child("indexer.weights_proj", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden, indexer_heads_, dense_quant(), false, dtype));
            indexer_ape_ = infinilm::weights::Weight(runtime, {compress_ratio_, q_lora / indexer_coff}, infinilm::runtime::DataType::kFloat32);
            indexer_wkv_ = infinilm::weights::Weight(runtime, {q_lora / indexer_coff, hidden}, dtype);
            indexer_wgate_ = infinilm::weights::Weight(runtime, {q_lora / indexer_coff, hidden}, dtype);
            indexer_norm_ = infinilm::weights::Weight(runtime, {indexer_head_dim_}, dtype);
            bind_weight("indexer.compressor.ape", indexer_ape_);
            bind_weight("indexer.compressor.wkv.weight", indexer_wkv_);
            bind_weight("indexer.compressor.wgate.weight", indexer_wgate_);
            bind_weight("indexer.compressor.norm.weight", indexer_norm_);
        }
    }
    local_heads_ = local_heads;
    head_dim_ = head_dim;
    o_groups_ = o_groups;
    o_lora_ = o_lora;
    model_dtype_ = dtype;
    attention_backend_ = attention_backend;
    rope_ = std::move(rope);
    if (!rope_) {
        throw std::invalid_argument("DeepseekV4 MLA requires shared RoPE");
    }
    q_head_norm_weight_ = infinilm::runtime::Tensor::empty_parameter(
        runtime, {head_dim}, dtype);
    infinilm::execution::submit<infini::ops::Fill>(
        runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
        q_head_norm_weight_, 1.0, q_head_norm_weight_);
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        local_heads, head_dim, 1.0f / std::sqrt(static_cast<float>(head_dim)), 1,
        cache_index(layer_idx, CacheSlot::SlidingWindow),
        infinilm::runtime::Tensor{}, infinilm::runtime::Tensor{}, attention_backend_,
        config->get_kv_quant_scheme(), std::vector<int64_t>{128, 0});
    attn_sink_ = infinilm::weights::Weight(
        runtime, {heads}, infinilm::runtime::DataType::kFloat32,
        0, rank_info.tp_rank, rank_info.tp_size);
    bind_weight("attn_sink", attn_sink_);
}

infinilm::runtime::Tensor DeepseekV4MLA::forward(const infinilm::runtime::Tensor &positions,
                                                   const infinilm::runtime::Tensor &hidden_states,
                                                   infinilm::global_state::ForwardContext &forward_context,
                                                   const infinilm::runtime::Tensor &hidden_packed,
                                                   const infinilm::runtime::Tensor &hidden_scale) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden = hidden_states;
    const auto hshape = hidden->shape();
    if (hshape.size() != 3) throw std::invalid_argument("DeepseekV4 MLA expects [batch, seq, hidden]");
    const size_t batch = hshape[0];
    const size_t seq = hshape[1];
    auto hidden_flat = hidden->view({batch * seq, hshape[2]});
    auto q_a = hidden_packed
        ? wq_a_->forward_quantized(runtime, hidden_packed, hidden_scale)
        : wq_a_->forward(runtime, hidden_flat);
    auto [q_latent_packed, q_latent_scale] = rms_norm_dynamic_quant(
        runtime, q_a, q_norm_->weight(), q_norm_->eps());
    auto q_linear = wq_b_->forward_quantized(
        runtime, q_latent_packed, q_latent_scale);
    auto q_norm_input = q_linear->view({batch * seq * local_heads_, head_dim_});
    auto q_norm_output = infinilm::runtime::Tensor::empty(
        runtime, q_norm_input->shape(), q_norm_input->dtype());
    infinilm::execution::submit<infini::ops::RmsNorm>(
        runtime, infini::ops::RmsNorm::DefaultConfig(runtime.device().type()),
        q_norm_input, q_head_norm_weight_,
        static_cast<float>(1e-6), q_norm_output);
    q_linear = q_norm_output->view({batch, seq, local_heads_ * head_dim_});
    if (layer_idx_ == 0 && std::getenv("INFINILM_V4_DEBUG_STATS") != nullptr) {
        auto last_hidden = hidden_flat->narrow(
            {{0, batch * seq - 1, 1}})->contiguous(runtime);
        auto last_q_latent = q_norm_->forward(
            runtime, wq_a_->forward(runtime, last_hidden));
        auto last_q_linear = wq_b_->forward(runtime, last_q_latent)
            ->view({local_heads_, head_dim_});
        auto last_q_norm = infinilm::runtime::Tensor::empty(
            runtime, last_q_linear->shape(), last_q_linear->dtype());
        infinilm::execution::submit<infini::ops::RmsNorm>(
            runtime, infini::ops::RmsNorm::DefaultConfig(runtime.device().type()),
            last_q_linear, q_head_norm_weight_, static_cast<float>(1e-6),
            last_q_norm);
        v4_debug_stats(runtime, last_q_norm, "layer0.q_last_individual");
    }
    auto kv_linear = kv_norm_->forward(
        runtime,
        hidden_packed ? wkv_->forward_quantized(runtime, hidden_packed, hidden_scale)
                       : wkv_->forward(runtime, hidden_flat));
    const bool paged = attention_backend_ != backends::AttentionBackend::STATIC_ATTN;
    infinilm::runtime::Tensor q, k, v;
    if (paged) {
        q = q_linear->view({seq, local_heads_, head_dim_});
        k = kv_linear->view({seq, 1, head_dim_});
        v = kv_linear->view({seq, 1, head_dim_});
    } else {
        q = q_linear->view({batch, seq, local_heads_, head_dim_});
        k = kv_linear->view({batch, seq, 1, head_dim_});
        v = kv_linear->view({batch, seq, 1, head_dim_});
    }
    const size_t rope_dim = std::min<size_t>(64, head_dim_);
    auto pos = positions;
    if (paged && pos->ndim() == 2) pos = pos->view({pos->numel()});
    infinilm::runtime::Tensor rope_positions;
    const auto &rope_offsets = forward_context.attn_metadata.input_offsets;
    if (rope_offsets && rope_offsets.value()->size(0) > 2
        && static_cast<size_t>(pos->numel()) + 1
               == rope_offsets.value()->size(0)) {
        // Packed decode: one new token per request at independent positions.
        rope_positions = pos;
    }
    infinilm::runtime::Tensor q_rope;
    infinilm::runtime::Tensor k_rope;
    const bool ascend_partial_rope =
        paged && runtime.device().type() == infinilm::runtime::Device::Type::kAscend;
    if (ascend_partial_rope) {
        const size_t start_position = forward_context.attn_metadata
            .first_past_sequence_length.value_or(0);
        q_rope = q->contiguous(runtime);
        k_rope = k->contiguous(runtime);
        apply_v4_partial_rope(
            runtime, q_rope, rope_, forward_context, start_position, rope_positions, false);
        apply_v4_partial_rope(
            runtime, k_rope, rope_, forward_context, start_position, rope_positions, false);
    } else {
        auto q_rot = q->narrow({
            {q->ndim() - 1, head_dim_ - rope_dim, rope_dim}})->contiguous(runtime);
        auto k_rot = k->narrow({
            {k->ndim() - 1, head_dim_ - rope_dim, rope_dim}})->contiguous(runtime);
        q_rot = rope_->forward(runtime, q_rot, pos, true);
        k_rot = rope_->forward(runtime, k_rot, pos, true);
        auto q_main = q->narrow({
            {q->ndim() - 1, 0, head_dim_ - rope_dim}})->contiguous(runtime);
        auto k_main = k->narrow({
            {k->ndim() - 1, 0, head_dim_ - rope_dim}})->contiguous(runtime);
        q_rope = v4_cat_alloc(
            runtime, {q_main, q_rot}, q->ndim() - 1, q->shape());
        k_rope = v4_cat_alloc(
            runtime, {k_main, k_rot}, k->ndim() - 1, k->shape());
    }
    if (layer_idx_ == 0) {
        const auto past = forward_context.attn_metadata
            .first_past_sequence_length.value_or(0);
        const std::string suffix = ".past" + std::to_string(past);
        v4_debug_stats(runtime,
                       q_rope->view({seq, local_heads_ * head_dim_}),
                       "layer0.q_rope" + suffix);
        v4_debug_stats(runtime, k_rope->view({seq, head_dim_}),
                       "layer0.k_rope" + suffix);
    }
    infinilm::runtime::Tensor attn_out;
    if (attention_backend_ != backends::AttentionBackend::STATIC_ATTN) {
        infinilm::runtime::Tensor sinks = attn_sink_;
        if (compress_ratio_ <= 1) {
            attn_out = sparse_shared_kv_attention(
                runtime, q_rope, k_rope, layer_idx_, local_heads_, head_dim_,
                sinks, forward_context);
        } else {
            infinilm::runtime::Tensor indexer_query;
            infinilm::runtime::Tensor indexer_weights;
            if (compress_ratio_ == 4) {
                auto indexer_q = indexer_wq_b_->forward_quantized(
                    runtime, q_latent_packed, q_latent_scale)
                    ->view({seq, indexer_heads_, indexer_head_dim_});
                const size_t indexer_rope_dim = std::min<size_t>(
                    rope_->rotary_dim(), indexer_head_dim_);
                if (ascend_partial_rope) {
                    indexer_query = indexer_q->contiguous(runtime);
                    const size_t start_position = forward_context.attn_metadata
                        .first_past_sequence_length.value_or(0);
                    apply_v4_partial_rope(
                        runtime, indexer_query, rope_, forward_context, start_position, rope_positions, false);
                } else {
                    auto indexer_q_rot = indexer_q->narrow({
                        {indexer_q->ndim() - 1,
                         indexer_head_dim_ - indexer_rope_dim,
                         indexer_rope_dim}})->contiguous(runtime);
                    indexer_q_rot = rope_->forward(
                        runtime, indexer_q_rot, pos, true);
                    auto indexer_q_main = indexer_q->narrow({
                        {indexer_q->ndim() - 1, 0,
                         indexer_head_dim_ - indexer_rope_dim}})->contiguous(runtime);
                    indexer_query = v4_cat_alloc(
                        runtime, {indexer_q_main, indexer_q_rot},
                        indexer_q->ndim() - 1, indexer_q->shape());
                }
                indexer_weights = indexer_weights_proj_->forward(
                    runtime, hidden_flat)->view({seq, indexer_heads_});
            }
            attn_out = compressed_sparse_shared_kv_attention(
                runtime, q_rope, k_rope, hidden, indexer_query,
                indexer_weights, layer_idx_, local_heads_, head_dim_,
                compress_ratio_, indexer_heads_, indexer_head_dim_,
                index_topk_, sinks,
                static_cast<infinilm::runtime::Tensor>(compressor_ape_),
                static_cast<infinilm::runtime::Tensor>(compressor_wkv_),
                static_cast<infinilm::runtime::Tensor>(compressor_wgate_),
                static_cast<infinilm::runtime::Tensor>(compressor_norm_),
                static_cast<infinilm::runtime::Tensor>(indexer_ape_),
                static_cast<infinilm::runtime::Tensor>(indexer_wkv_),
                static_cast<infinilm::runtime::Tensor>(indexer_wgate_),
                static_cast<infinilm::runtime::Tensor>(indexer_norm_),
                rope_, forward_context);
        }
    } else {
        attn_out = attn_->forward(q_rope, k_rope, v, forward_context);
    }
    if (layer_idx_ == 0) {
        const auto past = forward_context.attn_metadata
            .first_past_sequence_length.value_or(0);
        v4_debug_stats(runtime,
                       attn_out->view({seq, local_heads_ * head_dim_}),
                       "layer0.attn_raw.past" + std::to_string(past));
    }
    if (ascend_partial_rope) {
        const size_t start_position = forward_context.attn_metadata
            .first_past_sequence_length.value_or(0);
        apply_v4_partial_rope(
            runtime, attn_out, rope_, forward_context, start_position, rope_positions, true);
    } else if (paged) {
        auto output_rot = attn_out->narrow({
            {attn_out->ndim() - 1, head_dim_ - rope_dim, rope_dim}})->contiguous(runtime);
        output_rot = rope_->inverse(runtime, output_rot, pos, true);
        auto output_main = attn_out->narrow({
            {attn_out->ndim() - 1, 0, head_dim_ - rope_dim}})->contiguous(runtime);
        attn_out = v4_cat_alloc(runtime, {output_main, output_rot},
                                attn_out->ndim() - 1, attn_out->shape());
    }
    if (layer_idx_ == 0) {
        const auto past = forward_context.attn_metadata
            .first_past_sequence_length.value_or(0);
        v4_debug_stats(runtime,
                       attn_out->view({seq, local_heads_ * head_dim_}),
                       "layer0.attn_inverse.past" + std::to_string(past));
    }
    const size_t rows = attn_out->numel() / (local_heads_ * head_dim_);
    auto attn_flat = attn_out->view({rows, local_heads_ * head_dim_});
    auto projected = wo_a_->forward(runtime, attn_flat);
    auto output = wo_b_->forward(runtime, projected);
    if (layer_idx_ == 0) {
        v4_debug_stats(runtime, output, "layer0.attn_projected");
    }
    if (paged) return output->view({1, seq, hidden_states->shape()[2]});
    return output->view({batch, seq, hidden_states->shape()[2]});
}

DeepseekV4ExpertBank::DeepseekV4ExpertBank(infinilm::runtime::Runtime &runtime,
                                               const std::shared_ptr<infinilm::config::ModelConfig> &config,
                                               const engine::distributed::RankInfo &rank_info,
                                               size_t expert_count,
                                               size_t intermediate_size,
                                               bool grouped_layout)
    : expert_count_(expert_count), grouped_layout_(grouped_layout),
      fused_w13_(v4_fused_w13_enabled()) {
    models::validate_runtime_construction(config, runtime, rank_info);
    const size_t hidden = config->get<size_t>("hidden_size");
    const auto int8 = infinilm::runtime::DataType::kInt8;
    const auto fp32 = infinilm::runtime::DataType::kFloat32;
    const auto bf16 = infinilm::runtime::DataType::kBFloat16;
    if (intermediate_size % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::invalid_argument("DeepseekV4 expert intermediate size must be TP divisible");
    }
    local_intermediate_size_ = intermediate_size / static_cast<size_t>(rank_info.tp_size);

    if (grouped_layout_) {
        w13_bank_ = bind_derived_weight("w13_bank", infinilm::weights::Weight(
            runtime, {expert_count, hidden, 2 * local_intermediate_size_}, int8));
        w13_scale_bank_ = bind_derived_weight("w13_scale_bank", infinilm::weights::Weight(
            runtime, {expert_count, 2 * local_intermediate_size_}, fp32));
        grouped_w2_bank_ = bind_derived_weight("w2_bank", infinilm::weights::Weight(
            runtime, {expert_count, local_intermediate_size_, hidden}, int8));
        grouped_w2_scale_bank_ = bind_derived_weight("w2_scale_bank", infinilm::weights::Weight(
            runtime, {expert_count, hidden}, bf16));
        return;
    }

    w1_bank_ = bind_derived_weight("w1_bank", infinilm::weights::Weight(
        runtime, {expert_count, intermediate_size, hidden}, int8,
        1, rank_info.tp_rank, rank_info.tp_size));
    w1_scale_bank_ = bind_derived_weight("w1_scale_bank", infinilm::weights::Weight(
        runtime, {expert_count, intermediate_size, 1}, fp32,
        1, rank_info.tp_rank, rank_info.tp_size));
    w2_bank_ = bind_derived_weight("w2_bank", infinilm::weights::Weight(
        runtime, {expert_count, hidden, intermediate_size}, int8,
        2, rank_info.tp_rank, rank_info.tp_size));
    w2_scale_bank_ = bind_derived_weight("w2_scale_bank", infinilm::weights::Weight(
        runtime, {expert_count, hidden, 1}, fp32));
    w3_bank_ = bind_derived_weight("w3_bank", infinilm::weights::Weight(
        runtime, {expert_count, intermediate_size, hidden}, int8,
        1, rank_info.tp_rank, rank_info.tp_size));
    w3_scale_bank_ = bind_derived_weight("w3_scale_bank", infinilm::weights::Weight(
        runtime, {expert_count, intermediate_size, 1}, fp32,
        1, rank_info.tp_rank, rank_info.tp_size));
}

infinilm::runtime::Tensor DeepseekV4ExpertBank::slice(
    const infinilm::weights::Weight &weight, size_t expert_index) const {
    auto tensor = static_cast<infinilm::runtime::Tensor>(weight);
    if (!tensor || tensor->ndim() < 2 || expert_index >= expert_count_) {
        throw std::invalid_argument("DeepseekV4 expert bank slice is invalid");
    }
    auto narrowed = tensor->narrow({{0, expert_index, 1}});
    infinilm::runtime::Shape shape(narrowed->shape().begin() + 1,
                                    narrowed->shape().end());
    return narrowed->view(shape);
}

void DeepseekV4ExpertBank::process_weights_after_loading(
    infinilm::runtime::Runtime &runtime, bool) {
    if (!grouped_layout_ || !w13_bank_) {
        return;
    }
    auto repack_to_nz = [&](infinilm::weights::Weight &bank) {
        auto weight = static_cast<infinilm::runtime::Tensor>(bank);
        const auto &shape = weight->shape();
        if (shape.size() != 3 || shape[1] % 16 != 0 || shape[2] % 32 != 0) {
            throw std::invalid_argument(
                "DeepseekV4 expert bank layout cannot be converted to NZ");
        }

        const size_t experts = shape[0];
        const size_t input_channels = shape[1];
        const size_t output_channels = shape[2];
        const size_t expert_elements = input_channels * output_channels;
        for (size_t expert = 0; expert < experts; ++expert) {
            auto source = slice(bank, expert);
            auto repacked = source
                                ->view({input_channels / 16, 16,
                                        output_channels / 32, 32})
                                ->permute({2, 0, 1, 3})
                                ->contiguous(runtime);
            slice(bank, expert)
                ->view({expert_elements})
                ->copy_from(runtime, repacked->view({expert_elements}));
            runtime.syncStream();
        }

        auto nz = weight->view({experts, output_channels / 32,
                                input_channels / 16, 16, 32});
        bank.tensor() = nz;
    };
    if (fused_w13_) {
        repack_to_nz(w13_bank_);
    }
    if (std::getenv("INFINILM_DISABLE_W2_NZ") == nullptr) {
        repack_to_nz(grouped_w2_bank_);
    }
}

DeepseekV4Expert::DeepseekV4Expert(
    const std::shared_ptr<DeepseekV4ExpertBank> &bank,
    size_t expert_index,
    const std::shared_ptr<infinilm::config::ModelConfig> &config,
    const engine::distributed::RankInfo &rank_info)
    : quantization_(config->get_quantization_method()), communicator_(rank_info.comm) {
    const auto make_weight = [&](const infinilm::runtime::Tensor &tensor,
                                 int dim, int rank, int tp_size) {
        return infinilm::weights::Weight(tensor, dim, rank, tp_size);
    };

    if (bank->grouped_layout_) {
        const size_t local_i = bank->local_intermediate_size_;
        auto w13 = bank->slice(bank->w13_bank_, expert_index);
        auto w13_scale = bank->slice(bank->w13_scale_bank_, expert_index);
        auto packed_w2 = bank->slice(bank->grouped_w2_bank_, expert_index);
        auto packed_w2_scale = bank->slice(bank->grouped_w2_scale_bank_, expert_index);

        auto w1_view = w13->narrow({{1, 0, local_i}})->permute({1, 0});
        auto w3_view = w13->narrow({{1, local_i, local_i}})->permute({1, 0});
        auto w2_view = packed_w2->permute({1, 0});
        auto w1_scale_view =
            w13_scale->narrow({{0, 0, local_i}})->view({local_i, 1});
        auto w3_scale_view =
            w13_scale->narrow({{0, local_i, local_i}})->view({local_i, 1});
        auto w2_scale_view = packed_w2_scale->view({packed_w2_scale->shape()[0], 1});

        w1_ = make_weight(w1_view, 0, rank_info.tp_rank, rank_info.tp_size);
        w1_scale_ = make_weight(w1_scale_view, 0, rank_info.tp_rank, rank_info.tp_size);
        w2_ = make_weight(w2_view, 1, rank_info.tp_rank, rank_info.tp_size);
        w2_scale_ = make_weight(w2_scale_view, -1, 0, 1);
        w3_ = make_weight(w3_view, 0, rank_info.tp_rank, rank_info.tp_size);
        w3_scale_ = make_weight(w3_scale_view, 0, rank_info.tp_rank, rank_info.tp_size);
    } else {
        w1_ = make_weight(bank->slice(bank->w1_bank_, expert_index),
                          0, rank_info.tp_rank, rank_info.tp_size);
        w1_scale_ = make_weight(bank->slice(bank->w1_scale_bank_, expert_index),
                                0, rank_info.tp_rank, rank_info.tp_size);
        w2_ = make_weight(bank->slice(bank->w2_bank_, expert_index),
                          1, rank_info.tp_rank, rank_info.tp_size);
        w2_scale_ = make_weight(bank->slice(bank->w2_scale_bank_, expert_index),
                                -1, 0, 1);
        w3_ = make_weight(bank->slice(bank->w3_bank_, expert_index),
                          0, rank_info.tp_rank, rank_info.tp_size);
        w3_scale_ = make_weight(bank->slice(bank->w3_scale_bank_, expert_index),
                                0, rank_info.tp_rank, rank_info.tp_size);
    }

    bind_weight("w1.weight", w1_);
    bind_weight("w1.weight_scale", w1_scale_);
    bind_weight("w2.weight", w2_);
    bind_weight("w2.weight_scale", w2_scale_);
    bind_weight("w3.weight", w3_);
    bind_weight("w3.weight_scale", w3_scale_);
}

infinilm::runtime::Tensor DeepseekV4Expert::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states,
    bool allreduce) const {
    infinilm::quantization::ParamsMap p1{{"weight", w1_}, {"weight_scale", w1_scale_}};
    infinilm::quantization::ParamsMap p3{{"weight", w3_}, {"weight_scale", w3_scale_}};
    auto [gate, up] = quantization_->forward_gate_up(
        runtime, p1, p3, hidden_states, false, false);
    auto intermediate = infinilm::runtime::Tensor::empty(runtime, gate->shape(), gate->dtype());
    infinilm::layers::mlp::branch_activation_into(runtime, intermediate, up, gate);
    infinilm::quantization::ParamsMap p2{{"weight", w2_}, {"weight_scale", w2_scale_}};
    if (!allreduce) {
        return quantization_->forward(runtime, p2, intermediate, false);
    }
    return quantization_->forward_allreduce(runtime, p2, intermediate, false, communicator_);
}

DeepseekV4MoE::DeepseekV4MoE(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend,
    const layers::moe::EPConfig &)
    : layer_idx_(layer_idx),
      hidden_size_(config->get<size_t>("hidden_size")),
      expert_count_(config->get<size_t>("n_routed_experts")),
      topk_(config->get<size_t>("num_experts_per_tok")),
      routed_scaling_factor_(config->get_or<double>("routed_scaling_factor", 1.0)),
      swiglu_limit_(config->get_or<double>("swiglu_limit", 10.0)),
      hash_router_(layer_idx < config->get<size_t>("num_hash_layers")),
      fused_w13_(v4_fused_w13_enabled()),
      communicator_(rank_info.comm) {
    models::validate_runtime_construction(config, runtime, rank_info);
    const size_t intermediate = config->get<size_t>("moe_intermediate_size");
    if (intermediate % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::invalid_argument("DeepseekV4 MoE intermediate size must be TP divisible");
    }
    local_intermediate_size_ = intermediate / static_cast<size_t>(rank_info.tp_size);

    gate_ = bind_child("gate", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
        runtime, hidden_size_, expert_count_, false, config->get_dtype(),
        infinilm::runtime::DataType::kFloat32));
    if (!hash_router_) {
        gate_bias_ = infinilm::weights::Weight(
            runtime, {expert_count_}, infinilm::runtime::DataType::kFloat32);
        bind_weight("gate.bias", gate_bias_);
    } else {
        tid2eid_ = infinilm::weights::Weight(
            runtime,
            {config->get<size_t>("vocab_size"), topk_},
            infinilm::runtime::DataType::kInt32);
        bind_weight("gate.tid2eid", tid2eid_);
    }

    expert_bank_ = bind_child("expert_bank", std::make_shared<DeepseekV4ExpertBank>(
        runtime, config, rank_info, expert_count_, intermediate, true));
    experts_.reserve(expert_count_);
    for (size_t i = 0; i < expert_count_; ++i) {
        experts_.push_back(bind_child(
            "experts." + std::to_string(i),
            std::make_shared<DeepseekV4Expert>(expert_bank_, i, config, rank_info)));
    }

    const size_t shared = config->get<size_t>("n_shared_experts");
    if (shared) {
        auto shared_config = std::make_shared<infinilm::config::ModelConfig>(*config);
        const size_t shared_intermediate = intermediate * shared;
        shared_config->get_config_json()["moe_intermediate_size"] = shared_intermediate;
        shared_expert_bank_ = bind_child(
            "shared_expert_bank", std::make_shared<DeepseekV4ExpertBank>(
                runtime, shared_config, rank_info, 1, shared_intermediate, false));
        shared_experts_ = bind_child(
            "shared_experts", std::make_shared<DeepseekV4Expert>(
                shared_expert_bank_, 0, shared_config, rank_info));
    }
}

infinilm::runtime::Tensor DeepseekV4MoE::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &input_ids,
    infinilm::global_state::ForwardContext &) const {
    if (!expert_bank_ || !expert_bank_->grouped_layout_) {
        throw std::runtime_error("DeepseekV4 routed expert bank is unavailable");
    }
    if (hidden_states->shape().back() != hidden_size_) {
        throw std::invalid_argument("DeepseekV4 MoE hidden size mismatch");
    }

    const size_t tokens = hidden_states->numel() / hidden_size_;
    auto flat = hidden_states->view({tokens, hidden_size_});
    auto flat_mutable = flat;
    auto logits = gate_->forward(runtime, flat_mutable);

    std::optional<infinilm::runtime::Tensor> route_bias;
    std::optional<infinilm::runtime::Tensor> route_input_ids;
    std::optional<infinilm::runtime::Tensor> route_tid2eid;
    if (hash_router_) {
        if (!input_ids || input_ids->numel() != tokens) {
            throw std::invalid_argument(
                "DeepseekV4 hash router requires one input id per token");
        }
        route_input_ids = input_ids->view({tokens});
        route_tid2eid = static_cast<infinilm::runtime::Tensor>(tid2eid_);
    } else {
        route_bias = static_cast<infinilm::runtime::Tensor>(gate_bias_);
    }

    auto topk_weights = infinilm::runtime::Tensor::empty(
        runtime, {tokens, topk_}, infinilm::runtime::DataType::kFloat32);
    auto topk_ids = infinilm::runtime::Tensor::empty(
        runtime, {tokens, topk_}, infinilm::runtime::DataType::kInt32);
    auto normalized_scores = infinilm::runtime::Tensor::empty(
        runtime, {tokens, expert_count_}, infinilm::runtime::DataType::kFloat32);
    infinilm::execution::submit<infini::ops::DeepseekV4MoeGatingTopK>(
        runtime, infini::ops::DeepseekV4MoeGatingTopK::DefaultConfig(
                     runtime.device().type()),
        topk_weights, topk_ids, normalized_scores, logits, route_bias,
        route_input_ids, route_tid2eid, static_cast<int64_t>(topk_),
        routed_scaling_factor_);

    const size_t expanded_tokens = tokens * topk_;
    auto expanded_x = infinilm::runtime::Tensor::empty(
        runtime, {expanded_tokens, hidden_size_},
        infinilm::runtime::DataType::kInt8);
    auto expanded_row_idx = infinilm::runtime::Tensor::empty(
        runtime, {expanded_tokens}, infinilm::runtime::DataType::kInt32);
    auto expert_tokens_count_i32 = infinilm::runtime::Tensor::empty(
        runtime, {expert_count_}, infinilm::runtime::DataType::kInt32);
    auto input_scale = infinilm::runtime::Tensor::empty(
        runtime, {expanded_tokens}, infinilm::runtime::DataType::kFloat32);
    infinilm::execution::submit<infini::ops::DeepseekV4MoeInitRoutingQuant>(
        runtime, infini::ops::DeepseekV4MoeInitRoutingQuant::DefaultConfig(
                     runtime.device().type()),
        expanded_x, expanded_row_idx, expert_tokens_count_i32, input_scale,
        flat, topk_ids);
    auto expert_tokens_count = infinilm::runtime::Tensor::empty(
        runtime, {expert_count_}, infinilm::runtime::DataType::kInt64);
    infinilm::execution::submit<infini::ops::Cast>(
        runtime, infini::ops::Cast::DefaultConfig(runtime.device().type()),
        expert_tokens_count_i32, expert_tokens_count);

    auto expert_token_ends = infinilm::runtime::Tensor::empty(
        runtime, {expert_count_}, infinilm::runtime::DataType::kInt64);
    infinilm::execution::submit<infini::ops::Cumsum>(
        runtime, infini::ops::Cumsum::DefaultConfig(runtime.device().type()),
        expert_tokens_count, int64_t{0},
        std::optional<infinilm::runtime::DataType>{
            infinilm::runtime::DataType::kInt64},
        expert_token_ends);

    auto activated = infinilm::runtime::Tensor::empty(
        runtime, {expanded_tokens, local_intermediate_size_},
        infinilm::runtime::DataType::kInt8);
    auto activated_scale = infinilm::runtime::Tensor::empty(
        runtime, {expanded_tokens}, infinilm::runtime::DataType::kFloat32);
    if (fused_w13_) {
        infinilm::execution::submit<
            infini::ops::DeepseekV4MoeGroupedMatmulSwigluQuant>(
            runtime,
            infini::ops::DeepseekV4MoeGroupedMatmulSwigluQuant::DefaultConfig(
                runtime.device().type()),
            activated, activated_scale, expanded_x,
            static_cast<infinilm::runtime::Tensor>(expert_bank_->w13_bank_),
            static_cast<infinilm::runtime::Tensor>(expert_bank_->w13_scale_bank_),
            input_scale, expert_token_ends, swiglu_limit_);
    } else {
        auto gate_up = infinilm::runtime::Tensor::empty(
            runtime, {expanded_tokens, 2 * local_intermediate_size_},
            infinilm::runtime::DataType::kInt32);
        infinilm::execution::submit<infini::ops::DeepseekV4MoeGroupedMatmul>(
            runtime, infini::ops::DeepseekV4MoeGroupedMatmul::DefaultConfig(
                         runtime.device().type()),
            gate_up, expanded_x,
            static_cast<infinilm::runtime::Tensor>(expert_bank_->w13_bank_),
            std::nullopt, std::nullopt, expert_tokens_count, 1);
        infinilm::execution::submit<
            infini::ops::DeepseekV4MoeDequantSwigluQuant>(
            runtime, infini::ops::DeepseekV4MoeDequantSwigluQuant::DefaultConfig(
                         runtime.device().type()),
            activated, activated_scale, gate_up,
            static_cast<infinilm::runtime::Tensor>(expert_bank_->w13_scale_bank_),
            input_scale, expert_tokens_count);
    }

    auto expert_output = infinilm::runtime::Tensor::empty(
        runtime, {expanded_tokens, hidden_size_},
        infinilm::runtime::DataType::kBFloat16);
    infinilm::execution::submit<infini::ops::DeepseekV4MoeGroupedMatmul>(
        runtime, infini::ops::DeepseekV4MoeGroupedMatmul::DefaultConfig(
                     runtime.device().type()),
        expert_output, activated,
        static_cast<infinilm::runtime::Tensor>(expert_bank_->grouped_w2_bank_),
        std::optional<infinilm::runtime::Tensor>{
            static_cast<infinilm::runtime::Tensor>(expert_bank_->grouped_w2_scale_bank_)},
        std::optional<infinilm::runtime::Tensor>{activated_scale},
        expert_tokens_count, 1);

    auto combine_weights = infinilm::runtime::Tensor::empty(
        runtime, topk_weights->shape(), infinilm::runtime::DataType::kBFloat16);
    infinilm::execution::submit<infini::ops::Cast>(
        runtime, infini::ops::Cast::DefaultConfig(runtime.device().type()),
        topk_weights, combine_weights);
    auto routed = infinilm::runtime::Tensor::empty(
        runtime, {tokens, hidden_size_}, infinilm::runtime::DataType::kBFloat16);
    infinilm::execution::submit<infini::ops::DeepseekV4MoeTokenUnpermute>(
        runtime, infini::ops::DeepseekV4MoeTokenUnpermute::DefaultConfig(runtime.device().type()),
        routed, expert_output, expanded_row_idx, combine_weights);

    if (!shared_experts_) {
        infinilm::engine::distributed::allreduce_(
            runtime, routed, routed, infinicclSum, communicator_);
        return routed->view(hidden_states->shape());
    }

    // Routed and shared experts both produce TP partial sums. Add locally,
    // then reduce once instead of sending two small messages per layer.
    auto shared = shared_experts_->forward(runtime, hidden_states, false);
    auto combined = infinilm::runtime::Tensor::empty(
        runtime, routed->shape(), routed->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()),
        routed, shared, combined);
    infinilm::engine::distributed::allreduce_(
        runtime, combined, combined, infinicclSum, communicator_);
    return combined->view(hidden_states->shape());
}

DeepseekV4DecoderLayer::DeepseekV4DecoderLayer(infinilm::runtime::Runtime &runtime,
                                               std::shared_ptr<infinilm::config::ModelConfig> config,
                                               size_t layer_idx,
                                               const engine::distributed::RankInfo &rank_info,
                                               backends::AttentionBackend backend,
                                               const layers::moe::EPConfig &ep_config,
                                               std::shared_ptr<infinilm::layers::RoPE> rope) {
    models::validate_runtime_construction(config, runtime, rank_info);
    const auto hidden = config->get<size_t>("hidden_size");
    const auto eps = config->get<double>("rms_norm_eps");
    attn_norm_ = bind_child("attn_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, infinilm::runtime::DataType::kBFloat16));
    attn_ = bind_child("attn", std::make_shared<DeepseekV4MLA>(
        runtime, config, layer_idx, rank_info, backend, ep_config, std::move(rope)));
    ffn_norm_ = bind_child("ffn_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, infinilm::runtime::DataType::kFloat32));
    ffn_ = bind_child("ffn", std::make_shared<DeepseekV4MoE>(runtime, config, layer_idx, rank_info, backend, ep_config));
    hc_attn_fn_ = infinilm::weights::Weight(runtime, {24, 16384}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_attn_fn", hc_attn_fn_);
    hc_attn_base_ = infinilm::weights::Weight(runtime, {24}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_attn_base", hc_attn_base_);
    hc_attn_scale_ = infinilm::weights::Weight(runtime, {3}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_attn_scale", hc_attn_scale_);
    hc_ffn_fn_ = infinilm::weights::Weight(runtime, {24, 16384}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_ffn_fn", hc_ffn_fn_);
    hc_ffn_base_ = infinilm::weights::Weight(runtime, {24}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_ffn_base", hc_ffn_base_);
    hc_ffn_scale_ = infinilm::weights::Weight(runtime, {3}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_ffn_scale", hc_ffn_scale_);
}

infinilm::runtime::Tensor DeepseekV4DecoderLayer::forward(const infinilm::runtime::Tensor &positions,
                                                           const infinilm::runtime::Tensor &input_ids,
                                                           infinilm::runtime::Tensor hidden_states,
                                                           infinilm::runtime::Tensor &residual,
                                                           infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    int64_t hc_sinkhorn_iters = 10;
    if (forward_context.attn_metadata.input_offsets
        && forward_context.attn_metadata.input_offsets.value()
        && forward_context.attn_metadata.input_offsets.value()->size(0) <= 5) {
        hc_sinkhorn_iters = 20;
    }
    auto [attn_input, attn_post, attn_comb] = hc_pre(
        runtime, hidden_states, hc_attn_fn_, hc_attn_scale_, hc_attn_base_, 4,
        hc_sinkhorn_iters, 1e-6, 1e-6);
    auto [attn_packed, attn_scale] = rms_norm_dynamic_quant(
        runtime, attn_input, attn_norm_->weight(), attn_norm_->eps());
    auto attn_output = attn_->forward(
        positions, attn_input, forward_context, attn_packed, attn_scale);
    auto after_attn = hc_post(runtime, attn_output, hidden_states, attn_post, attn_comb);
    auto [ffn_input, ffn_post, ffn_comb] = hc_pre(
        runtime, after_attn, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_, 4,
        hc_sinkhorn_iters, 1e-6, 1e-6);
    ffn_input = ffn_norm_->forward(runtime, ffn_input);
    auto ffn_output = ffn_->forward(runtime, ffn_input, input_ids, forward_context);
    hidden_states = hc_post(runtime, ffn_output, after_attn, ffn_post, ffn_comb);
    residual = hidden_states;
    return hidden_states;
}

DeepseekV4MTP::DeepseekV4MTP(infinilm::runtime::Runtime &runtime,
                             std::shared_ptr<infinilm::config::ModelConfig> config,
                             const engine::distributed::RankInfo &rank_info,
                             backends::AttentionBackend backend,
                             const layers::moe::EPConfig &ep_config,
                             std::shared_ptr<infinilm::layers::RoPE> rope) {
    models::validate_runtime_construction(config, runtime, rank_info);
    const auto hidden = config->get<size_t>("hidden_size");
    const auto vocab = config->get<size_t>("vocab_size");
    const auto eps = config->get<double>("rms_norm_eps");
    hidden_size_ = hidden;
    hc_mult_ = config->get_or<size_t>("hc_mult", 4);
    norm_eps_ = eps;
    hc_eps_ = config->get_or<double>("hc_eps", 1e-6);
    attn_norm_ = bind_child("attn_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, infinilm::runtime::DataType::kBFloat16));
    attn_ = bind_child("attn", std::make_shared<DeepseekV4MLA>(
        runtime, config, config->get<size_t>("num_hidden_layers"),
        rank_info, backend, ep_config, std::move(rope)));
    ffn_norm_ = bind_child("ffn_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, infinilm::runtime::DataType::kFloat32));
    ffn_ = bind_child("ffn", std::make_shared<DeepseekV4MoE>(runtime, config, config->get<size_t>("num_hash_layers"), rank_info, backend, ep_config));
    enorm_ = bind_child("enorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, config->get_dtype()));
    hnorm_ = bind_child("hnorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, config->get_dtype()));
    e_proj_ = bind_child("e_proj", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden, hidden, dense_quant(), false, config->get_dtype()));
    h_proj_ = bind_child("h_proj", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden, hidden, dense_quant(), false, config->get_dtype()));
    head_ = bind_child("head", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden, vocab, dense_quant(), false, config->get_dtype()));
    norm_ = bind_child("norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, config->get_dtype()));
    emb_ = bind_child("emb", std::make_shared<infinilm::layers::Embedding>(runtime, vocab, hidden, std::nullopt, config->get_dtype()));
    hc_attn_fn_ = infinilm::weights::Weight(runtime, {24, 16384}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_attn_fn", hc_attn_fn_);
    hc_attn_base_ = infinilm::weights::Weight(runtime, {24}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_attn_base", hc_attn_base_);
    hc_attn_scale_ = infinilm::weights::Weight(runtime, {3}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_attn_scale", hc_attn_scale_);
    hc_ffn_fn_ = infinilm::weights::Weight(runtime, {24, 16384}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_ffn_fn", hc_ffn_fn_);
    hc_ffn_base_ = infinilm::weights::Weight(runtime, {24}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_ffn_base", hc_ffn_base_);
    hc_ffn_scale_ = infinilm::weights::Weight(runtime, {3}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_ffn_scale", hc_ffn_scale_);
    hc_head_fn_ = infinilm::weights::Weight(
        runtime, {hc_mult_, hc_mult_ * hidden},
        infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_head_fn", hc_head_fn_);
    hc_head_base_ = infinilm::weights::Weight(
        runtime, {hc_mult_}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_head_base", hc_head_base_);
    hc_head_scale_ = infinilm::weights::Weight(
        runtime, {1}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_head_scale", hc_head_scale_);
    hc_head_rms_weight_ = infinilm::runtime::Tensor::empty_parameter(
        runtime, {hc_mult_ * hidden}, infinilm::runtime::DataType::kFloat32);
    infinilm::execution::submit<infini::ops::Fill>(
        runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
        hc_head_rms_weight_, 1.0, hc_head_rms_weight_);
}

infinilm::runtime::Tensor DeepseekV4MTP::forward(
    const infinilm::runtime::Tensor &positions,
    const infinilm::runtime::Tensor &input_ids,
    const infinilm::runtime::Tensor &hidden_states,
    infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    if (!hidden_states || (hidden_states->ndim() != 3 && hidden_states->ndim() != 4)) {
        throw std::invalid_argument(
            "DeepseekV4 MTP target hidden state must be rank 3 or 4");
    }

    auto embedded = emb_->forward(runtime, input_ids);
    if (embedded->ndim() != 3 || embedded->shape().back() != hidden_size_) {
        throw std::invalid_argument("DeepseekV4 MTP embedding shape is invalid");
    }
    const size_t batch = embedded->size(0);
    const size_t seq = embedded->size(1);
    const size_t tokens = batch * seq;

    infinilm::runtime::Tensor target_streams;
    if (hidden_states->ndim() == 4) {
        if (hidden_states->shape() != infinilm::runtime::Shape{
                batch, seq, hc_mult_, hidden_size_}) {
            throw std::invalid_argument(
                "DeepseekV4 MTP stream target shape does not match inputs");
        }
        target_streams = hidden_states;
    } else {
        if (hidden_states->shape() != infinilm::runtime::Shape{
                batch, seq, hc_mult_ * hidden_size_}) {
            throw std::invalid_argument(
                "DeepseekV4 MTP flattened target shape does not match inputs");
        }
        target_streams = hidden_states->view(
            {batch, seq, hc_mult_, hidden_size_});
    }

    auto masked_embedding = infinilm::runtime::Tensor::empty(
        runtime, embedded->shape(), embedded->dtype());
    infinilm::execution::submit<infini::ops::Copy>(
        runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
        embedded, false, masked_embedding);
    infinilm::execution::submit<infini::ops::DeepseekV4MtpMaskEmbedding>(
        runtime,
        infini::ops::DeepseekV4MtpMaskEmbedding::DefaultConfig(
            runtime.device().type()),
        masked_embedding, positions);
    auto normalized_embedding = enorm_->forward(runtime, masked_embedding);
    auto embedding_projection = e_proj_->forward(runtime, normalized_embedding);
    auto embedding_stream = embedding_projection->view(
        {batch, seq, 1, hidden_size_});
    auto expanded_embedding = v4_broadcast_materialized(
        runtime, embedding_stream,
        {static_cast<int64_t>(batch), static_cast<int64_t>(seq),
         static_cast<int64_t>(hc_mult_), static_cast<int64_t>(hidden_size_)});

    auto flat_target = target_streams->view(
        {tokens * hc_mult_, hidden_size_});
    auto normalized_target = hnorm_->forward(runtime, flat_target);
    auto target_projection = h_proj_->forward(runtime, normalized_target)
        ->view({batch, seq, hc_mult_, hidden_size_});
    auto x = infinilm::runtime::Tensor::empty(
        runtime, target_projection->shape(), target_projection->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()),
        expanded_embedding, target_projection, x);

    auto [attn_input, attn_post, attn_comb] = hc_pre(
        runtime, x, hc_attn_fn_, hc_attn_scale_, hc_attn_base_,
        static_cast<int64_t>(hc_mult_), 20, norm_eps_, hc_eps_);
    auto [attn_packed, attn_scale] = rms_norm_dynamic_quant(
        runtime, attn_input, attn_norm_->weight(), attn_norm_->eps());
    auto attn_output = attn_->forward(
        positions, attn_input, forward_context, attn_packed, attn_scale);
    auto after_attn = hc_post(
        runtime, attn_output, x, attn_post, attn_comb);

    auto [ffn_input, ffn_post, ffn_comb] = hc_pre(
        runtime, after_attn, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_,
        static_cast<int64_t>(hc_mult_), 20, norm_eps_, hc_eps_);
    ffn_input = ffn_norm_->forward(runtime, ffn_input);
    auto ffn_output = ffn_->forward(
        runtime, ffn_input, input_ids, forward_context);
    auto output = hc_post(
        runtime, ffn_output, after_attn, ffn_post, ffn_comb);
    return output->view({batch, seq, hc_mult_ * hidden_size_});
}

infinilm::runtime::Tensor DeepseekV4MTP::logits(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states) const {
    if (!hidden_states || (hidden_states->ndim() != 3 && hidden_states->ndim() != 4)) {
        throw std::invalid_argument("DeepseekV4 MTP logits hidden state is invalid");
    }
    infinilm::runtime::Tensor streams;
    if (hidden_states->ndim() == 4) {
        streams = hidden_states;
    } else {
        const auto shape = hidden_states->shape();
        if (shape.back() != hc_mult_ * hidden_size_) {
            throw std::invalid_argument(
                "DeepseekV4 MTP logits hidden size mismatch");
        }
        streams = hidden_states->view(
            {shape[0], shape[1], hc_mult_, hidden_size_});
    }
    auto compressed = apply_hc_head(
        runtime, streams, hc_head_fn_, hc_head_scale_, hc_head_base_,
        hc_head_rms_weight_, norm_eps_);
    auto normalized = norm_->forward(runtime, compressed);
    return head_->forward(runtime, normalized);
}

DeepseekV4Model::DeepseekV4Model(infinilm::runtime::Runtime &runtime,
                                 std::shared_ptr<infinilm::config::ModelConfig> config,
                                 const engine::distributed::RankInfo &rank_info,
                                 backends::AttentionBackend backend,
                                 const layers::moe::EPConfig &ep_config,
                                 bool,
                                 std::shared_ptr<infinilm::layers::RoPE> rope,
                                 std::shared_ptr<infinilm::layers::RoPE> compressed_rope)
    {
    models::validate_runtime_construction(config, runtime, rank_info);
    config_ = config;
    const auto dtype = config->get_dtype();
    const size_t vocab = config->get<size_t>("vocab_size");
    const size_t hidden = config->get<size_t>("hidden_size");
    embed_tokens_ = bind_child("embed_tokens", std::make_shared<infinilm::layers::Embedding>(runtime, vocab, hidden, std::nullopt, dtype));
    const size_t layers = config->get<size_t>("num_hidden_layers");
    const auto &config_json = config->get_config_json();
    layers_.reserve(layers);
    for (size_t i = 0; i < layers; ++i) {
        size_t compress_ratio = 0;
        if (config_json.contains("compress_ratios") &&
            i < config_json["compress_ratios"].size()) {
            compress_ratio = config_json["compress_ratios"][i].get<size_t>();
        }
        auto layer_rope = compress_ratio > 1 ? compressed_rope : rope;
        auto layer = std::make_shared<DeepseekV4DecoderLayer>(
            runtime, config, i, rank_info, backend, ep_config,
            std::move(layer_rope));
        layers_.push_back(bind_child("layers." + std::to_string(i), std::move(layer)));
    }
    norm_ = bind_child("norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, config->get<double>("rms_norm_eps"), config->get_dtype()));
    hc_mult_ = config->get_or<size_t>("hc_mult", 4);
    hc_sinkhorn_iters_ = config->get_or<size_t>("hc_sinkhorn_iters", 20);
    hc_eps_ = config->get_or<double>("hc_eps", 1e-6);
    hc_head_fn_ = infinilm::weights::Weight(runtime, {hc_mult_, hc_mult_ * hidden}, infinilm::runtime::DataType::kFloat32);
    hc_head_base_ = infinilm::weights::Weight(runtime, {hc_mult_}, infinilm::runtime::DataType::kFloat32);
    hc_head_scale_ = infinilm::weights::Weight(runtime, {1}, infinilm::runtime::DataType::kFloat32);
    bind_weight("hc_head_fn", hc_head_fn_);
    bind_weight("hc_head_base", hc_head_base_);
    bind_weight("hc_head_scale", hc_head_scale_);
    hc_head_rms_weight_ = infinilm::runtime::Tensor::empty_parameter(
        runtime, {hc_mult_ * hidden}, infinilm::runtime::DataType::kFloat32);
    infinilm::execution::submit<infini::ops::Fill>(
        runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
        hc_head_rms_weight_, 1.0, hc_head_rms_weight_);

}

infinilm::runtime::Tensor DeepseekV4Model::forward(
    const infinilm::InfinilmModel::Input &input,
    infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto embedded = embed_tokens_->forward(runtime, input_ids);
    const auto shape = embedded->shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("DeepseekV4 embedding must be rank 3");
    }
    auto stream_view = embedded->view({shape[0], shape[1], 1, shape[2]});
    auto hidden_states = v4_broadcast_materialized(
        runtime, stream_view,
        {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1]),
         static_cast<int64_t>(hc_mult_), static_cast<int64_t>(shape[2])});
    v4_debug_stats(runtime, hidden_states, "embed");
    infinilm::runtime::Tensor residual;
    for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        hidden_states = layers_[layer_index]->forward(
            positions, input_ids, hidden_states, residual, forward_context);
        v4_debug_stats(runtime, hidden_states,
                       "layer." + std::to_string(layer_index));
    }
    return hidden_states->view(
        {shape[0], shape[1], hc_mult_ * shape[2]});
}

infinilm::runtime::Tensor DeepseekV4Model::head_hidden(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &target_hidden_states) const {
    if (!target_hidden_states || target_hidden_states->ndim() != 3) {
        throw std::invalid_argument(
            "DeepseekV4 target hidden state must be [batch, seq, 4 * hidden]");
    }
    const auto shape = target_hidden_states->shape();
    const size_t hidden = config_->get<size_t>("hidden_size");
    if (shape.back() != hc_mult_ * hidden) {
        throw std::invalid_argument("DeepseekV4 target hidden size mismatch");
    }
    auto streams = target_hidden_states->view(
        {shape[0], shape[1], hc_mult_, hidden});
    auto compressed = apply_hc_head(
        runtime, streams, hc_head_fn_, hc_head_scale_, hc_head_base_,
        hc_head_rms_weight_, config_->get<double>("rms_norm_eps"));
    v4_debug_stats(runtime, compressed, "hc_head");
    auto normalized = norm_->forward(runtime, compressed);
    v4_debug_stats(runtime, normalized, "final_norm");
    return normalized;
}

DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(infinilm::runtime::Runtime &runtime,
                                             std::shared_ptr<infinilm::config::ModelConfig> config,
                                             const engine::distributed::RankInfo &rank_info,
                                             backends::AttentionBackend backend,
                                             const layers::moe::EPConfig &ep_config,
                                             bool use_mla)
    : use_mla_(use_mla) {
    models::validate_runtime_construction(config, runtime, rank_info);
    model_config_ = config;
    rope_ = make_deepseek_v4_rope(runtime, config, false);
    compressed_rope_ = make_deepseek_v4_rope(runtime, config, true);
    model_ = bind_child("model", std::make_shared<DeepseekV4Model>(
        runtime, config, rank_info, backend, ep_config, use_mla, rope_,
        compressed_rope_));
    tp_rank_ = static_cast<infinilm::runtime::Size>(rank_info.tp_rank);
    tp_size_ = static_cast<infinilm::runtime::Size>(rank_info.tp_size);
    tp_comm_ = rank_info.comm;
    lm_head_ = bind_child(
        "lm_head",
        std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, config->get<size_t>("hidden_size"),
            config->get<size_t>("vocab_size"), dense_quant(), false,
            config->get_dtype(), tp_rank_, tp_size_));
    if (config->get<size_t>("num_nextn_predict_layers") > 0
        && std::getenv("INFINILM_DISABLE_MTP") == nullptr) {
        mtp_ = bind_child("mtp.0", std::make_shared<DeepseekV4MTP>(
            runtime, config, rank_info, backend, ep_config, rope_));
    }
}

infinilm::InfinilmModel::Output DeepseekV4ForCausalLM::forward(const Input &input, infinilm::global_state::ForwardContext &forward_context) const {
    for (auto &[ratio, state] : forward_context.deepseek_v4_sas_metadata) {
        if (state) {
            state->ready = false;
        }
    }
    for (auto &[ratio, state] :
             forward_context.deepseek_v4_compressor_metadata) {
        if (state) {
            state->ready = false;
        }
    }
    for (auto &[key, state] : forward_context.deepseek_v4_rope) {
        if (state) {
            state->gathered_this_forward = false;
        }
    }

    prepare_v4_qli_metadata_before_kernels(input, forward_context, model_config_);
    if (input.target_hidden_states && mtp_) {
        auto mtp_hidden = mtp_->forward(input.position_ids.value(), input.input_ids.value(),
                                        input.target_hidden_states.value(), forward_context);
        auto &runtime = forward_context.require_runtime();
        auto mtp_logits = mtp_->logits(runtime, mtp_hidden);
        v4_debug_topk(runtime, mtp_logits, "mtp");
        return {mtp_logits, mtp_hidden};
    }
    auto target_hidden = model_->forward(input, forward_context);
    auto &runtime = forward_context.require_runtime();
    if (!input.sample_all_positions
        && forward_context.attn_metadata.last_token_ids
        && forward_context.attn_metadata.last_token_ids.value()
        && target_hidden->size(0) == 1 && target_hidden->size(1) > 1) {
        const auto &indices =
            forward_context.attn_metadata.last_token_ids.value();
        const size_t requests = indices->size(0);
        const size_t hidden_width = target_hidden->size(2);
        auto selected = infinilm::runtime::Tensor::empty(
            runtime, {requests, hidden_width}, target_hidden->dtype());
        infinilm::execution::submit<infini::ops::Embedding>(
            runtime, infini::ops::Embedding::DefaultConfig(
                         runtime.device().type()),
            indices,
            target_hidden->view({target_hidden->numel() / hidden_width,
                                 hidden_width}),
            selected);
        target_hidden = selected->view({requests, 1, hidden_width});
    }
    auto hidden = model_->head_hidden(runtime, target_hidden);
    auto local_logits = lm_head_->forward(runtime, hidden);
    if (tp_size_ == 1) {
        v4_debug_topk(runtime, local_logits, "target");
        return {local_logits, target_hidden};
    }
    const auto &shape = local_logits->shape();
    const size_t batch = shape[0];
    const size_t seq = shape[1];
    const size_t local_vocab = shape[2];
    auto gathered = infinilm::runtime::Tensor::empty(
        runtime,
        {tp_size_ * batch * seq * local_vocab},
        local_logits->dtype());
    infinilm::engine::distributed::allgather_(
        runtime, gathered,
        local_logits->view({batch * seq * local_vocab}), tp_comm_);
    auto logits = gathered
        ->view({tp_size_, batch, seq, local_vocab})
        ->permute({1, 2, 0, 3})
        ->contiguous(runtime)
        ->view({batch, seq, tp_size_ * local_vocab});
    v4_debug_topk(runtime, logits, "target");
    return {logits, target_hidden};
}

void DeepseekV4ForCausalLM::reset_cache(const cache::CacheConfig *cache_config,
                                        global_state::ForwardContext &forward_context,
                                        backends::AttentionBackend backend,
                                        const engine::distributed::RankInfo &rank) {
    (void)use_mla_;
    if (cache_config == nullptr) {
        cache_config_.reset();
        forward_context.kv_cache_vec.clear();
        return;
    }
    auto next_config = cache_config->unique_copy();
    auto next_caches = allocate_cache_tensors(
        forward_context.require_runtime(), rank, cache_config, model_config_, backend);
    cache_config_ = std::move(next_config);
    forward_context.kv_cache_vec = std::move(next_caches);

    size_t max_batch_size = 1;
    if (const auto *paged_config =
            dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        max_batch_size = paged_config->max_batch_size();
    } else if (const auto *static_config =
            dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        max_batch_size = static_config->max_batch_size();
    }

    auto &runtime = forward_context.require_runtime();
    forward_context.deepseek_v4_qli_metadata.clear();
    for (size_t batch_size = 1; batch_size <= max_batch_size; ++batch_size) {
        auto state =
            std::make_shared<global_state::DeepseekV4QliMetadataState>();
        state->tensor = make_qli_metadata(
            runtime, batch_size, batch_size, batch_size);
        state->query_length = batch_size;
        state->pre_uploaded = true;
        forward_context.deepseek_v4_qli_metadata.emplace(
            batch_size, std::move(state));
    }
}

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(std::shared_ptr<infinilm::config::ModelConfig> config) {
    if (!config || config->get<std::string>("model_type") != "deepseek_v4") {
        throw std::invalid_argument("DeepseekV4 requires model_type=deepseek_v4");
    }
    auto &j = config->get_config_json();
    for (const char *key : {"hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads", "head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"}) {
        if (config->get<int64_t>(key) <= 0) throw std::invalid_argument(std::string("DeepseekV4 requires positive ") + key);
    }
    j["num_key_value_heads"] = 1;
    j["intermediate_size"] = j["moe_intermediate_size"];
    j["num_experts"] = j["n_routed_experts"];
    j["mlp_bias"] = false;
    j["attention_bias"] = false;
    j["attention_output_bias"] = false;
    if (!j.contains("dtype") && j.contains("torch_dtype")) j["dtype"] = j["torch_dtype"];
    j["quantization_config"] = {
        {"quant_method", "compressed-tensors"},
        {"format", "int-quantized"},
        {"config_groups", {{"group_0", {
            {"weights", {{"num_bits", 8}, {"type", "int"}, {"strategy", "channel"}, {"symmetric", true}, {"dynamic", false}}},
            {"input_activations", {{"num_bits", 8}, {"type", "int"}, {"strategy", "token"}, {"symmetric", true}, {"dynamic", true}}}
        }}}}
    };
    if (!j.contains("num_hash_layers")) j["num_hash_layers"] = 3;
    if (!j.contains("num_nextn_predict_layers")) j["num_nextn_predict_layers"] = 1;
    return config;
}

} // namespace infinilm::models::deepseek_v4

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    deepseek_v4,
    infinilm::models::deepseek_v4::DeepseekV4ForCausalLM,
    infinilm::models::deepseek_v4::create_deepseek_v4_model_config);
}
