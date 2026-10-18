#pragma once

#include "../global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace infinilm::engine {

// Validate the rectangular, ordinary static-cache prefill entirely on host,
// before any layer can write KV. Packed/nonuniform requests are not this path.
inline global_state::StaticPrefillHostDescription describe_static_prefill(
    size_t batch, size_t query_length, size_t capacity, size_t page_size,
    const infinilm::runtime::Tensor &past_cpu, const infinilm::runtime::Tensor &total_cpu,
    const std::optional<infinilm::runtime::Tensor> &offsets_cpu,
    const std::optional<infinilm::runtime::Tensor> &cu_seqlens_cpu,
    const std::optional<infinilm::runtime::Tensor> &rows_cpu,
    const std::optional<infinilm::runtime::Tensor> &slots_cpu) {
    using infinilm::runtime::DataType;
    using infinilm::runtime::Device;
    const auto cpu_int32 = [](const infinilm::runtime::Tensor &t, size_t n) {
        return t && t->device().type() == Device::Type::kCpu
            && t->dtype() == DataType::kInt32 && t->ndim() == 1
            && t->size(0) == n && t->is_contiguous();
    };
    if (capacity == 0 || page_size == 0 || query_length <= 1
        || (batch != 1 && batch != capacity) || batch > capacity
        || batch > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        || query_length > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / batch
        || page_size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / capacity
        || !cpu_int32(past_cpu, batch) || !cpu_int32(total_cpu, batch)
        || !offsets_cpu || !cpu_int32(*offsets_cpu, batch + 1)) {
        throw std::invalid_argument("static prefill requires rectangular CPU metadata and ordinary cache bounds");
    }
    if (cu_seqlens_cpu && !cpu_int32(*cu_seqlens_cpu, batch + 1)) {
        throw std::invalid_argument("static prefill cu_seqlens require CPU int32[batch+1]");
    }
    if (rows_cpu && (!*rows_cpu || (*rows_cpu)->device().type() != Device::Type::kCpu || (*rows_cpu)->dtype() != DataType::kInt32 || (*rows_cpu)->ndim() != 2 || (*rows_cpu)->size(0) != batch || (*rows_cpu)->size(1) != 1 || !(*rows_cpu)->is_contiguous())) {
        throw std::invalid_argument("static prefill rows require CPU int32[batch,1]");
    }
    if (batch == 1 && capacity > 1 && !rows_cpu) {
        throw std::invalid_argument("static prefill into a larger cache requires an explicit physical row");
    }
    if (slots_cpu && (!*slots_cpu || (*slots_cpu)->device().type() != Device::Type::kCpu || (*slots_cpu)->dtype() != DataType::kInt64 || (*slots_cpu)->ndim() != 1 || (*slots_cpu)->size(0) != batch * query_length || !(*slots_cpu)->is_contiguous())) {
        throw std::invalid_argument("static prefill slots require CPU int64[batch*query]");
    }
    const auto *past = reinterpret_cast<const int32_t *>(past_cpu->data());
    const auto *total = reinterpret_cast<const int32_t *>(total_cpu->data());
    const auto *offsets = reinterpret_cast<const int32_t *>((*offsets_cpu)->data());
    const auto *cu = cu_seqlens_cpu
                       ? reinterpret_cast<const int32_t *>((*cu_seqlens_cpu)->data())
                       : nullptr;
    const auto *rows = rows_cpu
                         ? reinterpret_cast<const int32_t *>((*rows_cpu)->data())
                         : nullptr;
    const auto *slots = slots_cpu
                          ? reinterpret_cast<const int64_t *>((*slots_cpu)->data())
                          : nullptr;
    if (offsets[0] != 0 || (cu && cu[0] != 0)) {
        throw std::invalid_argument("static prefill offsets must start at zero");
    }
    for (size_t i = 0; i < batch; ++i) {
        const auto row = rows ? rows[i] : static_cast<int32_t>(i);
        if (row < 0 || static_cast<size_t>(row) >= capacity
            || (batch == capacity && static_cast<size_t>(row) != i)
            || past[i] < 0 || static_cast<size_t>(past[i]) > page_size
            || query_length > page_size - static_cast<size_t>(past[i])
            || query_length > static_cast<size_t>(std::numeric_limits<int32_t>::max() - past[i])
            || total[i] != past[i] + static_cast<int32_t>(query_length)
            || (batch > 1 && past[i] != past[0])
            || offsets[i + 1] != static_cast<int32_t>((i + 1) * query_length)
            || (cu && static_cast<size_t>(total[i]) > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / batch)
            || (cu && cu[i + 1] != static_cast<int32_t>((i + 1) * static_cast<size_t>(total[i])))) {
            throw std::invalid_argument("static prefill requires identity/full or one valid row and uniform lengths");
        }
        for (size_t t = 0; slots && t < query_length; ++t) {
            const auto expected = static_cast<int64_t>(static_cast<size_t>(row) * page_size
                                                       + static_cast<size_t>(past[i]) + t);
            if (slots[i * query_length + t] != expected) {
                throw std::invalid_argument("static prefill slot does not match row, past and query position");
            }
        }
    }
    return {batch, query_length, capacity, page_size,
            static_cast<size_t>(rows ? rows[0] : 0),
            static_cast<size_t>(past[0]), static_cast<size_t>(total[0])};
}

// One ordinary static-cache row is one logical page. Owned by the worker so
// both metadata tensors keep their device addresses across eager decode calls.
class StaticDecodeMetadata {
public:
    void reset() noexcept {
        runtime_ = nullptr;
        query_batch_size_ = 0;
        cache_capacity_ = 0;
        page_size_ = 0;
        block_tables_.reset();
        slot_mapping_.reset();
    }

    void prepare(infinilm::runtime::Runtime &runtime, size_t query_batch_size,
                 size_t cache_capacity, size_t page_size) {
        runtime.assertOwnerThread();
        if (runtime.isGraphRecording()) {
            throw std::logic_error("static decode metadata cannot be prepared during graph recording");
        }
        if (runtime_ != nullptr && runtime_ != &runtime) {
            throw std::logic_error("static decode metadata belongs to another Runtime");
        }
        if (query_batch_size == 0 || cache_capacity == 0
            || query_batch_size > cache_capacity || page_size == 0
            || cache_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())
            || page_size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / cache_capacity) {
            throw std::invalid_argument("invalid static decode batch, cache capacity or page size");
        }
        if (runtime_ == &runtime && query_batch_size_ == query_batch_size
            && cache_capacity_ == cache_capacity && page_size_ == page_size) {
            return;
        }

        std::vector<int32_t> blocks(query_batch_size);
        std::vector<int64_t> slots(query_batch_size);
        for (size_t i = 0; i < query_batch_size; ++i) {
            blocks[i] = static_cast<int32_t>(i);
            slots[i] = static_cast<int64_t>(i * page_size);
        }
        auto block_tables = infinilm::runtime::Tensor::empty(
            runtime, {query_batch_size, 1}, infinilm::runtime::DataType::kInt32);
        auto slot_mapping = infinilm::runtime::Tensor::empty(
            runtime, {query_batch_size}, infinilm::runtime::DataType::kInt64);
        runtime.memcpyH2D(block_tables->data(), blocks.data(), blocks.size() * sizeof(int32_t), false);
        runtime.memcpyH2D(slot_mapping->data(), slots.data(), slots.size() * sizeof(int64_t), false);

        block_tables_ = std::move(block_tables);
        slot_mapping_ = std::move(slot_mapping);
        runtime_ = &runtime;
        query_batch_size_ = query_batch_size;
        cache_capacity_ = cache_capacity;
        page_size_ = page_size;
    }

    global_state::StaticDecodeHostDescription update(
        infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &past_cpu,
        const infinilm::runtime::Tensor &total_cpu,
        const std::optional<infinilm::runtime::Tensor> &rows_cpu = std::nullopt,
        const std::optional<infinilm::runtime::Tensor> &slots_cpu = std::nullopt) {
        runtime.assertOwnerThread();
        if (runtime_ != &runtime || !block_tables_ || !slot_mapping_) {
            throw std::logic_error("static decode metadata is not prepared for this Runtime");
        }
        if (runtime.isGraphRecording()) {
            throw std::logic_error("static decode metadata cannot be updated during graph recording");
        }
        const auto valid_lengths = [this](const infinilm::runtime::Tensor &tensor) {
            return tensor && tensor->device().type() == infinilm::runtime::Device::Type::kCpu
                && tensor->dtype() == infinilm::runtime::DataType::kInt32
                && tensor->ndim() == 1 && tensor->size(0) == query_batch_size_
                && tensor->is_contiguous();
        };
        if (!valid_lengths(past_cpu) || !valid_lengths(total_cpu)) {
            throw std::invalid_argument("static decode lengths require contiguous CPU int32[batch]");
        }
        if (rows_cpu && (!*rows_cpu || (*rows_cpu)->device().type() != infinilm::runtime::Device::Type::kCpu || (*rows_cpu)->dtype() != infinilm::runtime::DataType::kInt32 || (*rows_cpu)->ndim() != 2 || (*rows_cpu)->size(0) != query_batch_size_ || (*rows_cpu)->size(1) != 1 || !(*rows_cpu)->is_contiguous())) {
            throw std::invalid_argument("static decode rows require contiguous CPU int32[batch,1]");
        }
        if (slots_cpu && (!*slots_cpu || (*slots_cpu)->device().type() != infinilm::runtime::Device::Type::kCpu || (*slots_cpu)->dtype() != infinilm::runtime::DataType::kInt64 || (*slots_cpu)->ndim() != 1 || (*slots_cpu)->size(0) != query_batch_size_ || !(*slots_cpu)->is_contiguous())) {
            throw std::invalid_argument("static decode slots require contiguous CPU int64[batch]");
        }

        const auto *past = reinterpret_cast<const int32_t *>(past_cpu->data());
        const auto *total = reinterpret_cast<const int32_t *>(total_cpu->data());
        const auto *rows = rows_cpu
                             ? reinterpret_cast<const int32_t *>((*rows_cpu)->data())
                             : nullptr;
        const auto *given_slots = slots_cpu
                                    ? reinterpret_cast<const int64_t *>((*slots_cpu)->data())
                                    : nullptr;
        std::vector<int32_t> blocks(query_batch_size_);
        std::vector<int64_t> slots(query_batch_size_);
        std::unordered_set<int32_t> seen_rows;
        global_state::StaticDecodeHostDescription description;
        for (size_t i = 0; i < query_batch_size_; ++i) {
            const int32_t row = rows ? rows[i] : static_cast<int32_t>(i);
            if (row < 0 || static_cast<size_t>(row) >= cache_capacity_
                || !seen_rows.insert(row).second) {
                throw std::invalid_argument("static decode rows must be unique and inside cache capacity");
            }
            if (past[i] < 0 || static_cast<size_t>(past[i]) >= page_size_
                || past[i] == std::numeric_limits<int32_t>::max()
                || total[i] != past[i] + 1) {
                throw std::invalid_argument("static decode requires total length = past + 1 inside its cache row");
            }
            blocks[i] = row;
            slots[i] = static_cast<int64_t>(static_cast<size_t>(row) * page_size_
                                            + static_cast<size_t>(past[i]));
            if (given_slots && given_slots[i] != slots[i]) {
                throw std::invalid_argument("static decode slot does not match its row and past length");
            }
            description.identity_rows &= row == static_cast<int32_t>(i);
            description.uniform_lengths &= past[i] == past[0];
        }
        // All host inputs are validated before either stable device buffer changes.
        runtime.memcpyH2D(block_tables_->data(), blocks.data(), blocks.size() * sizeof(int32_t), false);
        runtime.memcpyH2D(slot_mapping_->data(), slots.data(), slots.size() * sizeof(int64_t), false);
        return description;
    }

    const infinilm::runtime::Tensor &block_tables() const {
        if (!block_tables_) {
            throw std::logic_error("static decode block tables are not prepared");
        }
        return block_tables_;
    }

    const infinilm::runtime::Tensor &slot_mapping() const {
        if (!slot_mapping_) {
            throw std::logic_error("static decode slot mapping is not prepared");
        }
        return slot_mapping_;
    }

private:
    infinilm::runtime::Runtime *runtime_ = nullptr;
    size_t query_batch_size_ = 0;
    size_t cache_capacity_ = 0;
    size_t page_size_ = 0;
    infinilm::runtime::Tensor block_tables_;
    infinilm::runtime::Tensor slot_mapping_;
};

} // namespace infinilm::engine
