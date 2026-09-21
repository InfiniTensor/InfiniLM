#include "infinicore/context/context.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/paged_attention_prefill.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace {
using infinicore::DataType;
using infinicore::Device;
using infinicore::Tensor;

int calls = 0;
int query_maximum = 0;
int key_maximum = 0;
std::vector<int32_t> key_offsets;

Tensor metadata(std::initializer_list<int32_t> values) {
    auto tensor = Tensor::empty({values.size()}, DataType::kInt32, Device::Type::kCpu);
    std::copy(values.begin(), values.end(), reinterpret_cast<int32_t *>(tensor->data()));
    return tensor;
}

template <class Function>
void expect_failure(Function function) {
    const auto previous_calls = calls;
    bool threw = false;
    try {
        function();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw);
    assert(calls == previous_calls);
}
} // namespace

// Inspect the production adapter's arguments without launching attention kernels.
namespace infinicore::op {
void mha_varlen_(Tensor,
                 const Tensor &,
                 const Tensor &,
                 const Tensor &,
                 const Tensor &,
                 const Tensor &cum_seqlens_k,
                 std::optional<Tensor> block_table,
                 int max_seqlen_q,
                 int max_seqlen_k,
                 std::optional<Tensor>,
                 float) {
    assert(block_table.has_value());
    ++calls;
    query_maximum = max_seqlen_q;
    key_maximum = max_seqlen_k;
    const auto *offsets = reinterpret_cast<const int32_t *>(cum_seqlens_k->data());
    key_offsets.assign(offsets, offsets + cum_seqlens_k->size(0));
}
} // namespace infinicore::op

int main() {
    using infinicore::op::paged_attention_prefill;
    using infinicore::op::paged_attention_prefill_;
    const Device cpu(Device::Type::kCpu);
    infinicore::context::setDevice(cpu);

    auto q = Tensor::empty({5, 4, 3}, DataType::kFloat32, cpu);
    auto k = Tensor::empty({4, 4, 2, 3}, DataType::kFloat32, cpu);
    auto v = Tensor::empty({4, 4, 2, 6}, DataType::kFloat32, cpu);
    auto blocks = Tensor::empty({2, 8}, DataType::kInt32, cpu);
    auto lengths = metadata({4, 7});
    auto offsets = metadata({0, 2, 5});
    auto invoke = [&](Tensor kv_lengths, Tensor query_offsets) {
        return paged_attention_prefill(q, k, v, blocks, kv_lengths, query_offsets, std::nullopt, 1.0f);
    };

    auto output = invoke(lengths, offsets);
    assert(output->shape() == infinicore::Shape({5, 4, 6}));
    assert(calls == 1);
    assert(query_maximum == 3 && key_maximum == 7);
    assert(key_offsets == std::vector<int32_t>({0, 4, 11}));

    invoke(metadata({0, 5}), metadata({0, 0, 5}));
    assert(query_maximum == 5 && key_maximum == 5);
    assert(key_offsets == std::vector<int32_t>({0, 0, 5}));

    expect_failure([&] { invoke(lengths, metadata({1, 2, 5})); });
    expect_failure([&] { invoke(lengths, metadata({0, 4, 3})); });
    expect_failure([&] { invoke(lengths, metadata({0, 2, 4})); });
    expect_failure([&] { invoke(lengths, metadata({0, 2, 6})); });
    expect_failure([&] { invoke(lengths, metadata({0, 5})); });
    expect_failure([&] { invoke(metadata({-1, 7}), offsets); });
    expect_failure([&] { invoke(metadata({4, 33}), offsets); });
    expect_failure([&] {
        invoke(lengths, Tensor::empty({3}, DataType::kInt64, cpu));
    });
    expect_failure([&] {
        invoke(Tensor::empty({2}, DataType::kFloat32, cpu), offsets);
    });
    expect_failure([&] {
        invoke(lengths, Tensor::strided_empty({3}, {2}, DataType::kInt32, cpu));
    });
    expect_failure([&] {
        invoke(Tensor::strided_empty({2}, {2}, DataType::kInt32, cpu), offsets);
    });
    expect_failure([&] {
        auto wrong_output = Tensor::empty(q->shape(), q->dtype(), cpu);
        paged_attention_prefill_(wrong_output, q, k, v, blocks, lengths, offsets, std::nullopt, 1.0f);
    });
    expect_failure([&] {
        auto wrong_blocks = Tensor::empty({1, 8}, DataType::kInt32, cpu);
        paged_attention_prefill(q, k, v, wrong_blocks, lengths, offsets, std::nullopt, 1.0f);
    });
    expect_failure([&] {
        auto wrong_v = Tensor::empty(v->shape(), DataType::kFloat16, cpu);
        paged_attention_prefill(q, k, wrong_v, blocks, lengths, offsets, std::nullopt, 1.0f);
    });
}
