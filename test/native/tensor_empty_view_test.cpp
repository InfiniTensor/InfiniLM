#include "runtime/tensor.hpp"

#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace infinilm::runtime;

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("tensor empty view assertion failed");
    }
}

template <class Error, class Function>
void expect_error(Function function) {
    try {
        function();
    } catch (const Error &) { return; }
    throw std::runtime_error("expected view exception was not raised");
}

void run() {
    const Device cpu{Device::Type::kCpu, 0};
    auto owner = std::make_shared<std::vector<float>>(8, 17.0F);
    std::weak_ptr<std::vector<float>> weak_owner = owner;
    auto base = Tensor::from_blob(owner->data(), {8}, DataType::kFloat32, cpu, owner);
    auto *end = reinterpret_cast<std::byte *>(owner->data() + 8);
    auto empty = base->narrow({{0, 8, 0}});
    auto flat = empty->view({0});
    auto multi = empty->view({2, 0, 3, 0});
    require(flat->data() == end && multi->data() == end);
    require(flat->strides() == Strides{1});
    require(multi->strides() == Strides({3, 3, 1, 1}));
    require(multi->numel() == 0 && multi->nbytes() == 0);
    expect_error<std::invalid_argument>([&] { (void)empty->view({}); });
    expect_error<std::invalid_argument>([&] { (void)empty->view({1}); });
    const auto max_size = std::numeric_limits<Size>::max();
    expect_error<std::length_error>([&] { (void)empty->view({max_size, 2}); });
    expect_error<std::length_error>([&] { (void)empty->view({0, max_size}); });
    owner.reset();
    base.reset();
    empty.reset();
    flat.reset();
    require(!weak_owner.expired());
    multi.reset();
    require(weak_owner.expired());

    // Preserve custom strides when the empty shape is unchanged.
    auto custom = Tensor::strided_from_blob(nullptr, {0, 3}, {9, 2}, DataType::kFloat32, cpu);
    require(custom->view({0, 3})->strides() == Strides({9, 2}));
    require(custom->view({0})->numel() == 0);

    float value = 31.0F;
    auto scalar = Tensor::from_blob(&value, {}, DataType::kFloat32, cpu);
    auto singleton = scalar->view({1, 1});
    require(singleton->strides() == Strides({1, 1}));
    require(singleton->data() == reinterpret_cast<std::byte *>(&value));
    require(singleton->view({})->shape().empty());
    require(scalar->view({})->strides().empty());
    expect_error<std::invalid_argument>([&] { (void)scalar->view({0}); });
    expect_error<std::invalid_argument>([&] { (void)scalar->view({2}); });

    std::vector<float> values(12);
    auto ordinary = Tensor::from_blob(values.data(), {3, 4}, DataType::kFloat32, cpu);
    auto reshaped = ordinary->view({2, 6});
    require(reshaped->strides() == Strides({6, 1}));
    require(reshaped->data() == ordinary->data());
    auto offset = ordinary->narrow({{0, 1, 2}})->view({8});
    require(offset->data() == reinterpret_cast<std::byte *>(values.data() + 4));
    expect_error<std::invalid_argument>([&] { (void)ordinary->view({11}); });
    expect_error<std::length_error>([&] { (void)ordinary->view({max_size, max_size}); });
    auto transposed = ordinary->permute({1, 0});
    expect_error<std::runtime_error>([&] { (void)transposed->view({12}); });
}
} // namespace

int main() {
    try {
        run();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "tensor_empty_view_test: %s\n", error.what());
        return 1;
    }
    std::puts("tensor_empty_view_test: PASS");
    return 0;
}
