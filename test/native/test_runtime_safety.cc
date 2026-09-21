#include "context/allocators/pinnable_block_allocator.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/memory.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/common/dispatcher.hpp"

#include <cassert>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace {
int plans = 0;
int cleanups = 0;
int runs = 0;

template <class Function>
void expect_failure(Function function) {
    bool threw = false;
    try {
        function();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw);
}

void *plan(infinicore::Device::Type, int failure) {
    ++plans;
    if (failure == 2) {
        throw std::runtime_error("plan failure");
    }
    return new int(failure);
}

void run(void *meta) {
    ++runs;
    if (*static_cast<int *>(meta) == 4) {
        throw std::runtime_error("execution failure");
    }
}

void cleanup(void **meta) {
    assert(*meta != nullptr);
    ++cleanups;
    delete static_cast<int *>(*meta);
    *meta = nullptr;
}
} // namespace

namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(TestGraphOperator, Device::Type, int);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(TestGraphOperator);

TestGraphOperator::TestGraphOperator(Device::Type device, int failure) {
    if (failure == 1) {
        throw std::runtime_error("validation failure");
    }
    INFINICORE_GRAPH_OP_DISPATCH(device, device, failure);
    if (failure == 3) {
        throw std::runtime_error("post-plan failure");
    }
}
} // namespace infinicore::op

int main() {
    using infinicore::Device;
    using infinicore::Memory;
    using infinicore::PinnableBlockAllocator;
    using infinicore::op::TestGraphOperator;
    using Cache = infinicore::common::LRUCache<int, int *>;

    static_assert(!std::is_copy_constructible_v<Memory>);
    static_assert(!std::is_copy_assignable_v<Memory>);
    static_assert(!std::is_move_constructible_v<Memory>);
    static_assert(!std::is_move_assignable_v<Memory>);
    static_assert(!std::is_copy_constructible_v<Cache>);
    static_assert(!std::is_copy_assignable_v<Cache>);
    static_assert(!std::is_move_constructible_v<Cache>);
    static_assert(!std::is_copy_constructible_v<TestGraphOperator>);

    const Device cpu(Device::Type::kCpu, 0);
    infinicore::context::setDevice(cpu);
    int releases = 0;
    {
        auto memory = std::make_shared<Memory>(
            new std::byte[8], 8, cpu, [&](std::byte *ptr) {
                ++releases;
                delete[] ptr;
            });
        auto shared = memory;
        memory.reset();
        assert(releases == 0);
        assert(shared->size() == 8);
    }
    assert(releases == 1);

    alignas(TestGraphOperator) std::byte storage[sizeof(TestGraphOperator)];
    std::memset(storage, 0xa5, sizeof(storage));
    expect_failure([&]() { new (storage) TestGraphOperator(cpu.type(), 1); });
    expect_failure([&]() { TestGraphOperator op(cpu.type(), 0); });
    TestGraphOperator::plan_dispatcher().registerDevice(cpu.type(), plan);
    expect_failure([&]() { TestGraphOperator op(cpu.type(), 0); });
    TestGraphOperator::run_dispatcher().registerDevice(cpu.type(), run);
    expect_failure([&]() { TestGraphOperator op(cpu.type(), 0); });
    assert(plans == 0 && cleanups == 0);
    TestGraphOperator::cleanup_dispatcher().registerDevice(cpu.type(), cleanup);
    expect_failure([&]() { TestGraphOperator op(cpu.type(), 2); });
    assert(plans == 1 && cleanups == 0);
    expect_failure([&]() { TestGraphOperator op(cpu.type(), 3); });
    assert(plans == 2 && cleanups == 1);
    {
        TestGraphOperator op(cpu.type(), 0);
        op.run();
        assert(runs == 1 && cleanups == 1);
    }
    assert(cleanups == 2);
    expect_failure([&]() {
        TestGraphOperator op(cpu.type(), 4);
        op.run();
    });
    assert(plans == 4 && cleanups == 3 && runs == 2);

    int deleted = 0;
    {
        infinicore::op::common::OpCache<int, int *> caches(
            2, [&](int *&value) {
                ++deleted;
                delete value;
            });
        auto &first = caches.getCache(cpu);
        first.put(1, new int(10));
        caches.getCache(cpu.type(), 8);
        assert(&first == &caches.getCache(cpu));
        assert(**first.get(1) == 10);
        first.put(2, new int(20));
        first.put(3, new int(30));
        assert(!first.contains(1));
        assert(deleted == 1);
        caches.clear();
        assert(deleted == 3);
    }
    assert(deleted == 3);

    PinnableBlockAllocator allocator(cpu);
    auto *first = allocator.allocate(16);
    auto *second = allocator.allocate(16);
    allocator.begin_pin_mode();
    allocator.retain_for_capture(first);
    auto lease = allocator.commit_pin_mode();
    allocator.deallocate(first);
    allocator.deallocate(second);
    allocator.mark_in_use_(first, true);
    assert(allocator.allocate(16) == second);
    allocator.deallocate(second);
    allocator.trim();
    allocator.mark_in_use_(first, false);
    allocator.trim();
    assert(allocator.allocate(16) == first);
    allocator.deallocate(first);
    for (int i = 0; i < 100; ++i) {
        allocator.mark_in_use_(first, true);
        allocator.deallocate(first);
    }
    assert(allocator.allocate(16) == first);
    auto *distinct = allocator.allocate(16);
    assert(distinct != first);
    allocator.deallocate(first);
    allocator.deallocate(distinct);
    lease.reset();
    allocator.trim();
    expect_failure([&]() { allocator.mark_in_use_(first, true); });
}
