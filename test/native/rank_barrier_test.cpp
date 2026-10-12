#ifdef NDEBUG
#undef NDEBUG // Native test assertions must run in release builds too.
#endif

#include "rank_barrier.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using infinilm::engine::RankBarrier;

namespace {

void test_reusable_barrier() {
    RankBarrier barrier(3);
    std::atomic<int> passed{0};
    std::vector<std::thread> threads;
    for (int rank = 0; rank < 3; ++rank) {
        threads.emplace_back([&] {
            barrier.wait();
            ++passed;
            barrier.wait();
            ++passed;
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    assert(passed.load() == 6);
}

void test_abort_wakes_waiters_and_preserves_first_error() {
    RankBarrier barrier(3);
    std::atomic<int> started{0};

    auto waiter = [&] {
        ++started;
        try {
            barrier.wait();
        } catch (const std::runtime_error &error) {
            return std::string(error.what());
        }
        return std::string("no error");
    };

    auto first_waiter = std::async(std::launch::async, waiter);
    auto second_waiter = std::async(std::launch::async, waiter);
    while (started.load() != 2) {
        std::this_thread::yield();
    }

    barrier.abort(std::make_exception_ptr(std::runtime_error("rank 1 failed")));
    barrier.abort(std::make_exception_ptr(std::runtime_error("later failure")));

    using namespace std::chrono_literals;
    assert(first_waiter.wait_for(2s) == std::future_status::ready);
    assert(second_waiter.wait_for(2s) == std::future_status::ready);
    assert(first_waiter.get() == "rank 1 failed");
    assert(second_waiter.get() == "rank 1 failed");

    try {
        barrier.rethrow_if_broken();
        assert(false && "broken barrier must rethrow");
    } catch (const std::runtime_error &error) {
        assert(std::string(error.what()) == "rank 1 failed");
    }
}

void test_zero_ranks_rejected() {
    try {
        RankBarrier barrier(0);
        (void)barrier;
        assert(false && "zero-rank barrier must be rejected");
    } catch (const std::invalid_argument &) {
    }
}

} // namespace

int main() {
    test_reusable_barrier();
    test_abort_wakes_waiters_and_preserves_first_error();
    test_zero_ranks_rejected();
    return 0;
}
