#pragma once

#include <condition_variable>
#include <mutex>

namespace infinilm::engine {
class RankBarrier {
public:
    explicit RankBarrier(size_t nranks);

    void wait();

private:
    const size_t thread_count_;
    size_t arrived_;
    size_t generation_;
    std::mutex mutex_;
    std::condition_variable cv_;
};
} // namespace infinilm::engine
