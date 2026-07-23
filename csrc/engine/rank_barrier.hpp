#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>

namespace infinilm::engine {
class RankBarrier {
public:
    explicit RankBarrier(size_t nranks);

    bool wait(bool success = true);

private:
    const size_t thread_count_;
    size_t arrived_;
    size_t generation_;
    bool generation_success_{true};
    std::vector<bool> completed_results_;
    std::mutex mutex_;
    std::condition_variable cv_;
};
} // namespace infinilm::engine
