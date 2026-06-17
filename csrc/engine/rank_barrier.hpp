#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>

namespace infinilm::engine {
class RankBarrier {
public:
    explicit RankBarrier(size_t nranks);

    void wait(const char *label = nullptr, int tp_rank = -1);

private:
    struct SubBarrier {
        size_t generation{0};
        size_t arrived{0};
    };

    SubBarrier &sub_for_(const char *label);

    const size_t thread_count_;
    std::mutex mutex_;
    std::condition_variable cv_;
    SubBarrier default_;
    std::unordered_map<std::string, SubBarrier> keyed_;
};
} // namespace infinilm::engine
