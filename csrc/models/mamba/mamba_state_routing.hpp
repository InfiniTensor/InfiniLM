#pragma once

// Mamba-local host mapping only. This is not a Runtime, kernel or cache framework.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::mamba::state_routing {

struct Plan {
    size_t requests{};
    size_t length{};
    bool reshape_packed{false};
    bool indexed{false};
    std::vector<size_t> initial;
    std::vector<size_t> final;
};

inline Plan make_plan(size_t batch, size_t length, size_t pool,
                      const std::optional<std::vector<int64_t>> &offsets,
                      const std::optional<std::vector<int64_t>> &initial,
                      const std::optional<std::vector<int64_t>> &final) {
    if (length && batch > std::numeric_limits<size_t>::max() / length) {
        throw std::invalid_argument("Mamba token count overflow");
    }
    Plan plan{batch, length, false, bool(initial || final), {}, {}};
    if (offsets) {
        if (offsets->empty() || offsets->front() != 0) {
            throw std::invalid_argument("Mamba offsets must start at zero");
        }
        const size_t requests = offsets->size() - 1;
        const size_t tokens = batch * length;
        if (offsets->back() < 0 || static_cast<uint64_t>(offsets->back()) != tokens) {
            throw std::invalid_argument("Mamba offsets must cover every input token");
        }
        // Existing scan is rectangular. Equal-length packed requests can be
        // represented by a view; ragged requests are not supported by this path.
        const size_t request_length = requests ? tokens / requests : 0;
        if ((requests == 0 && tokens != 0) || (requests && tokens % requests)) {
            throw std::invalid_argument("Mamba ragged packed input is not supported");
        }
        for (size_t i = 0; i <= requests; ++i) {
            if ((*offsets)[i] < 0 || static_cast<uint64_t>((*offsets)[i]) != i * request_length) {
                throw std::invalid_argument("Mamba ragged packed input is not supported");
            }
        }
        if (batch != requests) {
            if (batch != 1) {
                throw std::invalid_argument("Mamba packed input must have one physical batch row");
            }
            plan.reshape_packed = true;
        } else if (length != request_length) {
            throw std::invalid_argument("Mamba rectangular offsets mismatch");
        }
        plan.requests = requests;
        plan.length = request_length;
    }
    auto resolve = [&](const std::optional<std::vector<int64_t>> &indices) {
        if (indices && indices->size() != plan.requests) {
            throw std::invalid_argument("Mamba state indices must match request count");
        }
        std::vector<size_t> result;
        result.reserve(plan.requests);
        for (size_t i = 0; i < plan.requests; ++i) {
            if (!indices && i >= pool) {
                throw std::invalid_argument("Mamba prefix pool too small");
            }
            if (indices && ((*indices)[i] < 0 || static_cast<uint64_t>((*indices)[i]) >= pool)) {
                throw std::invalid_argument("Mamba state index out of range");
            }
            result.push_back(indices ? static_cast<size_t>((*indices)[i]) : i);
        }
        return result;
    };
    plan.initial = resolve(initial);
    plan.final = resolve(final);
    // Duplicate readers are valid forks. Duplicate writers have no defined
    // simultaneous result and are rejected before any recurrent operation.
    for (size_t i = 0; i < plan.final.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (plan.final[i] == plan.final[j]) {
                throw std::invalid_argument("Mamba final state indices must be unique");
            }
        }
    }
    return plan;
}

template <class CopyRow>
inline void gather(const Plan &plan, CopyRow copy_row) {
    for (size_t i = 0; i < plan.requests; ++i) {
        copy_row(i, plan.initial[i]);
    }
}

template <class CopyRow>
inline void commit(const Plan &plan, CopyRow copy_row) {
    // An empty request has the same final state as its initial state. Copying
    // it to a distinct final slot is intentional and uses the prior snapshot.
    for (size_t i = 0; i < plan.requests; ++i) {
        copy_row(plan.final[i], i);
    }
}

} // namespace infinilm::models::mamba::state_routing
