#pragma once

#include "runtime/tensor.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {

struct RequestRows {
    size_t first;
    size_t count;

    runtime::Tensor slice(const runtime::Tensor &tensor) const {
        if (!tensor || count == tensor->size(0)) {
            return tensor;
        }
        return tensor->narrow({{0, first, count}});
    }
};

template <typename Run>
void for_each_request(const std::vector<size_t> &offsets, Run &&run) {
    for (size_t i = 1; i < offsets.size(); ++i) {
        const RequestRows rows{offsets[i - 1], offsets[i] - offsets[i - 1]};
        if (rows.count != 0) {
            run(rows);
        }
    }
}

// Apply a shape-sensitive projection with the same row count as a standalone
// request and restore the scheduler's packed row order for downstream kernels.
template <typename Project>
runtime::Tensor project_requests(runtime::Runtime &rt, const std::vector<size_t> &offsets,
                                 Project &&project) {
    const size_t total = offsets.back();
    runtime::Tensor output;
    for_each_request(offsets, [&](const RequestRows &rows) {
        auto projected = project(rows);
        if (rows.count == total) {
            output = std::move(projected);
            return;
        }
        if (!output) {
            auto shape = projected->shape();
            shape[0] = total;
            output = runtime::Tensor::empty(rt, shape, projected->dtype());
        }
        rows.slice(output)->copy_from(rt, projected);
    });
    return output;
}

} // namespace infinilm::models::deepseek_v4
