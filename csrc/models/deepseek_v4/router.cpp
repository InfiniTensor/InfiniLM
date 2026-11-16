#include "router.hpp"

#include "execution/submit.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {

using runtime::DataType;
using runtime::Runtime;
using runtime::Shape;
using runtime::Tensor;
using namespace infini::ops;

template <class Op, class... Args>
void run(Runtime &rt, const Args &...args) {
    execution::submit<Op>(rt, Op::DefaultConfig(rt.device().type()), args...);
}

bool integer_dtype(DataType dtype) {
    return dtype == DataType::kInt32 || dtype == DataType::kInt64;
}

} // namespace

RoutedExperts route_experts(
    Runtime &rt, const Tensor &logits, size_t top_k, double route_scale, RouterScore score,
    const Tensor &correction_bias, const Tensor &token_ids, const Tensor &token_to_expert) {
    if (!logits || logits->dtype() != DataType::kFloat32 || logits->ndim() != 2
        || !logits->size(0) || !top_k || top_k > logits->size(1)
        || top_k > static_cast<size_t>(std::numeric_limits<int64_t>::max())
        || !std::isfinite(route_scale) || route_scale <= 0.0) {
        throw std::invalid_argument("Invalid DeepSeek V4 routing geometry or scale");
    }
    const auto tokens = logits->size(0);
    const auto experts = logits->size(1);
    const Shape selected_shape{tokens, top_k};
    const bool hash_routing = static_cast<bool>(token_to_expert);
    if (hash_routing != static_cast<bool>(token_ids) || (hash_routing && correction_bias)) {
        throw std::invalid_argument("DeepSeek V4 hash routing needs both lookup inputs and no bias");
    }
    if (hash_routing && (token_ids->shape() != Shape{tokens} || !integer_dtype(token_ids->dtype())
        || !integer_dtype(token_to_expert->dtype()) || token_to_expert->ndim() != 2
        || token_to_expert->size(1) != top_k || !token_to_expert->size(0))) {
        throw std::invalid_argument("Invalid DeepSeek V4 hash routing table or token IDs");
    }
    if (correction_bias && (correction_bias->shape() != Shape{experts}
        || correction_bias->dtype() != DataType::kFloat32)) {
        throw std::invalid_argument("Invalid DeepSeek V4 expert correction bias");
    }

    const auto float_tensor = [&](const Shape &shape) {
        return Tensor::empty(rt, shape, DataType::kFloat32);
    };
    auto scores = float_tensor(logits->shape());
    switch (score) {
    case RouterScore::SqrtSoftplus: {
        auto positive = float_tensor(logits->shape());
        run<Softplus>(rt, logits, 1.0, 20.0, positive);
        run<Sqrt>(rt, positive, scores);
        break;
    }
    case RouterScore::Sigmoid:
        run<Sigmoid>(rt, logits, scores);
        break;
    case RouterScore::Softmax:
        run<Softmax>(rt, logits, int64_t{1}, std::optional<DataType>{}, scores);
        break;
    default:
        throw std::invalid_argument("Unknown DeepSeek V4 router score function");
    }

    auto indices = Tensor::empty(rt, selected_shape, DataType::kInt64);
    if (hash_routing) {
        auto selected = token_to_expert->dtype() == DataType::kInt64
                            ? indices
                            : Tensor::empty(rt, selected_shape, token_to_expert->dtype());
        run<IndexSelect>(rt, token_to_expert, token_ids, int64_t{0}, selected);
        if (selected->dtype() != DataType::kInt64) {
            run<Cast>(rt, selected, indices);
        }
    } else {
        auto ranking = scores;
        if (correction_bias) {
            ranking = float_tensor(scores->shape());
            run<Add>(rt, scores, correction_bias, 1.0, ranking);
        }
        auto ranked_values = float_tensor(selected_shape);
        run<Topk>(rt, ranking, static_cast<int64_t>(top_k), int64_t{1}, true, true,
                  ranked_values, indices);
    }
    auto weights = float_tensor(selected_shape);
    run<Gather>(rt, scores, indices, int64_t{1}, false, weights);
    if (score != RouterScore::Softmax) {
        auto total = float_tensor({tokens, 1});
        run<Sum>(rt, weights, std::optional{std::vector<int64_t>{1}}, true,
                 std::optional<DataType>{}, total);
        auto normalized = float_tensor(selected_shape);
        run<Div>(rt, weights, total, normalized);
        weights = normalized;
    }
    auto scale = float_tensor({1});
    run<Fill>(rt, scale, route_scale, scale);
    auto result = float_tensor(selected_shape);
    run<Mul>(rt, weights, scale, result);
    return {result, indices};
}

} // namespace infinilm::models::deepseek_v4
