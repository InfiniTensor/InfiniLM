#include "infinicore/ops/random_sample.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<RandomSample::schema> &RandomSample::dispatcher() {
    static common::OpDispatcher<RandomSample::schema> dispatcher_;
    return dispatcher_;
};

void RandomSample::execute(
    Tensor indices, Tensor logits,
    float random_val, float topp, int topk, float temperature) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, logits);
    infinicore::context::setDevice(logits->device());
    dispatcher().lookup(logits->device().getType())(
        indices, logits, random_val, topp, topk, temperature);
}

Tensor random_sample(
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    auto indices = Tensor::empty({}, DataType::I32, logits->device());
    random_sample_(indices, logits, random_val, topp, topk, temperature);
    return indices;
}

void random_sample_(
    Tensor indices,
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    RandomSample::execute(indices, logits, random_val, topp, topk, temperature);
}

} // namespace infinicore::op
