#include "infinicore/ops/causal_softmax.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/add.h"
#include "base/fill.h"
#include "base/softmax.h"
#include "base/tril.h"
#include "base/triu.h"

#include <limits>
#include <optional>

namespace infinicore::op::causal_softmax_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    std::optional<Tensor> mask_owner;
    TensorMeta output, input;
    std::optional<TensorMeta> mask;
    graph::GraphTensor output_tensor, input_tensor;
    std::optional<graph::GraphTensor> mask_tensor;
    int64_t score_diagonal;
    int64_t mask_diagonal;
};

} // namespace

void *plan(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(output->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);

    const auto &shape = output->shape();
    INFINICORE_ASSERT(shape.size() == 2 || shape.size() == 3);
    INFINICORE_ASSERT(input->shape() == shape);

    const auto seq_len = shape[shape.size() - 2];
    const auto total_seq_len = shape[shape.size() - 1];
    INFINICORE_ASSERT(seq_len <= total_seq_len);

    const bool reuses_input = output->data() == input->data() && output->strides() == input->strides();
    const bool needs_mask = seq_len != 1 || !reuses_input;
    std::optional<Tensor> mask;
    if (needs_mask) {
        mask = Tensor::empty({seq_len, total_seq_len}, output->dtype(), output->device());
    }

    return new PlannedMeta{
        mask,
        TensorMeta(output),
        TensorMeta(input),
        mask ? std::optional<TensorMeta>(TensorMeta(*mask)) : std::nullopt,
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        mask ? std::optional<graph::GraphTensor>(graph::GraphTensor(*mask)) : std::nullopt,
        static_cast<int64_t>(total_seq_len - seq_len),
        static_cast<int64_t>(total_seq_len - seq_len + 1)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    const auto device_type = planned->output.device.type();
    auto fill_config = ::infinicore::op::infiniops::defaultConfigForDevice<infini::ops::Fill>(
        device_type);
    auto triu_config = ::infinicore::op::infiniops::defaultConfigForDevice<infini::ops::Triu>(
        device_type);
    auto tril_config = ::infinicore::op::infiniops::defaultConfigForDevice<infini::ops::Tril>(
        device_type);
    auto add_config = ::infinicore::op::infiniops::defaultConfigForDevice<infini::ops::Add>(
        device_type);
    auto softmax_config = ::infinicore::op::infiniops::defaultConfigForDevice<
        infini::ops::Softmax>(device_type);

    auto input = planned->input.tensor(planned->input_tensor);
    auto output = planned->output.tensor(planned->output_tensor);

    if (planned->mask.has_value()) {
        auto mask = planned->mask->tensor(planned->mask_tensor.value());

        infini::ops::Fill::Call(
            handle,
            fill_config,
            mask,
            -std::numeric_limits<double>::infinity(),
            mask);

        infini::ops::Triu::Call(
            handle,
            triu_config,
            mask,
            planned->mask_diagonal,
            mask);

        infini::ops::Tril::Call(
            handle,
            tril_config,
            input,
            planned->score_diagonal,
            output);

        infini::ops::Add::Call(
            handle,
            add_config,
            output,
            mask,
            output);
    }

    infini::ops::Softmax::Call(
        handle,
        softmax_config,
        output,
        static_cast<int64_t>(-1),
        std::optional<infini::ops::DataType>{},
        output);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(CausalSoftmax::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(CausalSoftmax::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(CausalSoftmax::cleanup_dispatcher(), &cleanup);
    return true;
}();

} // namespace infinicore::op::causal_softmax_impl::infiniops
#endif
