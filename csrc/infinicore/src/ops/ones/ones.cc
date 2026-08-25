#include "infinicore/ops/ones.hpp"

#include "../../../utils/custom_types.h"
#include "../../utils.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infinicore::op {
namespace {

struct CpuPlannedMeta {
    graph::GraphTensor output;
};

template <typename T>
void fill_cpu(Tensor output, T value) {
    std::fill_n(reinterpret_cast<T *>(output->data()), output->numel(), value);
}

void *plan_cpu(Tensor output) {
    INFINICORE_ASSERT(output->device().type() == Device::Type::kCpu);
    return new CpuPlannedMeta{graph::GraphTensor(output)};
}

void run_cpu(void *planned_meta) {
    auto *planned = reinterpret_cast<CpuPlannedMeta *>(planned_meta);
    Tensor output = planned->output;
    context::setDevice(output->device());

    switch (output->dtype()) {
    case DataType::kInt8:
        fill_cpu<int8_t>(output, 1);
        break;
    case DataType::kInt16:
        fill_cpu<int16_t>(output, 1);
        break;
    case DataType::kInt32:
        fill_cpu<int32_t>(output, 1);
        break;
    case DataType::kInt64:
        fill_cpu<int64_t>(output, 1);
        break;
    case DataType::kUInt8:
        fill_cpu<uint8_t>(output, 1);
        break;
    case DataType::kUInt16:
        fill_cpu<uint16_t>(output, 1);
        break;
    case DataType::kUInt32:
        fill_cpu<uint32_t>(output, 1);
        break;
    case DataType::kUInt64:
        fill_cpu<uint64_t>(output, 1);
        break;
    case DataType::kFloat16:
        fill_cpu<fp16_t>(output, utils::cast<fp16_t, float>(1.0f));
        break;
    case DataType::kBFloat16:
        fill_cpu<bf16_t>(output, utils::cast<bf16_t, float>(1.0f));
        break;
    case DataType::kFloat32:
        fill_cpu<float>(output, 1.0f);
        break;
    case DataType::kFloat64:
        fill_cpu<double>(output, 1.0);
        break;
    default:
        throw std::runtime_error("Ones does not support this CPU tensor dtype.");
    }
}

void cleanup_cpu(void **planned_meta_ptr) {
    delete *reinterpret_cast<CpuPlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Ones);

Ones::Ones(Tensor output) {
    INFINICORE_ASSERT(output);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().type(), output);
}

void Ones::execute(Tensor output) {
    context::setDevice(output->device());
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Ones, output);
}

void ones_(Tensor output) {
    Ones::execute(output);
}

static bool cpu_registered = []() {
    Ones::plan_dispatcher().registerDevice(Device::Type::kCpu, &plan_cpu);
    Ones::run_dispatcher().registerDevice(Device::Type::kCpu, &run_cpu);
    Ones::cleanup_dispatcher().registerDevice(Device::Type::kCpu, &cleanup_cpu);
    return true;
}();

} // namespace infinicore::op
