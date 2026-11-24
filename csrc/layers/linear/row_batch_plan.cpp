#include "row_batch_plan.hpp"
#include "execution/submit.hpp"
#include "layers/quantization/none_quantization.hpp"
#include "runtime/utils.hpp"

#include <base/gemm.h>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <infini/rt/device_guard.h>
#include <optional>
#include <stdexcept>
#include <utility>

namespace infinilm::nn {
namespace {
using infinilm::runtime::DataType;
using infinilm::runtime::Runtime;
using infinilm::runtime::Tensor;

bool overlaps(const Tensor &a, const Tensor &b) {
    const auto p = reinterpret_cast<uintptr_t>(a->data());
    const auto q = reinterpret_cast<uintptr_t>(b->data());
    return p <= q ? q - p < a->nbytes() : p - q < b->nbytes();
}
} // namespace

struct RowBatchPlan::PreparedGemm {
    Tensor input, weight, packed_weight, output;
    infini::ops::Tensor a, b, y;
    std::shared_ptr<infinilm::runtime::Memory> workspace;
    std::unique_ptr<infini::ops::Operator<infini::ops::Gemm>> operation;
    std::shared_ptr<infinilm::quantization::NoneQuantization> quantization;
    float alpha;
    bool packed;

    PreparedGemm(Runtime &runtime, Tensor staging, Tensor w, Tensor result,
                 std::shared_ptr<infinilm::quantization::NoneQuantization> quant,
                 float scale)
        : input(std::move(staging)), weight(std::move(w)),
          packed_weight(quant->weight_prepacked() ? weight : weight->permute({1, 0})),
          output(std::move(result)), a(infinilm::execution::detail::view(input)),
          b(infinilm::execution::detail::view(packed_weight)),
          y(infinilm::execution::detail::view(output)), quantization(std::move(quant)),
          alpha(scale), packed(quantization->weight_prepacked()) {
        const std::optional<infini::ops::Tensor> c;
        const std::optional<float> alpha_value{alpha}, beta{0.0F};
        const std::optional<int> transpose;
        // Use the generated public const-reference factory ABI, as the existing
        // recorded-call path does; do not instantiate private provider dispatch.
        operation = infini::ops::Gemm::Make(
            infini::ops::Gemm::DefaultConfig(runtime.device().type()), a, std::as_const(b),
            c, alpha_value, beta, transpose, transpose, std::as_const(y));
        const auto bytes = operation->workspace_size_in_bytes();
        if (bytes) {
            workspace = runtime.allocateMemory(bytes);
        }
        infini::ops::Handle handle;
        handle.set_stream(runtime.stream());
        handle.set_workspace(workspace ? workspace->data() : nullptr);
        handle.set_workspace_size_in_bytes(bytes);
        // Bind submission fields once; direct public instance calls keep the
        // same Ops provider/precision while avoiding per-call Handle cloning.
        operation->set_handle(handle);
        operation->set_stream(handle.stream());
        operation->set_workspace(handle.workspace());
        operation->set_workspace_size_in_bytes(bytes);
    }

    void submit() const {
        (*operation)(a, b, std::optional<infini::ops::Tensor>{},
                     std::optional<float>{alpha}, std::optional<float>{0.0F},
                     std::optional<int>{}, std::optional<int>{}, y);
    }
};

RowBatchPlan::RowBatchPlan(std::shared_ptr<Runtime> runtime, Layers layers,
                           const Tensors &input_layouts)
    : runtime_(std::move(runtime)), layers_(std::move(layers)) {
    if (!runtime_) {
        throw std::invalid_argument("RowBatchPlan requires a Runtime");
    }
    check_context();
    const auto k = layers_.size();
    if ((k != 1 && k != 4) || input_layouts.size() != k) {
        throw std::invalid_argument("RowBatchPlan requires K1/K4 input layouts");
    }
    // Validate every task before allocation or submission begins.
    for (size_t i = 0; i < k; ++i) {
        const auto &layer = layers_[i];
        const auto &input = input_layouts[i];
        if (!layer || !input || layer->has_bias() || layer->tp_size() != 2 || !layer->communicator() || (layer->dtype() != DataType::kFloat32 && layer->dtype() != DataType::kFloat16) || layer->output_dtype() != layer->dtype() || !input->is_contiguous() || input->ndim() != 2 || input->size(0) == 0 || input->dtype() != layer->dtype() || input->device() != runtime_->device()) {
            throw std::invalid_argument("unsupported RowBatchPlan task metadata");
        }
        const auto quant = std::dynamic_pointer_cast<infinilm::quantization::NoneQuantization>(layer->get_quantization());
        const auto weight = layer->weight();
        if (!quant || !weight || !weight->is_contiguous() || weight->ndim() != 2 || weight->device() != runtime_->device() || weight->dtype() != layer->dtype() || input->size(1) != weight->size(quant->weight_prepacked() ? 0 : 1) || layer->out_features() != weight->size(quant->weight_prepacked() ? 1 : 0)) {
            throw std::invalid_argument("RowBatchPlan weight/layout mismatch");
        }
        if (i && (layer->communicator() != layers_[0]->communicator() || layer->tp_rank() != layers_[0]->tp_rank() || layer->dtype() != layers_[0]->dtype())) {
            throw std::invalid_argument("RowBatchPlan tasks must share TP rank, communicator and dtype");
        }
    }
    communicator_ = layers_[0]->communicator();
    ccl_dtype_ = layers_[0]->dtype() == DataType::kFloat32 ? infinicclFloat32 : infinicclFloat16;
    gemms_.reserve(k);
    outputs_.reserve(k);
    in_flight_inputs_.reserve(k);
    sends_.reserve(k);
    receives_.reserve(k);
    counts_.reserve(k);
    infini::rt::DeviceGuard guard{runtime_->device().type(), runtime_->device().index(), true};
    try {
        for (size_t i = 0; i < k; ++i) {
            auto staging = Tensor::empty(*runtime_, input_layouts[i]->shape(), layers_[i]->dtype());
            auto output = Tensor::empty(*runtime_, {input_layouts[i]->size(0), layers_[i]->out_features()}, layers_[i]->dtype());
            auto prepared = std::make_unique<PreparedGemm>(
                *runtime_, std::move(staging), layers_[i]->weight(), output,
                std::dynamic_pointer_cast<infinilm::quantization::NoneQuantization>(layers_[i]->get_quantization()),
                layers_[i]->alpha());
            sends_.push_back(output->data());
            receives_.push_back(output->data());
            counts_.push_back(output->numel());
            outputs_.push_back(std::move(output));
            gemms_.push_back(std::move(prepared));
        }
    } catch (...) {
        runtime_->syncStream();
        throw;
    }
}

RowBatchPlan::~RowBatchPlan() noexcept {
    try {
        runtime_->assertOwnerThread();
        infini::rt::DeviceGuard guard{runtime_->device().type(), runtime_->device().index(), true};
        if (state_ != State::Ready) {
            runtime_->syncStream();
        }
        gemms_.clear();
        outputs_.clear();
        in_flight_inputs_.clear();
        layers_.clear();
    } catch (...) { std::terminate(); }
}

void RowBatchPlan::check_context() const {
    runtime_->assertOwnerThread();
    if (runtime_->isGraphRecording() || !runtime_->ownerDrainFailStopEnabled()) {
        throw std::logic_error("RowBatchPlan requires Eager owner drain fail-stop Runtime");
    }
}

void RowBatchPlan::check_inputs(const Tensors &inputs) const {
    if (inputs.size() != gemms_.size()) {
        throw std::invalid_argument("RowBatchPlan input count differs");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &x = inputs[i];
        const auto &task = *gemms_[i];
        const auto weight = layers_[i]->weight();
        if (!x || !x->is_contiguous() || x->shape() != task.input->shape() || x->dtype() != task.input->dtype() || x->device() != runtime_->device() || !weight || weight->data() != task.weight->data() || weight->shape() != task.weight->shape() || !weight->is_contiguous() || weight->dtype() != task.weight->dtype() || weight->device() != runtime_->device() || layers_[i]->alpha() != task.alpha || layers_[i]->get_quantization().get() != task.quantization.get() || task.quantization->weight_prepacked() != task.packed) {
            throw std::invalid_argument("RowBatchPlan input or prepared layer binding changed");
        }
        for (const auto &output : outputs_) {
            if (overlaps(x, output)) {
                throw std::invalid_argument("RowBatchPlan input overlaps a borrowed output");
            }
        }
    }
}

const RowBatchPlan::Tensors &RowBatchPlan::submit(const Tensors &inputs) {
    check_context();
    if (state_ != State::Ready) {
        throw std::logic_error("RowBatchPlan must complete before reuse; failed plans cannot retry");
    }
    check_inputs(inputs);
    in_flight_inputs_.assign(inputs.begin(), inputs.end());
    infini::rt::DeviceGuard guard{runtime_->device().type(), runtime_->device().index(), true};
    state_ = State::InFlight;
    try {
        for (size_t i = 0; i < gemms_.size(); ++i) {
            // The outer guard already selected this HYGON device and restores it.
            if (runtime_->device().type() == infini::rt::Device::Type::kHygon) {
                INFINILM_CHECK_ERROR(infini::rt::runtime::MemcpyAsync(
                    gemms_[i]->input->data(), inputs[i]->data(), inputs[i]->nbytes(),
                    infini::rt::runtime::kMemcpyDeviceToDevice, runtime_->stream()));
            } else {
                runtime_->memcpyD2D(gemms_[i]->input->data(), inputs[i]->data(), inputs[i]->nbytes());
            }
        }
        for (const auto &gemm : gemms_) {
            gemm->submit();
        }
        // Match existing LM collective failure semantics: a partially submitted
        // communication group cannot be retried or unwound rank-locally.
        infinicclResult_t result;
        try {
            result = infinicclAllReduceBatch(sends_.data(), receives_.data(), counts_.data(),
                                             counts_.size(), ccl_dtype_, infinicclSum,
                                             communicator_, reinterpret_cast<void *>(runtime_->stream()));
        } catch (...) { std::_Exit(86); }
        if (result != infinicclSuccess) {
            std::_Exit(86);
        }
    } catch (...) {
        state_ = State::Failed;
        throw;
    }
    return outputs_;
}

void RowBatchPlan::complete() {
    check_context();
    if (state_ == State::Failed) {
        throw std::logic_error("failed RowBatchPlan cannot be reused");
    }
    if (state_ == State::Ready) {
        return;
    }
    runtime_->syncStream();
    in_flight_inputs_.clear();
    state_ = State::Ready;
}

} // namespace infinilm::nn
