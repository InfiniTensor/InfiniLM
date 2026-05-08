#include "operators.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef INFINILM_ENABLE_INFINIOPS
#include "acl/acl_rt.h"
#include "ascend/add/kernel.h"
#include "ascend/add_rms_norm/kernel.h"
#include "ascend/causal_softmax/kernel.h"
#include "ascend/device_.h"
#include "ascend/embedding/kernel.h"
#include "ascend/graph_cleanup_.h"
#include "ascend/linear/kernel.h"
#include "ascend/matmul/kernel.h"
#include "ascend/mha_fwd_kvcache/kernel.h"
#include "ascend/mha_varlen_fwd/kernel.h"
#include "ascend/mul/kernel.h"
#include "ascend/paged_attention/kernel_atb.h"
#include "ascend/reshape_and_cache/kernel.h"
#include "ascend/reshape_and_cache/kernel_atb.h"
#include "ascend/rms_norm/kernel.h"
#include "ascend/rotary_embedding/kernel.h"
#include "ascend/scaled_softmax/kernel.h"
#include "ascend/swiglu/kernel.h"
#include "ascend/top_k_top_p_sampler/kernel.h"
#include "ascend/top_k_top_p_sampler/kernel_atb.h"
#endif

namespace infinilm::backends::ops {

struct GraphTaskUpdate {
    std::function<void()> runner;
    bool defer_acl_cleanup = false;
    std::vector<std::function<void()>> pending_acl_cleanups;
#ifdef INFINILM_ENABLE_INFINIOPS
    aclrtTaskGrp handle = nullptr;
#endif

    void run_pending_acl_cleanups() {
        for (auto &cleanup : pending_acl_cleanups) {
            cleanup();
        }
        pending_acl_cleanups.clear();
    }
};

namespace {

#ifdef INFINILM_ENABLE_INFINIOPS
infini::ops::DataType to_dtype(infinicore::DataType dtype) {
    switch (dtype) {
    case infinicore::DataType::I8:
        return infini::ops::DataType::kInt8;
    case infinicore::DataType::I16:
        return infini::ops::DataType::kInt16;
    case infinicore::DataType::I32:
        return infini::ops::DataType::kInt32;
    case infinicore::DataType::I64:
        return infini::ops::DataType::kInt64;
    case infinicore::DataType::U8:
        return infini::ops::DataType::kUInt8;
    case infinicore::DataType::U16:
        return infini::ops::DataType::kUInt16;
    case infinicore::DataType::U32:
        return infini::ops::DataType::kUInt32;
    case infinicore::DataType::U64:
        return infini::ops::DataType::kUInt64;
    case infinicore::DataType::F16:
        return infini::ops::DataType::kFloat16;
    case infinicore::DataType::BF16:
        return infini::ops::DataType::kBFloat16;
    case infinicore::DataType::F32:
        return infini::ops::DataType::kFloat32;
    case infinicore::DataType::F64:
        return infini::ops::DataType::kFloat64;
    default:
        throw std::runtime_error("InfiniOps adapter: unsupported dtype");
    }
}

infini::ops::Device to_device(const infinicore::Device &device) {
    switch (device.getType()) {
    case infinicore::Device::Type::CPU:
        return {infini::ops::Device::Type::kCpu, static_cast<int>(device.getIndex())};
    case infinicore::Device::Type::ASCEND:
        return {infini::ops::Device::Type::kAscend, static_cast<int>(device.getIndex())};
    default:
        throw std::runtime_error("InfiniOps adapter: unsupported device");
    }
}

infini::ops::Tensor as_tensor(const infinicore::Tensor &tensor) {
    std::vector<std::size_t> shape(tensor->shape().begin(), tensor->shape().end());
    std::vector<std::ptrdiff_t> strides(tensor->strides().begin(), tensor->strides().end());
    return {
        const_cast<std::byte *>(tensor->data()),
        shape,
        to_dtype(tensor->dtype()),
        to_device(tensor->device()),
        strides,
    };
}

std::optional<infini::ops::Tensor> as_optional_tensor(const std::optional<infinicore::Tensor> &tensor) {
    if (!tensor.has_value() || !tensor.value()) {
        return std::nullopt;
    }
    return as_tensor(tensor.value());
}

infini::ops::Handle handle() {
    infini::ops::Handle h;
    h.set_stream(infinicore::context::getStream());
    return h;
}

infini::ops::Config config(std::size_t implementation_index = 0) {
    infini::ops::Config cfg;
    cfg.set_implementation_index(implementation_index);
    return cfg;
}

thread_local GraphTaskUpdates *active_graph_task_capture = nullptr;

bool is_ri_capturing() {
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model_ri = nullptr;
    auto ret = aclmdlRICaptureGetInfo(
        static_cast<aclrtStream>(infinicore::context::getStream()),
        &status,
        &model_ri);
    return ret == ACL_SUCCESS && status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
}

class InfiniOpsGraphOperator : public infinicore::graph::GraphOperator {
public:
    explicit InfiniOpsGraphOperator(std::function<void()> runner,
                                    std::shared_ptr<GraphTaskUpdate> graph_task_update = nullptr,
                                    bool defer_acl_cleanup = false)
        : runner_(std::move(runner)),
          graph_task_update_(std::move(graph_task_update)),
          defer_acl_cleanup_(defer_acl_cleanup) {}

    void run() const override {
#ifdef INFINILM_ENABLE_INFINIOPS
        if (graph_task_update_ && is_ri_capturing()) {
            auto stream = static_cast<aclrtStream>(infinicore::context::getStream());
            auto ret = aclmdlRICaptureTaskGrpBegin(stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("InfiniOps adapter: aclmdlRICaptureTaskGrpBegin failed");
            }
            aclrtTaskGrp handle = nullptr;
            if (defer_acl_cleanup_) {
                infini::ops::ascend::DeferredAclCleanupScope cleanup_scope;
                runner_();
                ret = aclmdlRICaptureTaskGrpEnd(stream, &handle);
                if (ret == ACL_SUCCESS && graph_task_update_) {
                    graph_task_update_->pending_acl_cleanups = cleanup_scope.Release();
                }
            } else {
                runner_();
                ret = aclmdlRICaptureTaskGrpEnd(stream, &handle);
            }
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("InfiniOps adapter: aclmdlRICaptureTaskGrpEnd failed");
            }
            graph_task_update_->handle = handle;
            return;
        }
#endif
        runner_();
    }

private:
    std::function<void()> runner_;
    std::shared_ptr<GraphTaskUpdate> graph_task_update_;
    bool defer_acl_cleanup_;
};

void record_or_run(std::function<void()> runner,
                   std::shared_ptr<GraphTaskUpdate> graph_task_update = nullptr,
                   bool defer_acl_cleanup = false) {
    if (infinicore::context::isGraphRecording()) {
        infinicore::context::addGraphOperator(
            std::make_shared<InfiniOpsGraphOperator>(
                std::move(runner), graph_task_update, defer_acl_cleanup));
        return;
    }

    runner();
}

void record_or_run_updatable(std::function<void()> runner,
                             bool defer_acl_cleanup = false) {
    std::shared_ptr<GraphTaskUpdate> update;
    if (infinicore::context::isGraphRecording() && active_graph_task_capture != nullptr) {
        update = std::make_shared<GraphTaskUpdate>();
        update->runner = runner;
        update->defer_acl_cleanup = defer_acl_cleanup;
        active_graph_task_capture->push_back(update);
    }
    record_or_run(std::move(runner), std::move(update), defer_acl_cleanup);
}

bool unquantized(const infinicore::nn::BaseLinear &linear) {
    return linear.get_quantization()->get_quant_scheme()
        == infinicore::quantization::QuantScheme::NONE;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t f32;
    std::memcpy(&f32, &value, sizeof(f32));
    const std::uint16_t sign = (f32 >> 16) & 0x8000;
    const std::int32_t exponent = static_cast<std::int32_t>((f32 >> 23) & 0xFF) - 127;
    std::uint32_t mantissa = f32 & 0x7FFFFF;

    if (exponent >= 16) {
        if (exponent == 128 && mantissa != 0) {
            return static_cast<std::uint16_t>(sign | 0x7E00);
        }
        return static_cast<std::uint16_t>(sign | 0x7C00);
    }
    if (exponent >= -14) {
        return static_cast<std::uint16_t>(
            sign | ((exponent + 15) << 10) | (mantissa >> 13));
    }
    if (exponent >= -24) {
        mantissa |= 0x800000;
        mantissa >>= (-14 - exponent);
        return static_cast<std::uint16_t>(sign | (mantissa >> 13));
    }
    return sign;
}

std::uint16_t f32_to_bf16_bits(float value) {
    std::uint32_t bits32;
    std::memcpy(&bits32, &value, sizeof(bits32));
    const std::uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    return static_cast<std::uint16_t>((bits32 + rounding_bias) >> 16);
}

infinicore::Tensor require_contiguous(const infinicore::Tensor &tensor,
                                      const char *name) {
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(std::string("InfiniOps adapter: ") + name + " must be contiguous");
    }
    return tensor;
}

void require_same_dtype_device(const infinicore::Tensor &a,
                               const infinicore::Tensor &b,
                               const char *name) {
    if (a->dtype() != b->dtype()) {
        throw std::runtime_error(std::string("InfiniOps adapter: ") + name + " dtype mismatch");
    }
    if (a->device().getType() != b->device().getType()
        || a->device().getIndex() != b->device().getIndex()) {
        throw std::runtime_error(std::string("InfiniOps adapter: ") + name + " device mismatch");
    }
}

infinicore::Shape broadcast_shape(const infinicore::Shape &a,
                                  const infinicore::Shape &b,
                                  const char *name) {
    const auto ndim = std::max(a.size(), b.size());
    infinicore::Shape out(ndim, 1);
    for (std::size_t i = 0; i < ndim; ++i) {
        const auto ai = a.size() > i ? a[a.size() - 1 - i] : 1;
        const auto bi = b.size() > i ? b[b.size() - 1 - i] : 1;
        if (ai != bi && ai != 1 && bi != 1) {
            throw std::runtime_error(std::string("InfiniOps adapter: ") + name + " shapes are not broadcastable");
        }
        out[ndim - 1 - i] = std::max(ai, bi);
    }
    return out;
}

void require_shape(const infinicore::Shape &actual,
                   const infinicore::Shape &expected,
                   const char *name) {
    if (actual != expected) {
        throw std::runtime_error(std::string("InfiniOps adapter: ") + name + " output shape mismatch");
    }
}

infinicore::Tensor broadcast_view(const infinicore::Tensor &tensor,
                                  const infinicore::Shape &out_shape,
                                  const char *name) {
    const auto &shape = tensor->shape();
    const auto &strides = tensor->strides();
    if (shape == out_shape) {
        return tensor;
    }

    infinicore::Strides out_strides(out_shape.size(), 0);
    for (std::size_t i = 0; i < out_shape.size(); ++i) {
        const auto out_dim = out_shape[out_shape.size() - 1 - i];
        if (shape.size() <= i) {
            out_strides[out_shape.size() - 1 - i] = 0;
            continue;
        }

        const auto in_index = shape.size() - 1 - i;
        const auto in_dim = shape[in_index];
        if (in_dim == out_dim) {
            out_strides[out_shape.size() - 1 - i] = strides[in_index];
        } else if (in_dim == 1) {
            out_strides[out_shape.size() - 1 - i] = 0;
        } else {
            throw std::runtime_error(std::string("InfiniOps adapter: ") + name + " cannot be broadcast to output shape");
        }
    }

    return tensor->as_strided(out_shape, out_strides);
}

infinicore::Shape matmul_output_shape(const infinicore::Tensor &a,
                                      const infinicore::Tensor &b) {
    if (a->ndim() < 2 || b->ndim() < 2) {
        throw std::runtime_error("InfiniOps adapter: matmul requires tensors with at least 2 dimensions");
    }
    const auto a_k = a->size(a->ndim() - 1);
    const auto b_k = b->size(b->ndim() - 2);
    if (a_k != b_k) {
        throw std::runtime_error("InfiniOps adapter: matmul reduction dimensions mismatch");
    }

    infinicore::Shape a_batch(a->shape().begin(), a->shape().end() - 2);
    infinicore::Shape b_batch(b->shape().begin(), b->shape().end() - 2);
    auto out = broadcast_shape(a_batch, b_batch, "matmul batch");
    out.push_back(a->size(a->ndim() - 2));
    out.push_back(b->size(b->ndim() - 1));
    return out;
}

int64_t max_sequence_length(const infinicore::Tensor &cu_seqlens) {
    if (cu_seqlens->dtype() != infinicore::DataType::I32) {
        throw std::runtime_error("InfiniOps adapter: cu_seqlens must be int32");
    }
    require_contiguous(cu_seqlens, "cu_seqlens");

    const auto source_device = cu_seqlens->device();
    auto host = source_device.getType() == infinicore::Device::Type::CPU
                  ? cu_seqlens
                  : cu_seqlens->to(infinicore::Device::cpu());
    const auto *data = reinterpret_cast<const int32_t *>(host->data());
    int64_t max_len = 0;
    for (std::size_t i = 0; i + 1 < host->size(0); ++i) {
        max_len = std::max<int64_t>(max_len, data[i + 1] - data[i]);
    }
    if (source_device.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::setDevice(source_device);
    }
    return max_len;
}

infinicore::Tensor scalar_tensor(infinicore::DataType dtype,
                                 const infinicore::Device &device,
                                 float value) {
    auto out = infinicore::Tensor::empty({1}, dtype, device);
    auto cpu = infinicore::Device::cpu();
    if (dtype == infinicore::DataType::F16) {
        auto host_value = f32_to_f16_bits(value);
        auto host = infinicore::Tensor::from_blob(&host_value, {1}, dtype, cpu);
        out->copy_from(host);
    } else if (dtype == infinicore::DataType::BF16) {
        auto host_value = f32_to_bf16_bits(value);
        auto host = infinicore::Tensor::from_blob(&host_value, {1}, dtype, cpu);
        out->copy_from(host);
    } else if (dtype == infinicore::DataType::F32) {
        auto host = infinicore::Tensor::from_blob(&value, {1}, dtype, cpu);
        out->copy_from(host);
    } else {
        throw std::runtime_error("InfiniOps adapter: unsupported scalar dtype");
    }
    return out;
}
#endif

void require_enabled() {
#ifndef INFINILM_ENABLE_INFINIOPS
    throw std::runtime_error("InfiniOps adapter was not enabled at build time");
#endif
}

template <typename... Args>
void ignore_args(const Args &...) {}

[[noreturn]] void missing_operator(const char *name) {
    throw std::runtime_error(std::string("InfiniLM ops wrapper: external InfiniOps operator is not wired: ") + name);
}

} // namespace

bool should_use(const infinicore::Device &device) {
#ifdef INFINILM_ENABLE_INFINIOPS
    return device.getType() == infinicore::Device::Type::ASCEND;
#else
    (void)device;
    return false;
#endif
}

void begin_graph_task_capture() {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (active_graph_task_capture != nullptr) {
        throw std::runtime_error("InfiniOps adapter: graph task capture is already active");
    }
    active_graph_task_capture = new GraphTaskUpdates();
#endif
}

GraphTaskUpdates end_graph_task_capture() {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (active_graph_task_capture == nullptr) {
        return {};
    }
    GraphTaskUpdates updates = std::move(*active_graph_task_capture);
    delete active_graph_task_capture;
    active_graph_task_capture = nullptr;
    return updates;
#else
    return {};
#endif
}

void update_graph_tasks(const GraphTaskUpdates &updates) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto stream = static_cast<aclrtStream>(infinicore::context::getStream());
    for (const auto &update : updates) {
        if (!update || update->handle == nullptr || !update->runner) {
            continue;
        }
        auto ret = aclmdlRICaptureTaskUpdateBegin(stream, update->handle);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("InfiniOps adapter: aclmdlRICaptureTaskUpdateBegin failed");
        }
        if (update->defer_acl_cleanup) {
            infini::ops::ascend::DeferredAclCleanupScope cleanup_scope;
            update->runner();
            ret = aclmdlRICaptureTaskUpdateEnd(stream);
            if (ret == ACL_SUCCESS) {
                update->run_pending_acl_cleanups();
                update->pending_acl_cleanups = cleanup_scope.Release();
            }
        } else {
            update->runner();
            ret = aclmdlRICaptureTaskUpdateEnd(stream);
        }
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("InfiniOps adapter: aclmdlRICaptureTaskUpdateEnd failed");
        }
    }
#else
    (void)updates;
#endif
}

infinicore::Tensor embedding(const infinicore::Tensor &input_ids,
                             const infinicore::Tensor &weight) {
#ifdef INFINILM_ENABLE_INFINIOPS
    require_contiguous(input_ids, "embedding input_ids");
    require_contiguous(weight, "embedding weight");
    std::vector<std::size_t> out_shape(input_ids->shape().begin(), input_ids->shape().end());
    out_shape.push_back(weight->size(1));
    auto out = infinicore::Tensor::empty(out_shape, weight->dtype(), weight->device());
    auto input_ids_g = infinicore::graph::GraphTensor(input_ids);
    auto weight_g = infinicore::graph::GraphTensor(weight);
    auto out_g = infinicore::graph::GraphTensor(out);
    record_or_run([input_ids_g, weight_g, out_g]() {
        infini::ops::Embedding::Call(handle(), config(), as_tensor(input_ids_g), as_tensor(weight_g), as_tensor(out_g));
    });
    return out;
#else
    require_enabled();
    return {};
#endif
}

infinicore::Tensor linear(const infinicore::Tensor &input,
                          const infinicore::Tensor &weight,
                          std::optional<infinicore::Tensor> bias) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto input_contiguous = require_contiguous(input, "linear input");
    auto weight_contiguous = require_contiguous(weight, "linear weight");
    if (bias.has_value()) {
        require_contiguous(bias.value(), "linear bias");
    }
    std::vector<std::size_t> out_shape(input_contiguous->shape().begin(), input_contiguous->shape().end());
    out_shape.back() = weight_contiguous->size(weight_contiguous->ndim() - 2);
    auto out = infinicore::Tensor::empty(out_shape, input_contiguous->dtype(), input_contiguous->device());
    auto input_g = infinicore::graph::GraphTensor(input_contiguous);
    auto weight_g = infinicore::graph::GraphTensor(weight_contiguous);
    auto out_g = infinicore::graph::GraphTensor(out);
    std::optional<infinicore::graph::GraphTensor> bias_g;
    if (bias.has_value()) {
        bias_g.emplace(bias.value());
    }
    record_or_run([input_g, weight_g, bias_g, out_g]() {
        std::optional<infini::ops::Tensor> bias_opt;
        if (bias_g.has_value()) {
            bias_opt.emplace(as_tensor(bias_g.value()));
        }
        infini::ops::Linear::Call(
            handle(), config(), as_tensor(input_g), as_tensor(weight_g),
            bias_opt, as_tensor(out_g));
    });
    return out;
#else
    require_enabled();
    return {};
#endif
}

infinicore::Tensor linear_w8a8i8(const infinicore::Tensor &input,
                                 const infinicore::Tensor &weight,
                                 const infinicore::Tensor &weight_scale,
                                 std::optional<infinicore::Tensor> bias) {
    ignore_args(input, weight, weight_scale, bias);
    missing_operator("linear_w8a8i8");
}

infinicore::Tensor linear_w4a16_awq(const infinicore::Tensor &input,
                                    const infinicore::Tensor &qweight,
                                    const infinicore::Tensor &scales,
                                    const infinicore::Tensor &qzeros,
                                    std::optional<infinicore::Tensor> bias) {
    ignore_args(input, qweight, scales, qzeros, bias);
    missing_operator("linear_w4a16_awq");
}

infinicore::Tensor linear_w4a16_gptq_qy(const infinicore::Tensor &input,
                                        const infinicore::Tensor &qweight,
                                        const infinicore::Tensor &qzeros,
                                        const infinicore::Tensor &scales,
                                        int64_t group_size,
                                        int64_t bits) {
    ignore_args(input, qweight, qzeros, scales, group_size, bits);
    missing_operator("linear_w4a16_gptq_qy");
}

infinicore::Tensor add(const infinicore::Tensor &a,
                       const infinicore::Tensor &b) {
#ifdef INFINILM_ENABLE_INFINIOPS
    require_same_dtype_device(a, b, "add");
    auto out_shape = broadcast_shape(a->shape(), b->shape(), "add");
    auto out = infinicore::Tensor::empty(out_shape, a->dtype(), a->device());
    add_(out, a, b);
    return out;
#else
    (void)a;
    (void)b;
    require_enabled();
    return {};
#endif
}

void add_(const infinicore::Tensor &out,
          const infinicore::Tensor &a,
          const infinicore::Tensor &b) {
#ifdef INFINILM_ENABLE_INFINIOPS
    require_same_dtype_device(a, b, "add");
    require_same_dtype_device(a, out, "add");
    require_shape(out->shape(), broadcast_shape(a->shape(), b->shape(), "add"), "add");
    auto a_view = broadcast_view(a, out->shape(), "add lhs");
    auto b_view = broadcast_view(b, out->shape(), "add rhs");
    auto a_g = infinicore::graph::GraphTensor(a_view);
    auto b_g = infinicore::graph::GraphTensor(b_view);
    auto out_g = infinicore::graph::GraphTensor(out);
    record_or_run([a_g, b_g, out_g]() {
        infini::ops::Add::Call(handle(), config(), as_tensor(a_g), as_tensor(b_g), as_tensor(out_g));
    });
#else
    (void)out;
    (void)a;
    (void)b;
    require_enabled();
#endif
}

void rms_norm_(const infinicore::Tensor &out,
               const infinicore::Tensor &input,
               const infinicore::Tensor &weight,
               float eps) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto input_g = infinicore::graph::GraphTensor(input);
    auto weight_g = infinicore::graph::GraphTensor(weight);
    auto out_g = infinicore::graph::GraphTensor(out);
    record_or_run([input_g, weight_g, out_g, eps]() {
        infini::ops::RmsNorm::Call(handle(), config(), as_tensor(input_g), as_tensor(weight_g), eps, as_tensor(out_g));
    });
#else
    (void)out;
    (void)input;
    (void)weight;
    (void)eps;
    require_enabled();
#endif
}

infinicore::Tensor rms_norm(const infinicore::Tensor &input,
                            const infinicore::Tensor &weight,
                            float eps) {
    auto out = infinicore::Tensor::empty(input->shape(), input->dtype(), input->device());
    rms_norm_(out, input, weight, eps);
    return out;
}

infinicore::Tensor layer_norm(const infinicore::Tensor &input,
                              const infinicore::Tensor &weight,
                              const infinicore::Tensor &bias,
                              float eps) {
    ignore_args(input, weight, bias, eps);
    missing_operator("layer_norm");
}

void add_rms_norm_(infinicore::Tensor &input,
                   infinicore::Tensor &residual,
                   const infinicore::Tensor &weight,
                   float eps) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto residual_out = infinicore::Tensor::empty(residual->shape(), residual->dtype(), residual->device());
    auto input_g = infinicore::graph::GraphTensor(input);
    auto residual_g = infinicore::graph::GraphTensor(residual);
    auto weight_g = infinicore::graph::GraphTensor(weight);
    auto residual_out_g = infinicore::graph::GraphTensor(residual_out);
    record_or_run([input_g, residual_g, weight_g, residual_out_g, eps]() {
        infini::ops::AddRmsNorm::Call(
            handle(), config(), as_tensor(input_g), as_tensor(residual_g), as_tensor(weight_g), eps,
            as_tensor(input_g), as_tensor(residual_out_g));
    });
    residual = residual_out;
#else
    (void)input;
    (void)residual;
    (void)weight;
    (void)eps;
    require_enabled();
#endif
}

infinicore::Tensor swiglu(const infinicore::Tensor &up,
                          const infinicore::Tensor &gate) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto out = infinicore::Tensor::empty(up->shape(), up->dtype(), up->device());
    auto up_g = infinicore::graph::GraphTensor(up);
    auto gate_g = infinicore::graph::GraphTensor(gate);
    auto out_g = infinicore::graph::GraphTensor(out);
    record_or_run([up_g, gate_g, out_g]() {
        infini::ops::Swiglu::Call(handle(), config(), as_tensor(up_g), as_tensor(gate_g), as_tensor(out_g));
    });
    return out;
#else
    require_enabled();
    return {};
#endif
}

void reshape_and_cache(const infinicore::Tensor &key,
                       const infinicore::Tensor &value,
                       const infinicore::Tensor &kv_cache,
                       const infinicore::Tensor &slot_mapping) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto key_g = infinicore::graph::GraphTensor(key);
    auto value_g = infinicore::graph::GraphTensor(value);
    auto kv_cache_g = infinicore::graph::GraphTensor(kv_cache);
    auto slot_mapping_g = infinicore::graph::GraphTensor(slot_mapping);
    record_or_run([key_g, value_g, kv_cache_g, slot_mapping_g]() {
        infini::ops::ReshapeAndCache::Call(
            handle(), config(2), as_tensor(key_g), as_tensor(value_g), as_tensor(kv_cache_g),
            as_tensor(slot_mapping_g), as_tensor(kv_cache_g));
    });
#else
    (void)key;
    (void)value;
    (void)kv_cache;
    (void)slot_mapping;
    require_enabled();
#endif
}

void kv_caching_(const infinicore::Tensor &k_cache,
                 const infinicore::Tensor &v_cache,
                 const infinicore::Tensor &k,
                 const infinicore::Tensor &v,
                 const infinicore::Tensor &past_sequence_lengths) {
    ignore_args(k_cache, v_cache, k, v, past_sequence_lengths);
    missing_operator("kv_caching_");
}

void paged_caching_(const infinicore::Tensor &k_cache,
                    const infinicore::Tensor &v_cache,
                    const infinicore::Tensor &k,
                    const infinicore::Tensor &v,
                    const infinicore::Tensor &slot_mapping) {
    ignore_args(k_cache, v_cache, k, v, slot_mapping);
    missing_operator("paged_caching_");
}

void rotary_embedding(const infinicore::Tensor &positions,
                      const infinicore::Tensor &query,
                      const infinicore::Tensor &key,
                      const infinicore::Tensor &cos_sin_cache,
                      int64_t head_size,
                      std::optional<infinicore::Tensor> query_out,
                      std::optional<infinicore::Tensor> key_out) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto positions_g = infinicore::graph::GraphTensor(positions);
    auto query_g = infinicore::graph::GraphTensor(query);
    auto key_g = infinicore::graph::GraphTensor(key);
    auto cos_sin_cache_g = infinicore::graph::GraphTensor(cos_sin_cache);
    std::optional<infinicore::graph::GraphTensor> query_out_g;
    std::optional<infinicore::graph::GraphTensor> key_out_g;
    if (query_out.has_value()) {
        query_out_g.emplace(query_out.value());
    }
    if (key_out.has_value()) {
        key_out_g.emplace(key_out.value());
    }
    record_or_run([positions_g, query_g, key_g, cos_sin_cache_g, head_size, query_out_g, key_out_g]() {
        std::optional<infini::ops::Tensor> key_opt = as_tensor(key_g);
        std::optional<infini::ops::Tensor> query_out_opt;
        std::optional<infini::ops::Tensor> key_out_opt;
        if (query_out_g.has_value()) {
            query_out_opt.emplace(as_tensor(query_out_g.value()));
        }
        if (key_out_g.has_value()) {
            key_out_opt.emplace(as_tensor(key_out_g.value()));
        }
        infini::ops::RotaryEmbedding::Call(
            handle(), config(), as_tensor(positions_g), as_tensor(query_g), key_opt,
            head_size, as_tensor(cos_sin_cache_g), true, head_size,
            query_out_opt, key_out_opt, false);
    });
#else
    (void)positions;
    (void)query;
    (void)key;
    (void)cos_sin_cache;
    (void)head_size;
    (void)query_out;
    (void)key_out;
    require_enabled();
#endif
}

void mha_varlen_fwd(const infinicore::Tensor &out,
                    const infinicore::Tensor &q,
                    const infinicore::Tensor &k,
                    const infinicore::Tensor &v,
                    const infinicore::Tensor &cu_seqlens_q,
                    const infinicore::Tensor &cu_seqlens_k,
                    const infinicore::Tensor &block_table,
                    int64_t max_seqlen_q,
                    int64_t max_seqlen_k,
                    float softmax_scale) {
#ifdef INFINILM_ENABLE_INFINIOPS
    (void)max_seqlen_q;
    (void)max_seqlen_k;
    const auto actual_max_seqlen_q = max_sequence_length(cu_seqlens_q);
    const auto actual_max_seqlen_k = max_sequence_length(cu_seqlens_k);
    auto q_g = infinicore::graph::GraphTensor(q);
    auto k_g = infinicore::graph::GraphTensor(k);
    auto v_g = infinicore::graph::GraphTensor(v);
    auto out_g = infinicore::graph::GraphTensor(out);
    auto cu_seqlens_q_g = infinicore::graph::GraphTensor(cu_seqlens_q);
    auto cu_seqlens_k_g = infinicore::graph::GraphTensor(cu_seqlens_k);
    auto block_table_g = infinicore::graph::GraphTensor(block_table);
    record_or_run([q_g, k_g, v_g, out_g, cu_seqlens_q_g, cu_seqlens_k_g, block_table_g,
                   actual_max_seqlen_q, actual_max_seqlen_k, softmax_scale]() {
        std::optional<infini::ops::Tensor> block_table_opt = as_tensor(block_table_g);
        std::optional<infini::ops::Tensor> no_tensor;
        std::optional<int64_t> no_generator;
        infini::ops::MhaVarlenFwd::Call(
            handle(), config(), as_tensor(q_g), as_tensor(k_g), as_tensor(v_g), as_tensor(out_g),
            as_tensor(cu_seqlens_q_g), as_tensor(cu_seqlens_k_g), no_tensor,
            no_tensor, block_table_opt, no_tensor, actual_max_seqlen_q, actual_max_seqlen_k,
            0.0f, softmax_scale, false, true, -1, 0, 0.0f, false, no_generator, 0);
    });
#else
    (void)out;
    (void)q;
    (void)k;
    (void)v;
    (void)cu_seqlens_q;
    (void)cu_seqlens_k;
    (void)block_table;
    (void)max_seqlen_q;
    (void)max_seqlen_k;
    (void)softmax_scale;
    require_enabled();
#endif
}

void mha_fwd_kvcache(const infinicore::Tensor &out,
                     const infinicore::Tensor &q,
                     const infinicore::Tensor &kcache,
                     const infinicore::Tensor &vcache,
                     const infinicore::Tensor &seqlens_k,
                     const infinicore::Tensor &block_table,
                     float softmax_scale,
                     std::optional<infinicore::Tensor> seqlens_k_host) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto out_g = infinicore::graph::GraphTensor(out);
    auto q_g = infinicore::graph::GraphTensor(q);
    auto kcache_g = infinicore::graph::GraphTensor(kcache);
    auto vcache_g = infinicore::graph::GraphTensor(vcache);
    auto seqlens_k_g = infinicore::graph::GraphTensor(seqlens_k);
    auto block_table_g = infinicore::graph::GraphTensor(block_table);
    record_or_run_updatable([out_g, q_g, kcache_g, vcache_g, seqlens_k_g, block_table_g, softmax_scale, seqlens_k_host]() {
        std::optional<infini::ops::Tensor> seqlens_k_opt = seqlens_k_host.has_value() ? as_tensor(seqlens_k_host.value()) : as_tensor(seqlens_k_g);
        std::optional<infini::ops::Tensor> block_table_opt = as_tensor(block_table_g);
        std::optional<infini::ops::Tensor> no_tensor;
        infini::ops::MhaFwdKvcache::Call(
            handle(), config(), as_tensor(q_g), as_tensor(kcache_g), as_tensor(vcache_g),
            no_tensor, no_tensor, seqlens_k_opt, no_tensor, no_tensor,
            no_tensor, no_tensor, block_table_opt, no_tensor, as_tensor(out_g),
            softmax_scale, true, -1, 0, 0.0f, false, 0);
    }, true);
#else
    (void)out;
    (void)q;
    (void)kcache;
    (void)vcache;
    (void)seqlens_k;
    (void)block_table;
    (void)softmax_scale;
    (void)seqlens_k_host;
    require_enabled();
#endif
}

void paged_attention(const infinicore::Tensor &out,
                     const infinicore::Tensor &q,
                     const infinicore::Tensor &kcache,
                     const infinicore::Tensor &vcache,
                     const infinicore::Tensor &seqlens_k,
                     const infinicore::Tensor &block_table,
                     int64_t num_heads,
                     int64_t num_kv_heads,
                     int64_t head_size,
                     float softmax_scale,
                     std::optional<infinicore::Tensor> seqlens_k_host,
                     std::optional<infinicore::Tensor> block_table_host) {
#ifdef INFINILM_ENABLE_INFINIOPS
    const auto block_size = static_cast<int64_t>(kcache->size(1));
    auto out_g = infinicore::graph::GraphTensor(out);
    auto q_g = infinicore::graph::GraphTensor(q);
    auto kcache_g = infinicore::graph::GraphTensor(kcache);
    auto vcache_g = infinicore::graph::GraphTensor(vcache);
    auto seqlens_k_g = infinicore::graph::GraphTensor(seqlens_k);
    auto block_table_g = infinicore::graph::GraphTensor(block_table);
    record_or_run_updatable([out_g, q_g, kcache_g, vcache_g, seqlens_k_g, block_table_g,
                             num_heads, num_kv_heads, head_size, softmax_scale, block_size,
                             seqlens_k_host, block_table_host]() {
        std::optional<infini::ops::Tensor> seq_lens_host_opt;
        std::optional<infini::ops::Tensor> block_table_host_opt;
        if (seqlens_k_host.has_value()) {
            seq_lens_host_opt.emplace(as_tensor(seqlens_k_host.value()));
        }
        if (block_table_host.has_value()) {
            block_table_host_opt.emplace(as_tensor(block_table_host.value()));
        }
        infini::ops::PagedAttention::Call(
            handle(), config(), as_tensor(q_g), as_tensor(kcache_g), as_tensor(vcache_g),
            as_tensor(seqlens_k_g), as_tensor(block_table_g), num_heads, num_kv_heads,
            head_size, static_cast<double>(softmax_scale), block_size, as_tensor(out_g),
            seq_lens_host_opt, block_table_host_opt);
    });
#else
    (void)out;
    (void)q;
    (void)kcache;
    (void)vcache;
    (void)seqlens_k;
    (void)block_table;
    (void)num_heads;
    (void)num_kv_heads;
    (void)head_size;
    (void)softmax_scale;
    (void)seqlens_k_host;
    (void)block_table_host;
    require_enabled();
#endif
}

void paged_attention_prefill_(const infinicore::Tensor &out,
                              const infinicore::Tensor &q,
                              const infinicore::Tensor &kcache,
                              const infinicore::Tensor &vcache,
                              const infinicore::Tensor &block_tables,
                              const infinicore::Tensor &sequence_lengths,
                              const infinicore::Tensor &input_offsets,
                              std::optional<infinicore::Tensor> alibi_slopes,
                              float softmax_scale) {
    ignore_args(input_offsets, alibi_slopes);
    paged_attention(
        out, q, kcache, vcache, sequence_lengths, block_tables,
        static_cast<int64_t>(q->size(1)),
        static_cast<int64_t>(kcache->size(kcache->ndim() - 2)),
        static_cast<int64_t>(q->size(q->ndim() - 1)),
        softmax_scale);
}

void paged_attention_(const infinicore::Tensor &out,
                      const infinicore::Tensor &q,
                      const infinicore::Tensor &kcache,
                      const infinicore::Tensor &vcache,
                      const infinicore::Tensor &block_tables,
                      const infinicore::Tensor &sequence_lengths,
                      std::optional<infinicore::Tensor> alibi_slopes,
                      float softmax_scale) {
    ignore_args(alibi_slopes);
    paged_attention(
        out, q, kcache, vcache, sequence_lengths, block_tables,
        static_cast<int64_t>(q->size(1)),
        static_cast<int64_t>(kcache->size(kcache->ndim() - 2)),
        static_cast<int64_t>(q->size(q->ndim() - 1)),
        softmax_scale);
}

infinicore::Tensor flash_attention(const infinicore::Tensor &q,
                                   const infinicore::Tensor &k,
                                   const infinicore::Tensor &v,
                                   const infinicore::Tensor &sequence_lengths,
                                   float softmax_scale,
                                   bool causal) {
    ignore_args(q, k, v, sequence_lengths, softmax_scale, causal);
    missing_operator("flash_attention");
}

infinicore::Tensor matmul(const infinicore::Tensor &a,
                          const infinicore::Tensor &b,
                          float scale) {
#ifdef INFINILM_ENABLE_INFINIOPS
    require_same_dtype_device(a, b, "matmul");
    auto out = infinicore::Tensor::empty(matmul_output_shape(a, b), a->dtype(), a->device());
    auto a_g = infinicore::graph::GraphTensor(a);
    auto b_g = infinicore::graph::GraphTensor(b);
    auto out_g = infinicore::graph::GraphTensor(out);
    record_or_run([a_g, b_g, out_g]() {
        infini::ops::Matmul::Call(handle(), config(), as_tensor(a_g), as_tensor(b_g), as_tensor(out_g), false, false);
    });

    if (std::abs(scale - 1.0f) <= 1e-6f) {
        return out;
    }

    auto scale_value = scalar_tensor(out->dtype(), out->device(), scale);
    auto scaled = infinicore::Tensor::empty(out->shape(), out->dtype(), out->device());
    auto scale_g = infinicore::graph::GraphTensor(scale_value);
    auto scaled_g = infinicore::graph::GraphTensor(scaled);
    record_or_run([out_g, scale_value, scale_g, scaled_g]() {
        (void)scale_value;
        infini::ops::Mul::Call(handle(), config(), as_tensor(out_g), as_tensor(scale_g), as_tensor(scaled_g));
    });
    return scaled;
#else
    (void)a;
    (void)b;
    (void)scale;
    require_enabled();
    return {};
#endif
}

infinicore::Tensor conv2d(const infinicore::Tensor &input,
                          const infinicore::Tensor &weight,
                          const infinicore::Tensor &bias,
                          const std::vector<size_t> &pads,
                          const std::vector<size_t> &strides,
                          const std::vector<size_t> &dilations) {
    ignore_args(input, weight, bias, pads, strides, dilations);
    missing_operator("conv2d");
}

void softmax_(const infinicore::Tensor &out,
              const infinicore::Tensor &input,
              int axis) {
#ifdef INFINILM_ENABLE_INFINIOPS
    require_same_dtype_device(input, out, "softmax");
    require_shape(out->shape(), input->shape(), "softmax");
    if (input->ndim() == 0) {
        throw std::runtime_error("InfiniOps adapter: softmax requires at least 1 dimension");
    }
    const auto last_axis = static_cast<int>(input->ndim()) - 1;
    if (axis != -1 && axis != last_axis) {
        throw std::runtime_error("InfiniOps adapter: softmax currently supports only the last axis");
    }
    require_contiguous(input, "softmax input");
    require_contiguous(out, "softmax output");
    const auto vocab = input->size(input->ndim() - 1);
    auto input_2d = input->view({input->numel() / vocab, vocab});
    auto out_2d = out->view({out->numel() / vocab, vocab});
    auto input_g = infinicore::graph::GraphTensor(input_2d);
    auto out_g = infinicore::graph::GraphTensor(out_2d);
    record_or_run([input_g, out_g]() {
        infini::ops::ScaledSoftmax::Call(handle(), config(), as_tensor(input_g), 1.0, as_tensor(out_g));
    });
#else
    (void)out;
    (void)input;
    (void)axis;
    require_enabled();
#endif
}

infinicore::Tensor gelu(const infinicore::Tensor &input) {
    ignore_args(input);
    missing_operator("gelu");
}

infinicore::Tensor gelu_tanh(const infinicore::Tensor &input) {
    ignore_args(input);
    missing_operator("gelu_tanh");
}

infinicore::Tensor relu(const infinicore::Tensor &input) {
    ignore_args(input);
    missing_operator("relu");
}

void causal_softmax_(const infinicore::Tensor &out,
                     const infinicore::Tensor &input) {
#ifdef INFINILM_ENABLE_INFINIOPS
    require_same_dtype_device(input, out, "causal_softmax");
    require_shape(out->shape(), input->shape(), "causal_softmax");
    auto input_g = infinicore::graph::GraphTensor(input);
    auto out_g = infinicore::graph::GraphTensor(out);
    record_or_run([input_g, out_g]() {
        infini::ops::CausalSoftmax::Call(handle(), config(), as_tensor(input_g), as_tensor(out_g));
    });
#else
    (void)out;
    (void)input;
    require_enabled();
#endif
}

void mha_varlen_(const infinicore::Tensor &out,
                 const infinicore::Tensor &q,
                 const infinicore::Tensor &k,
                 const infinicore::Tensor &v,
                 const infinicore::Tensor &cu_seqlens_q,
                 const infinicore::Tensor &cu_seqlens_k,
                 const infinicore::Tensor &block_table,
                 int64_t max_seqlen_q,
                 int64_t max_seqlen_k,
                 std::optional<infinicore::Tensor> alibi_slopes,
                 float softmax_scale) {
    ignore_args(alibi_slopes);
    mha_varlen_fwd(out, q, k, v, cu_seqlens_q, cu_seqlens_k, block_table,
                   max_seqlen_q, max_seqlen_k, softmax_scale);
}

infinicore::Tensor mha_kvcache(const infinicore::Tensor &q,
                               const infinicore::Tensor &kcache,
                               const infinicore::Tensor &vcache,
                               const infinicore::Tensor &seqlens_k,
                               const infinicore::Tensor &block_table,
                               std::optional<infinicore::Tensor> alibi_slopes,
                               float softmax_scale) {
    ignore_args(alibi_slopes);
    auto out = infinicore::Tensor::empty(q->shape(), q->dtype(), q->device());
    mha_fwd_kvcache(out, q, kcache, vcache, seqlens_k, block_table, softmax_scale);
    return out;
}

infinicore::Tensor per_tensor_quant_i8(const infinicore::Tensor &input,
                                       const infinicore::Tensor &scale,
                                       const infinicore::Tensor &zero_point,
                                       bool saturate) {
    ignore_args(input, scale, zero_point, saturate);
    missing_operator("per_tensor_quant_i8");
}

void per_tensor_dequant_i8_(const infinicore::Tensor &out,
                            const infinicore::Tensor &input,
                            const infinicore::Tensor &scale,
                            const infinicore::Tensor &zero_point) {
    ignore_args(out, input, scale, zero_point);
    missing_operator("per_tensor_dequant_i8_");
}

infinicore::Tensor sample_from_logits(const infinicore::Tensor &logits,
                                      const infinicore::Tensor &input_offsets,
                                      float temperature,
                                      int top_k,
                                      float top_p) {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (!should_use(logits->device())) {
        throw std::runtime_error("InfiniOps adapter: logits sampling is only implemented for Ascend");
    }
    if (temperature <= 0.0f) {
        throw std::runtime_error("InfiniOps adapter: sampling temperature must be positive");
    }
    if (logits->dtype() != infinicore::DataType::F16 && logits->dtype() != infinicore::DataType::BF16) {
        throw std::runtime_error("InfiniOps adapter: Ascend sampling requires float16 or bfloat16 logits");
    }
    if (logits->ndim() != 3) {
        throw std::runtime_error("InfiniOps adapter: logits must be 3D [batch, seq, vocab]");
    }

    auto offsets_cpu = input_offsets->device().getType() == infinicore::Device::Type::CPU
                         ? input_offsets
                         : input_offsets->to(infinicore::Device::cpu());
    offsets_cpu = offsets_cpu->is_contiguous() ? offsets_cpu : offsets_cpu->contiguous();
    if (offsets_cpu->dtype() != infinicore::DataType::I32) {
        throw std::runtime_error("InfiniOps adapter: input_offsets must be int32");
    }

    const auto batch_size = logits->size(0);
    const auto total_len = logits->size(1);
    const auto vocab_size = logits->size(2);
    const auto n_req = offsets_cpu->size(0) - 1;
    const auto effective_top_k = top_k <= 0 ? static_cast<int64_t>(vocab_size)
                                            : std::min<int64_t>(top_k, static_cast<int64_t>(vocab_size));
    const double scale = 1.0 / static_cast<double>(temperature);
    const bool needs_scale = std::abs(scale - 1.0) > 1e-6;
    std::optional<infinicore::Tensor> scale_value;
    if (needs_scale) {
        scale_value = scalar_tensor(logits->dtype(), logits->device(), static_cast<float>(scale));
    }

    int64_t top_k_value = effective_top_k;
    float top_p_value = top_p;
    std::optional<infini::ops::Tensor> top_k_tensor;
    std::optional<infini::ops::Tensor> top_p_tensor;
    if (top_k > 0) {
        top_k_tensor.emplace(
            &top_k_value, std::vector<std::size_t>{1},
            infini::ops::DataType::kInt64,
            infini::ops::Device{infini::ops::Device::Type::kCpu});
    }
    if (top_p < 1.0f) {
        top_p_tensor.emplace(
            &top_p_value, std::vector<std::size_t>{1},
            infini::ops::DataType::kFloat32,
            infini::ops::Device{infini::ops::Device::Type::kCpu});
    }

    auto logits_2d = logits->view({batch_size * total_len, vocab_size});
    auto sampled_i32 = infinicore::Tensor::empty({n_req}, infinicore::DataType::I32, logits->device());
    const auto *offsets = reinterpret_cast<const int32_t *>(offsets_cpu->data());

    for (std::size_t i = 0; i < n_req; ++i) {
        const auto row = offsets[i + 1] - 1;
        if (row < 0 || static_cast<size_t>(row) >= batch_size * total_len) {
            throw std::runtime_error("InfiniOps adapter: input_offsets contains an invalid row");
        }

        auto score = logits_2d->narrow({{0, static_cast<size_t>(row), 1}});
        require_contiguous(score, "sampling logits row");
        auto sampler_logits = score;
        if (scale_value.has_value()) {
            auto scaled_score = infinicore::Tensor::empty(score->shape(), score->dtype(), score->device());
            infini::ops::Mul::Call(handle(), config(), as_tensor(score), as_tensor(scale_value.value()), as_tensor(scaled_score));
            sampler_logits = scaled_score;
        }

        auto out = sampled_i32->narrow({{0, i, 1}});
        const auto sampler_config = top_k == 1 ? config() : config(1);
        infini::ops::TopKTopPSampler::Call(
            handle(), sampler_config, as_tensor(sampler_logits), top_k_tensor,
            top_p_tensor, as_tensor(out));
    }

    // ATB sampling can complete outside the stream dependency observed by the
    // following D2H copy, so wait at device scope before reading token ids.
    infinicore::context::syncDevice();
    auto sampled_i32_cpu = sampled_i32->to(infinicore::Device::cpu());
    infinicore::context::syncStream();

    auto sampled_i64_cpu = infinicore::Tensor::empty({n_req}, infinicore::DataType::I64, infinicore::Device::cpu());
    const auto *src = reinterpret_cast<const int32_t *>(sampled_i32_cpu->data());
    auto *dst = reinterpret_cast<int64_t *>(sampled_i64_cpu->data());
    for (std::size_t i = 0; i < n_req; ++i) {
        dst[i] = static_cast<int64_t>(src[i]);
    }

    infinicore::context::setDevice(logits->device());
    return sampled_i64_cpu;
#else
    (void)logits;
    (void)input_offsets;
    (void)temperature;
    (void)top_k;
    (void)top_p;
    require_enabled();
    return {};
#endif
}

void random_sample_(const infinicore::Tensor &out,
                    const infinicore::Tensor &score,
                    float random_val,
                    float top_p,
                    int top_k,
                    float temperature) {
    ignore_args(out, score, random_val, top_p, top_k, temperature);
    missing_operator("random_sample_");
}

void allreduce_(const infinicore::Tensor &out,
                const infinicore::Tensor &input,
                infinicclReduceOp_t op,
                infinicclComm_t communicator) {
    if (communicator == nullptr) {
        throw std::runtime_error("InfiniLM ops wrapper: allreduce communicator is null");
    }
    auto status = infinicclAllReduce(
        const_cast<std::byte *>(input->data()),
        const_cast<std::byte *>(out->data()),
        out->numel(),
        static_cast<infiniDtype_t>(out->dtype()),
        op,
        communicator,
        infinicore::context::getStream());
    if (status != INFINI_STATUS_SUCCESS) {
        throw std::runtime_error("InfiniLM ops wrapper: infinicclAllReduce failed");
    }
}

infinicore::Tensor Embedding::forward(const infinicore::Tensor &indices) const {
    if (should_use(weight()->device())) {
        return embedding(indices, weight());
    }
    return infinicore::nn::Embedding::forward(indices);
}

infinicore::Tensor RMSNorm::forward(const infinicore::Tensor &x) const {
    if (should_use(x->device())) {
        auto out = infinicore::Tensor::empty(x->shape(), x->dtype(), x->device());
        rms_norm_(out, x, weight(), static_cast<float>(eps()));
        return out;
    }
    return infinicore::nn::RMSNorm::forward(x);
}

void RMSNorm::forward_inplace(infinicore::Tensor &x, infinicore::Tensor &residual) const {
    if (!should_use(x->device())) {
        infinicore::nn::RMSNorm::forward_inplace(x, residual);
        return;
    }

    if (!residual) {
        residual = x;
        x = forward(x);
        return;
    }

    add_rms_norm_(x, residual, weight(), static_cast<float>(eps()));
}

infinicore::Tensor ReplicatedLinear::forward(infinicore::Tensor &input) const {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (should_use(input->device()) && unquantized(*this)) {
        std::optional<infinicore::Tensor> bias_opt = has_bias() ? std::make_optional(bias()) : std::nullopt;
        return linear(input, weight(), bias_opt);
    }
#endif
    return infinicore::nn::Linear::forward(input);
}

infinicore::Tensor ColumnParallelLinear::forward(infinicore::Tensor &input) const {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (should_use(input->device()) && unquantized(*this)) {
        std::optional<infinicore::Tensor> bias_opt = has_bias() ? std::make_optional(bias()) : std::nullopt;
        return linear(input, weight(), bias_opt);
    }
#endif
    return infinicore::nn::ColumnParallelLinear::forward(input);
}

infinicore::Tensor RowParallelLinear::forward(infinicore::Tensor &input) const {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (should_use(input->device()) && unquantized(*this)) {
        std::optional<infinicore::Tensor> bias_opt = has_bias() ? std::make_optional(bias()) : std::nullopt;
        auto output = linear(input, weight(), bias_opt);
        if ((tp_size_ > 1) && (communicator_ != nullptr)) {
            allreduce_(output, output, INFINICCL_SUM, communicator_);
        }
        return output;
    }
#endif
    return infinicore::nn::RowParallelLinear::forward(input);
}

} // namespace infinilm::backends::ops
