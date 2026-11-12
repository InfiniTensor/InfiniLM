#include "vocab_parallel_lm_head.hpp"

#include "../../global_state/global_state.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <infinicore/context/context.hpp>
#include <infinicore/graph/graph.hpp>
#include <infinicore/ops/rearrange.hpp>

namespace infinilm::layers::linear {
namespace {

bool supported_dtype(const infinicore::DataType &dtype) {
    return dtype == infinicore::DataType::kBFloat16
           || dtype == infinicore::DataType::kFloat16;
}

bool vocab_parallel_env_enabled() {
    const char *flag = std::getenv("INFINI_VOCAB_PARALLEL");
    // Unset or "1" → allow eligible TP sharding; "0" forces unreplicated.
    return flag == nullptr || std::strcmp(flag, "1") == 0;
}

bool eligible(size_t vocab, const infinicore::DataType &dtype,
              const infinicore::Device &device, int world, int pp_size) {
    if (!vocab_parallel_env_enabled()) {
        return false;
    }
    return (world == 2 || world == 4) && pp_size == 1
           && vocab % static_cast<size_t>(world) == 0 && supported_dtype(dtype)
           && device.type() == infinicore::Device::Type::kQy;
}

infinicclDataType_t to_ccl_dtype(const infinicore::DataType &dtype) {
    if (dtype == infinicore::DataType::kFloat16) {
        return infinicclFloat16;
    }
    if (dtype == infinicore::DataType::kBFloat16) {
        return infinicclBFloat16;
    }
    throw std::invalid_argument("vocab gather requires BF16/FP16");
}

void check(infinicclResult_t status) {
    if (status != infinicclSuccess) {
        throw std::runtime_error("vocab gather: InfiniCCL error "
                                 + std::to_string(static_cast<int>(status)));
    }
}

// Root=0 gather via NCCL Send/Recv (Denglin NCCL has no Gather).
class VocabGather : public infinicore::graph::GraphOperator {
public:
    VocabGather(const infinicore::Tensor &local,
                const infinicore::Tensor &packed,
                int rank, int world, infinicclComm_t comm)
        : local_(local), packed_(packed), rank_(rank), world_(world), comm_(comm) {}

    void run() const override {
        const auto count = local_->numel();
        const auto type = to_ccl_dtype(local_->dtype());
        const auto stream = reinterpret_cast<void *>(infinicore::context::getStream());
        if (rank_ == 0) {
            auto *base = reinterpret_cast<char *>(packed_->data());
            const size_t bytes = count * local_->element_size();
            for (int peer = 1; peer < world_; ++peer) {
                check(infinicclRecv(base + static_cast<size_t>(peer) * bytes,
                                    count, type, peer, comm_, stream));
            }
        } else {
            check(infinicclSend(local_->data(), count, type, /*peer=*/0, comm_, stream));
        }
    }

private:
    infinicore::graph::GraphTensor local_;
    mutable infinicore::graph::GraphTensor packed_;
    int rank_, world_;
    infinicclComm_t comm_;
};

} // namespace

size_t vocab_parallel_logits_bank_width(size_t vocab_size) {
    const auto &rank_info =
        infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int world = rank_info.tp_size;
    const int rank = rank_info.tp_rank;
    const auto device = infinicore::context::getDevice();
    const auto dtype =
        infinilm::global_state::get_infinilm_config().model_config->get_dtype();
    if (!eligible(vocab_size, dtype, device, world, rank_info.pp_size)) {
        return vocab_size;
    }
    // Root gathers to full vocab; non-root keeps the local shard.
    return rank == 0 ? vocab_size
                     : vocab_size / static_cast<size_t>(world);
}

infinicore::Tensor gather_vocab_logits(const infinicore::Tensor &local,
                                       int rank, int world,
                                       infinicclComm_t communicator) {
    if (!local->is_contiguous() || !supported_dtype(local->dtype())
        || local->ndim() == 0 || rank < 0 || rank >= world || !communicator) {
        throw std::invalid_argument(
            "vocab gather requires contiguous BF16/FP16 logits and a valid TP group");
    }
    const auto width = local->shape().back();
    const auto rows = local->numel() / width;
    auto packed = rank == 0
                      ? infinicore::Tensor::empty(
                            {static_cast<size_t>(world), rows, width},
                            local->dtype(), local->device())
                      : local;
    if (rank == 0) {
        auto first = packed->narrow({{0, 0, 1}})->view(local->shape());
        infinicore::op::rearrange_(first, local);
    }
    auto op = std::make_shared<VocabGather>(local, packed, rank, world, communicator);
    if (infinicore::context::isGraphRecording()) {
        infinicore::context::addGraphOperator(op);
    } else {
        op->run();
    }
    if (rank != 0) {
        return local;
    }
    auto shape = local->shape();
    shape.back() *= static_cast<size_t>(world);
    return packed->permute({1, 0, 2})->contiguous()->view(shape);
}

VocabParallelLMHead::VocabParallelLMHead(
    size_t in_features, size_t vocab_size, bool bias,
    const infinicore::DataType &dtype, const infinicore::Device &device,
    int rank, int world, infinicclComm_t communicator, int pp_size)
    : BaseLinear(in_features, vocab_size,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, device,
                 eligible(vocab_size, dtype, device, world, pp_size) ? 0 : -1,
                 eligible(vocab_size, dtype, device, world, pp_size) ? rank : 0,
                 eligible(vocab_size, dtype, device, world, pp_size) ? world : 1),
      rank_(rank),
      world_(eligible(vocab_size, dtype, device, world, pp_size) ? world : 1),
      communicator_(communicator) {
    if (!vocab_parallel_env_enabled()
        && (world == 2 || world == 4)
        && device.type() == infinicore::Device::Type::kQy) {
        spdlog::info("[vocab-parallel] disabled by env INFINI_VOCAB_PARALLEL=0 "
                     "(unreplicated lm_head at tp={})",
                     world);
    }
    if (world_ > 1) {
        if (!communicator_) {
            throw std::invalid_argument("vocab parallel requires a TP communicator");
        }
        spdlog::info(
            "[vocab-parallel] rank={} world={} local_vocab={} weight_shape=[{},{}] "
            "dtype={} gather_root=0",
            rank_, world_, vocab_size / static_cast<size_t>(world_),
            weight()->shape()[0], weight()->shape()[1],
            dtype == infinicore::DataType::kFloat16 ? "float16" : "bfloat16");
    }
}

infinicore::Tensor VocabParallelLMHead::forward(infinicore::Tensor &input) const {
    auto local = BaseLinear::forward(input);
    return world_ == 1 ? local : gather_vocab_logits(local, rank_, world_, communicator_);
}

} // namespace infinilm::layers::linear
