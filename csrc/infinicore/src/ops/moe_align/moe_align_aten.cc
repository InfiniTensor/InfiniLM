#include "infinicore/ops/moe_align.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ATen.h>

#include <cstdint>
#include <vector>

namespace infinicore::op::moe_align_impl::aten {
namespace {

void run_moe_align_host(at::Tensor sorted_token_ids,
                        at::Tensor expert_ids,
                        at::Tensor num_tokens_post_padded,
                        at::Tensor topk_ids,
                        const at::Tensor *expert_map,
                        int64_t num_experts,
                        int64_t block_size,
                        bool pad_sorted_token_ids) {
    auto topk_cpu = topk_ids.to(at::kCPU).contiguous().to(at::kLong).flatten();
    const int64_t numel = topk_cpu.numel();
    const auto *topk_ptr = topk_cpu.data_ptr<int64_t>();

    std::vector<int64_t> mapped(static_cast<size_t>(numel));
    if (expert_map != nullptr) {
        auto map_cpu = expert_map->to(at::kCPU).contiguous().to(at::kLong).flatten();
        INFINICORE_ASSERT(map_cpu.numel() >= num_experts);
        const auto *map_ptr = map_cpu.data_ptr<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
            const int64_t eid = topk_ptr[i];
            if (eid >= 0 && eid < num_experts) {
                mapped[static_cast<size_t>(i)] = map_ptr[eid];
            } else {
                mapped[static_cast<size_t>(i)] = -1;
            }
        }
    } else {
        for (int64_t i = 0; i < numel; ++i) {
            mapped[static_cast<size_t>(i)] = topk_ptr[i];
        }
    }

    std::vector<int32_t> sorted;
    std::vector<int32_t> blocks;
    sorted.reserve(static_cast<size_t>(numel + num_experts * (block_size - 1)));
    blocks.reserve(static_cast<size_t>((numel + block_size - 1) / block_size + num_experts));

    for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        std::vector<int32_t> token_ids;
        for (int64_t i = 0; i < numel; ++i) {
            if (mapped[static_cast<size_t>(i)] == expert_id) {
                token_ids.push_back(static_cast<int32_t>(i));
            }
        }
        if (token_ids.empty()) {
            continue;
        }
        const int64_t num_blocks = (static_cast<int64_t>(token_ids.size()) + block_size - 1) / block_size;
        const int64_t padded = num_blocks * block_size;
        sorted.insert(sorted.end(), token_ids.begin(), token_ids.end());
        sorted.insert(sorted.end(), static_cast<size_t>(padded - token_ids.size()), static_cast<int32_t>(numel));
        blocks.insert(blocks.end(), static_cast<size_t>(num_blocks), static_cast<int32_t>(expert_id));
    }

    const int64_t num_tokens_post = static_cast<int64_t>(blocks.size()) * block_size;
    auto sorted_out = sorted_token_ids.to(at::kCPU).contiguous();
    auto expert_out = expert_ids.to(at::kCPU).contiguous();
    auto num_out = num_tokens_post_padded.to(at::kCPU).contiguous();

    INFINICORE_ASSERT(sorted_out.numel() >= num_tokens_post);
    INFINICORE_ASSERT(expert_out.numel() >= static_cast<int64_t>(blocks.size()));
    INFINICORE_ASSERT(num_out.numel() >= 1);

    auto *sorted_ptr = sorted_out.data_ptr<int32_t>();
    auto *expert_ptr = expert_out.data_ptr<int32_t>();
    auto *num_ptr = num_out.data_ptr<int32_t>();

    if (pad_sorted_token_ids) {
        std::fill(sorted_ptr, sorted_ptr + sorted_out.numel(), static_cast<int32_t>(numel));
    }
    std::fill(expert_ptr, expert_ptr + expert_out.numel(), static_cast<int32_t>(-1));

    if (num_tokens_post > 0) {
        std::copy(sorted.begin(), sorted.begin() + num_tokens_post, sorted_ptr);
    }
    std::copy(blocks.begin(), blocks.end(), expert_ptr);
    num_ptr[0] = static_cast<int32_t>(num_tokens_post);

    sorted_token_ids.copy_(sorted_out.to(sorted_token_ids.device()));
    expert_ids.copy_(expert_out.to(expert_ids.device()));
    num_tokens_post_padded.copy_(num_out.to(num_tokens_post_padded.device()));
}

struct PlannedMeta {
    Tensor sorted_token_ids;
    Tensor expert_ids;
    Tensor num_tokens_post_padded;
    Tensor topk_ids;
    Tensor expert_map;
    bool has_expert_map;
    size_t num_experts;
    size_t block_size;
    bool pad_sorted_token_ids;
};

void *plan_no_map(Tensor sorted_token_ids,
                  Tensor expert_ids,
                  Tensor num_tokens_post_padded,
                  const Tensor &topk_ids,
                  const size_t num_experts,
                  const size_t block_size,
                  const bool pad_sorted_token_ids) {
    INFINICORE_ASSERT(sorted_token_ids->device().type() == Device::Type::kQy);
    return new PlannedMeta{
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        Tensor{},
        false,
        num_experts,
        block_size,
        pad_sorted_token_ids};
}

void *plan_with_map(Tensor sorted_token_ids,
                    Tensor expert_ids,
                    Tensor num_tokens_post_padded,
                    const Tensor &topk_ids,
                    const Tensor &expert_map,
                    const size_t num_experts,
                    const size_t block_size,
                    const bool pad_sorted_token_ids) {
    INFINICORE_ASSERT(sorted_token_ids->device().type() == Device::Type::kQy);
    return new PlannedMeta{
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        expert_map,
        true,
        num_experts,
        block_size,
        pad_sorted_token_ids};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    using namespace ::infinicore::adaptor;
    set_aten_stream_to_infinicore();

    at::Tensor expert_map_aten;
    const at::Tensor *expert_map_ptr = nullptr;
    if (p->has_expert_map) {
        expert_map_aten = to_aten_tensor(p->expert_map);
        expert_map_ptr = &expert_map_aten;
    }

    run_moe_align_host(
        to_aten_tensor(p->sorted_token_ids),
        to_aten_tensor(p->expert_ids),
        to_aten_tensor(p->num_tokens_post_padded),
        to_aten_tensor(p->topk_ids),
        expert_map_ptr,
        static_cast<int64_t>(p->num_experts),
        static_cast<int64_t>(p->block_size),
        p->pad_sorted_token_ids);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    MoeAlign::plan_dispatcher().registerDevice(Device::Type::kQy, &plan_no_map, true);
    MoeAlign::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MoeAlign::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);

    MoeAlignWithExpertMap::plan_dispatcher().registerDevice(Device::Type::kQy, &plan_with_map, true);
    MoeAlignWithExpertMap::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MoeAlignWithExpertMap::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::moe_align_impl::aten
#endif
