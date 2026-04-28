#include <musa_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"

#include "../cuda/kernel_v2.cuh"

namespace op::paged_attention::moore {

namespace {
constexpr int kMaxSplits = 8;

constexpr size_t ceilDiv(size_t a, size_t b) {
    return (a + b - 1) / b;
}

inline int getSmCount() {
    int device = 0;
    if (musaGetDevice(&device) != musaSuccess) {
        return 0;
    }
    int sm_count = 0;
    if (musaDeviceGetAttribute(&sm_count, musaDevAttrMultiProcessorCount, device) != musaSuccess) {
        return 0;
    }
    return sm_count;
}

// A lightweight FA2-style "waves" heuristic.
//
// Important: our split-kv kernel shards the KV sequence length, so the main "work"
// dimension is tokens, not the number of pages. We use an upper bound for seqlen_k
// (max pages * page size), which matches common decode microbench where all seqs
// share the same cache length.
inline int chooseNumSplitsHeuristic(size_t num_heads, size_t num_seqs, size_t seqlen_k, int sm_count) {
    if (sm_count <= 0) {
        return 1;
    }
    if (num_heads == 0 || num_seqs == 0) {
        return 1;
    }
    if (seqlen_k <= 256) {
        return 1;
    }

    const size_t base_blocks = num_heads * num_seqs;
    int best_splits = 1;
    // Baseline: one kernel, base_blocks CTAs, each scanning seqlen_k tokens.
    size_t best_score = (ceilDiv(base_blocks, static_cast<size_t>(sm_count)) * seqlen_k);

    size_t prev_work_per_block = seqlen_k;
    for (int s = 2; s <= kMaxSplits; ++s) {
        const size_t blocks = base_blocks * static_cast<size_t>(s);
        const size_t waves_split = ceilDiv(blocks, static_cast<size_t>(sm_count));
        const size_t work_per_block = ceilDiv(seqlen_k, static_cast<size_t>(s));
        // If this split count doesn't reduce per-block work vs the previous split, it's effectively redundant.
        if (work_per_block == prev_work_per_block) {
            continue;
        }
        prev_work_per_block = work_per_block;
        // Combine is one extra kernel with base_blocks blocks; approximate as one more wave unit.
        const size_t waves_combine = ceilDiv(base_blocks, static_cast<size_t>(sm_count));
        const size_t score = waves_split * work_per_block + waves_combine;
        if (score < best_score) {
            best_score = score;
            best_splits = s;
        }
    }
    return best_splits;
}
} // namespace

inline bool envBool(const char *name) {
    if (const char *env = std::getenv(name)) {
        return (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
    }
    return false;
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128Warp(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
    op::paged_attention::cuda::flashAttentionDecodeWarpKernel<Tindex, Tdata, 128>(
        out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, o_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128Cta(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
    // Default CTA variant (lower overhead).
    op::paged_attention::cuda::flashAttentionDecodeCtaKernel<Tindex, Tdata, 128, 64, 8>(
        out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, o_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128CtaTile16(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
    op::paged_attention::cuda::flashAttentionDecodeCtaKernel<Tindex, Tdata, 128, 64, 16>(
        out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, o_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128Cta32(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
    // Experimental 1-warp CTA variant for head_dim=128 (kPack=4).
    op::paged_attention::cuda::flashAttentionDecodeCtaKernel<Tindex, Tdata, 128, 32, 8>(
        out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, o_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128Cta32Tile16(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
    op::paged_attention::cuda::flashAttentionDecodeCtaKernel<Tindex, Tdata, 128, 32, 16>(
        out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, o_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128CtaGqa4(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
    // GQA fused kernel: CTA computes 4 query heads for one KV head (head_dim=128).
    op::paged_attention::cuda::flashAttentionDecodeCtaGqaKernel<Tindex, Tdata, 128, 64, 8, 4>(
        out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, o_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128SplitKv(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    int num_splits) {
    op::paged_attention::cuda::flashAttentionDecodeSplitKvWarpKernel<Tindex, Tdata, 128>(
        partial_acc, partial_m, partial_l,
        q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, num_splits);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128SplitKvCta(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    int num_splits) {
    op::paged_attention::cuda::flashAttentionDecodeSplitKvCtaKernel<Tindex, Tdata, 128, 64, 8>(
        partial_acc, partial_m, partial_l,
        q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, num_splits);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128SplitKvCtaTile16(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    int num_splits) {
    op::paged_attention::cuda::flashAttentionDecodeSplitKvCtaKernel<Tindex, Tdata, 128, 64, 16>(
        partial_acc, partial_m, partial_l,
        q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, num_splits);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128SplitKvCta32(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    int num_splits) {
    op::paged_attention::cuda::flashAttentionDecodeSplitKvCtaKernel<Tindex, Tdata, 128, 32, 8>(
        partial_acc, partial_m, partial_l,
        q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, num_splits);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128SplitKvCta32Tile16(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    int num_splits) {
    op::paged_attention::cuda::flashAttentionDecodeSplitKvCtaKernel<Tindex, Tdata, 128, 32, 16>(
        partial_acc, partial_m, partial_l,
        q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, num_splits);
}

template <typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd128SplitKvCombine(
    Tdata *out,
    const float *partial_acc,
    const float *partial_m,
    const float *partial_l,
    int num_splits,
    ptrdiff_t o_stride) {
    op::paged_attention::cuda::flashAttentionDecodeSplitKvCombineWarpKernel<Tdata, 128>(
        out, partial_acc, partial_m, partial_l, num_splits, o_stride);
}

template <typename Tindex>
infiniStatus_t launch_decode_hd128_impl(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    infiniDtype_t dtype,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    musaStream_t stream) {

    // Default decode config (2026-01-22):
    //   decode_flash_cta8_64_gqa_splitkv_4
    // Users can override any knob via the corresponding INFINIOP_FLASH_* env vars.
    bool use_cta = true;
    if (const char *env = std::getenv("INFINIOP_FLASH_DECODE_KERNEL")) {
        // Backward-compatible: any non-"cta" value means "warp".
        use_cta = (std::strcmp(env, "cta") == 0);
    }
    bool use_gqa_fused = true;
    if (const char *env = std::getenv("INFINIOP_FLASH_GQA_FUSED")) {
        if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0) {
            use_gqa_fused = false;
        } else {
            use_gqa_fused = (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
        }
    }
    int cta_tile = 8;
    if (const char *env = std::getenv("INFINIOP_FLASH_CTA_TILE")) {
        const int v = std::atoi(env);
        if (v == 8 || v == 16) {
            cta_tile = v;
        }
    }
    int cta_threads = 64;
    if (const char *env = std::getenv("INFINIOP_FLASH_CTA_THREADS")) {
        const int v = std::atoi(env);
        if (v == 32 || v == 64) {
            cta_threads = v;
        }
    }
    dim3 block(use_cta ? static_cast<uint32_t>(cta_threads) : 32);

    bool use_split = true;
    bool use_split_auto = false;
    if (const char *env = std::getenv("INFINIOP_FLASH_DECODE_SPLITKV")) {
        if (std::strcmp(env, "auto") == 0) {
            use_split_auto = true;
            use_split = false;
        } else {
            if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0) {
                use_split = false;
            } else {
                use_split = (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
            }
        }
    }
    int num_splits = 4;
    bool fixed_num_splits = true;
    if (const char *env = std::getenv("INFINIOP_FLASH_NUM_SPLITS")) {
        if (std::strcmp(env, "auto") == 0) {
            fixed_num_splits = false;
        } else {
            num_splits = std::atoi(env);
            fixed_num_splits = (num_splits > 0);
        }
    }
    if (num_splits < 1) {
        num_splits = 1;
    }
    if (num_splits > kMaxSplits) {
        num_splits = kMaxSplits;
    }

    const bool debug_dispatch = envBool("INFINIOP_FLASH_DEBUG_DISPATCH");
    auto dump_dispatch = [&](const char *path) {
        if (!debug_dispatch) {
            return;
        }
        // Avoid spamming: only print when the key dispatch signature changes.
        struct Sig {
            const char *path;
            int dtype;
            size_t heads;
            size_t kv_heads;
            size_t seqs;
            size_t pbs;
            size_t max_blocks;
            int cta_tile;
            int cta_threads;
            int split;
            int split_auto;
            int num_splits;
            int fixed;
            int gqa_fused;
        };
        static Sig last{};
        static bool has_last = false;

        Sig cur{
            path,
            static_cast<int>(dtype),
            num_heads,
            num_kv_heads,
            num_seqs,
            page_block_size,
            max_num_blocks_per_seq,
            cta_tile,
            cta_threads,
            static_cast<int>(use_split),
            static_cast<int>(use_split_auto),
            num_splits,
            static_cast<int>(fixed_num_splits),
            static_cast<int>(use_gqa_fused),
        };

        if (has_last && cur.path == last.path && cur.dtype == last.dtype && cur.heads == last.heads && cur.kv_heads == last.kv_heads && cur.seqs == last.seqs && cur.pbs == last.pbs && cur.max_blocks == last.max_blocks && cur.cta_tile == last.cta_tile && cur.cta_threads == last.cta_threads && cur.split == last.split && cur.split_auto == last.split_auto && cur.num_splits == last.num_splits && cur.fixed == last.fixed && cur.gqa_fused == last.gqa_fused) {
            return;
        }
        last = cur;
        has_last = true;

        fprintf(stderr,
                "[INFINIOP][paged_attention][hd128] dispatch: path=%s dtype=%d heads=%zu kv_heads=%zu seqs=%zu "
                "pbs=%zu max_blocks=%zu cta_tile=%d cta_threads=%d split=%d split_auto=%d num_splits=%d fixed=%d gqa_fused=%d\n",
                path, static_cast<int>(dtype), num_heads, num_kv_heads, num_seqs,
                page_block_size, max_num_blocks_per_seq, cta_tile, cta_threads,
                static_cast<int>(use_split), static_cast<int>(use_split_auto), num_splits, static_cast<int>(fixed_num_splits),
                static_cast<int>(use_gqa_fused));
    };

    // Split-kv auto mode: decide whether to split based on a heuristic.
    if (use_split_auto) {
        // Approximate seqlen_k by the per-seq KV capacity (paged KV upper bound).
        const size_t seqlen_k = max_num_blocks_per_seq * page_block_size;
        const int sm_count = getSmCount();
        num_splits = chooseNumSplitsHeuristic(num_heads, num_seqs, seqlen_k, sm_count);
        if (const char *dbg = std::getenv("INFINIOP_FLASH_DEBUG_SPLITS")) {
            if (std::strcmp(dbg, "1") == 0 || std::strcmp(dbg, "true") == 0) {
                static size_t last_seqlen_k = 0;
                if (last_seqlen_k != seqlen_k) {
                    last_seqlen_k = seqlen_k;
                    fprintf(stderr,
                            "[INFINIOP][paged_attention] splitkv auto(mode): sm=%d heads=%zu seqs=%zu seqlen_k~%zu -> num_splits=%d\n",
                            sm_count, num_heads, num_seqs, seqlen_k, num_splits);
                }
            }
        }
        // If auto picks 1, fall back to non-split to avoid extra workspace and kernel overhead.
        use_split = (num_splits > 1);
    }

    // const bool debug_dispatch = [] {
    //     if (const char *env = std::getenv("INFINIOP_FLASH_DEBUG_DISPATCH")) {
    //         return (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
    //     }
    //     return false;
    // }();

    // const char *selected_path = "unknown";

    // Optional: fuse GQA groups (4) when seqlen_q=1 decode and alibi is disabled.
    // This reuses K/V loads across query heads that share the same KV head.
    // Controlled by INFINIOP_FLASH_GQA_FUSED (default: enabled).
    if (use_gqa_fused && use_cta && !use_split && alibi_slopes == nullptr && num_kv_heads > 0 && num_heads == num_kv_heads * 4) {
        dump_dispatch("cta_gqa_fused");
        dim3 grid_gqa(static_cast<uint64_t>(num_kv_heads), static_cast<uint64_t>(num_seqs), 1);
        if (dtype == INFINI_DTYPE_F16) {
            flashAttentionDecodeHd128CtaGqa4<Tindex, half><<<grid_gqa, 64, 0, stream>>>(
                static_cast<half *>(out),
                static_cast<const half *>(q),
                static_cast<const half *>(k_cache),
                static_cast<const half *>(v_cache),
                block_tables, cache_lens, nullptr,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                q_stride, k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride, o_stride);
            return INFINI_STATUS_SUCCESS;
        }
        if (dtype == INFINI_DTYPE_BF16) {
            flashAttentionDecodeHd128CtaGqa4<Tindex, __mt_bfloat16><<<grid_gqa, 64, 0, stream>>>(
                static_cast<__mt_bfloat16 *>(out),
                static_cast<const __mt_bfloat16 *>(q),
                static_cast<const __mt_bfloat16 *>(k_cache),
                static_cast<const __mt_bfloat16 *>(v_cache),
                block_tables, cache_lens, nullptr,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                q_stride, k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride, o_stride);
            return INFINI_STATUS_SUCCESS;
        }
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    dim3 grid(static_cast<uint64_t>(num_heads), static_cast<uint64_t>(num_seqs), 1);
    if (use_split) {
        dump_dispatch(use_cta ? "splitkv_cta" : "splitkv_warp");
        // }
        if (!fixed_num_splits) {
            // Approximate seqlen_k by the per-seq KV capacity (paged KV upper bound).
            const size_t seqlen_k = max_num_blocks_per_seq * page_block_size;
            const int sm_count = getSmCount();
            num_splits = chooseNumSplitsHeuristic(num_heads, num_seqs, seqlen_k, sm_count);
            if (const char *dbg = std::getenv("INFINIOP_FLASH_DEBUG_SPLITS")) {
                if (std::strcmp(dbg, "1") == 0 || std::strcmp(dbg, "true") == 0) {
                    static size_t last_seqlen_k = 0;
                    if (last_seqlen_k != seqlen_k) {
                        last_seqlen_k = seqlen_k;
                        fprintf(stderr,
                                "[INFINIOP][paged_attention] splitkv auto: sm=%d heads=%zu seqs=%zu seqlen_k~%zu -> num_splits=%d\n",
                                sm_count, num_heads, num_seqs, seqlen_k, num_splits);
                    }
                }
            }
        }

        const size_t n = num_seqs * num_heads;
        const size_t acc_elems = static_cast<size_t>(kMaxSplits) * n * 128;
        const size_t m_elems = static_cast<size_t>(kMaxSplits) * n;
        const size_t l_elems = static_cast<size_t>(kMaxSplits) * n;
        const size_t needed_bytes = (acc_elems + m_elems + l_elems) * sizeof(float);
        if (workspace == nullptr || workspace_size < needed_bytes) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        float *ws = static_cast<float *>(workspace);
        float *partial_acc = ws;
        float *partial_m = partial_acc + acc_elems;
        float *partial_l = partial_m + m_elems;

        dim3 grid_split(static_cast<uint64_t>(num_heads), static_cast<uint64_t>(num_seqs), static_cast<uint64_t>(num_splits));
        dim3 block_split(use_cta ? static_cast<uint32_t>(cta_threads) : 32);

        if (dtype == INFINI_DTYPE_F16) {
            if (use_cta) {
                if (cta_threads == 32) {
                    if (cta_tile == 16) {
                        flashAttentionDecodeHd128SplitKvCta32Tile16<Tindex, half><<<grid_split, block_split, 0, stream>>>(
                            partial_acc, partial_m, partial_l,
                            static_cast<const half *>(q),
                            static_cast<const half *>(k_cache),
                            static_cast<const half *>(v_cache),
                            block_tables, cache_lens, alibi_slopes,
                            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                            q_stride, k_batch_stride, k_row_stride, k_head_stride,
                            v_batch_stride, v_row_stride, v_head_stride, num_splits);
                    } else {
                        flashAttentionDecodeHd128SplitKvCta32<Tindex, half><<<grid_split, block_split, 0, stream>>>(
                            partial_acc, partial_m, partial_l,
                            static_cast<const half *>(q),
                            static_cast<const half *>(k_cache),
                            static_cast<const half *>(v_cache),
                            block_tables, cache_lens, alibi_slopes,
                            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                            q_stride, k_batch_stride, k_row_stride, k_head_stride,
                            v_batch_stride, v_row_stride, v_head_stride, num_splits);
                    }
                } else {
                    if (cta_tile == 16) {
                        flashAttentionDecodeHd128SplitKvCtaTile16<Tindex, half><<<grid_split, block_split, 0, stream>>>(
                            partial_acc, partial_m, partial_l,
                            static_cast<const half *>(q),
                            static_cast<const half *>(k_cache),
                            static_cast<const half *>(v_cache),
                            block_tables, cache_lens, alibi_slopes,
                            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                            q_stride, k_batch_stride, k_row_stride, k_head_stride,
                            v_batch_stride, v_row_stride, v_head_stride, num_splits);
                    } else {
                        flashAttentionDecodeHd128SplitKvCta<Tindex, half><<<grid_split, block_split, 0, stream>>>(
                            partial_acc, partial_m, partial_l,
                            static_cast<const half *>(q),
                            static_cast<const half *>(k_cache),
                            static_cast<const half *>(v_cache),
                            block_tables, cache_lens, alibi_slopes,
                            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                            q_stride, k_batch_stride, k_row_stride, k_head_stride,
                            v_batch_stride, v_row_stride, v_head_stride, num_splits);
                    }
                }
            } else {
                flashAttentionDecodeHd128SplitKv<Tindex, half><<<grid_split, block_split, 0, stream>>>(
                    partial_acc, partial_m, partial_l,
                    static_cast<const half *>(q),
                    static_cast<const half *>(k_cache),
                    static_cast<const half *>(v_cache),
                    block_tables, cache_lens, alibi_slopes,
                    num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                    q_stride, k_batch_stride, k_row_stride, k_head_stride,
                    v_batch_stride, v_row_stride, v_head_stride, num_splits);
            }
            flashAttentionDecodeHd128SplitKvCombine<half><<<grid, 32, 0, stream>>>(
                static_cast<half *>(out), partial_acc, partial_m, partial_l, num_splits, o_stride);
            return INFINI_STATUS_SUCCESS;
        }
        if (dtype == INFINI_DTYPE_BF16) {
            if (use_cta) {
                if (cta_threads == 32) {
                    if (cta_tile == 16) {
                        flashAttentionDecodeHd128SplitKvCta32Tile16<Tindex, __mt_bfloat16><<<grid_split, block_split, 0, stream>>>(
                            partial_acc, partial_m, partial_l,
                            static_cast<const __mt_bfloat16 *>(q),
                            static_cast<const __mt_bfloat16 *>(k_cache),
                            static_cast<const __mt_bfloat16 *>(v_cache),
                            block_tables, cache_lens, alibi_slopes,
                            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                            q_stride, k_batch_stride, k_row_stride, k_head_stride,
                            v_batch_stride, v_row_stride, v_head_stride, num_splits);
                    } else {
                        flashAttentionDecodeHd128SplitKvCta32<Tindex, __mt_bfloat16><<<grid_split, block_split, 0, stream>>>(
                            partial_acc, partial_m, partial_l,
                            static_cast<const __mt_bfloat16 *>(q),
                            static_cast<const __mt_bfloat16 *>(k_cache),
                            static_cast<const __mt_bfloat16 *>(v_cache),
                            block_tables, cache_lens, alibi_slopes,
                            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                            q_stride, k_batch_stride, k_row_stride, k_head_stride,
                            v_batch_stride, v_row_stride, v_head_stride, num_splits);
                    }
                } else {
                    if (cta_tile == 16) {
                        flashAttentionDecodeHd128SplitKvCtaTile16<Tindex, __mt_bfloat16><<<grid_split, block_split, 0, stream>>>(
                            partial_acc, partial_m, partial_l,
                            static_cast<const __mt_bfloat16 *>(q),
                            static_cast<const __mt_bfloat16 *>(k_cache),
                            static_cast<const __mt_bfloat16 *>(v_cache),
                            block_tables, cache_lens, alibi_slopes,
                            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                            q_stride, k_batch_stride, k_row_stride, k_head_stride,
                            v_batch_stride, v_row_stride, v_head_stride, num_splits);
                    } else {
                        flashAttentionDecodeHd128SplitKvCta<Tindex, __mt_bfloat16><<<grid_split, block_split, 0, stream>>>(
                            partial_acc, partial_m, partial_l,
                            static_cast<const __mt_bfloat16 *>(q),
                            static_cast<const __mt_bfloat16 *>(k_cache),
                            static_cast<const __mt_bfloat16 *>(v_cache),
                            block_tables, cache_lens, alibi_slopes,
                            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                            q_stride, k_batch_stride, k_row_stride, k_head_stride,
                            v_batch_stride, v_row_stride, v_head_stride, num_splits);
                    }
                }
            } else {
                flashAttentionDecodeHd128SplitKv<Tindex, __mt_bfloat16><<<grid_split, block_split, 0, stream>>>(
                    partial_acc, partial_m, partial_l,
                    static_cast<const __mt_bfloat16 *>(q),
                    static_cast<const __mt_bfloat16 *>(k_cache),
                    static_cast<const __mt_bfloat16 *>(v_cache),
                    block_tables, cache_lens, alibi_slopes,
                    num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                    q_stride, k_batch_stride, k_row_stride, k_head_stride,
                    v_batch_stride, v_row_stride, v_head_stride, num_splits);
            }
            flashAttentionDecodeHd128SplitKvCombine<__mt_bfloat16><<<grid, 32, 0, stream>>>(
                static_cast<__mt_bfloat16 *>(out), partial_acc, partial_m, partial_l, num_splits, o_stride);
            return INFINI_STATUS_SUCCESS;
        }
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    dump_dispatch(use_cta ? "cta_nosplit" : "warp_nosplit");

    if (dtype == INFINI_DTYPE_F16) {
        if (use_cta) {
            if (cta_tile == 16) {
                if (cta_threads == 32) {
                    flashAttentionDecodeHd128Cta32Tile16<Tindex, half><<<grid, block, 0, stream>>>(
                        static_cast<half *>(out),
                        static_cast<const half *>(q),
                        static_cast<const half *>(k_cache),
                        static_cast<const half *>(v_cache),
                        block_tables, cache_lens, alibi_slopes,
                        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                        q_stride, k_batch_stride, k_row_stride, k_head_stride,
                        v_batch_stride, v_row_stride, v_head_stride, o_stride);
                } else {
                    flashAttentionDecodeHd128CtaTile16<Tindex, half><<<grid, block, 0, stream>>>(
                        static_cast<half *>(out),
                        static_cast<const half *>(q),
                        static_cast<const half *>(k_cache),
                        static_cast<const half *>(v_cache),
                        block_tables, cache_lens, alibi_slopes,
                        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                        q_stride, k_batch_stride, k_row_stride, k_head_stride,
                        v_batch_stride, v_row_stride, v_head_stride, o_stride);
                }
            } else {
                if (cta_threads == 32) {
                    flashAttentionDecodeHd128Cta32<Tindex, half><<<grid, block, 0, stream>>>(
                        static_cast<half *>(out),
                        static_cast<const half *>(q),
                        static_cast<const half *>(k_cache),
                        static_cast<const half *>(v_cache),
                        block_tables, cache_lens, alibi_slopes,
                        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                        q_stride, k_batch_stride, k_row_stride, k_head_stride,
                        v_batch_stride, v_row_stride, v_head_stride, o_stride);
                } else {
                    flashAttentionDecodeHd128Cta<Tindex, half><<<grid, block, 0, stream>>>(
                        static_cast<half *>(out),
                        static_cast<const half *>(q),
                        static_cast<const half *>(k_cache),
                        static_cast<const half *>(v_cache),
                        block_tables, cache_lens, alibi_slopes,
                        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                        q_stride, k_batch_stride, k_row_stride, k_head_stride,
                        v_batch_stride, v_row_stride, v_head_stride, o_stride);
                }
            }
        } else {
            flashAttentionDecodeHd128Warp<Tindex, half><<<grid, block, 0, stream>>>(
                static_cast<half *>(out),
                static_cast<const half *>(q),
                static_cast<const half *>(k_cache),
                static_cast<const half *>(v_cache),
                block_tables, cache_lens, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                q_stride, k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride, o_stride);
        }
        return INFINI_STATUS_SUCCESS;
    }
    if (dtype == INFINI_DTYPE_BF16) {
        if (use_cta) {
            if (cta_tile == 16) {
                if (cta_threads == 32) {
                    flashAttentionDecodeHd128Cta32Tile16<Tindex, __mt_bfloat16><<<grid, block, 0, stream>>>(
                        static_cast<__mt_bfloat16 *>(out),
                        static_cast<const __mt_bfloat16 *>(q),
                        static_cast<const __mt_bfloat16 *>(k_cache),
                        static_cast<const __mt_bfloat16 *>(v_cache),
                        block_tables, cache_lens, alibi_slopes,
                        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                        q_stride, k_batch_stride, k_row_stride, k_head_stride,
                        v_batch_stride, v_row_stride, v_head_stride, o_stride);
                } else {
                    flashAttentionDecodeHd128CtaTile16<Tindex, __mt_bfloat16><<<grid, block, 0, stream>>>(
                        static_cast<__mt_bfloat16 *>(out),
                        static_cast<const __mt_bfloat16 *>(q),
                        static_cast<const __mt_bfloat16 *>(k_cache),
                        static_cast<const __mt_bfloat16 *>(v_cache),
                        block_tables, cache_lens, alibi_slopes,
                        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                        q_stride, k_batch_stride, k_row_stride, k_head_stride,
                        v_batch_stride, v_row_stride, v_head_stride, o_stride);
                }
            } else {
                if (cta_threads == 32) {
                    flashAttentionDecodeHd128Cta32<Tindex, __mt_bfloat16><<<grid, block, 0, stream>>>(
                        static_cast<__mt_bfloat16 *>(out),
                        static_cast<const __mt_bfloat16 *>(q),
                        static_cast<const __mt_bfloat16 *>(k_cache),
                        static_cast<const __mt_bfloat16 *>(v_cache),
                        block_tables, cache_lens, alibi_slopes,
                        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                        q_stride, k_batch_stride, k_row_stride, k_head_stride,
                        v_batch_stride, v_row_stride, v_head_stride, o_stride);
                } else {
                    flashAttentionDecodeHd128Cta<Tindex, __mt_bfloat16><<<grid, block, 0, stream>>>(
                        static_cast<__mt_bfloat16 *>(out),
                        static_cast<const __mt_bfloat16 *>(q),
                        static_cast<const __mt_bfloat16 *>(k_cache),
                        static_cast<const __mt_bfloat16 *>(v_cache),
                        block_tables, cache_lens, alibi_slopes,
                        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                        q_stride, k_batch_stride, k_row_stride, k_head_stride,
                        v_batch_stride, v_row_stride, v_head_stride, o_stride);
                }
            }
        } else {
            flashAttentionDecodeHd128Warp<Tindex, __mt_bfloat16><<<grid, block, 0, stream>>>(
                static_cast<__mt_bfloat16 *>(out),
                static_cast<const __mt_bfloat16 *>(q),
                static_cast<const __mt_bfloat16 *>(k_cache),
                static_cast<const __mt_bfloat16 *>(v_cache),
                block_tables, cache_lens, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                q_stride, k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride, o_stride);
        }
        return INFINI_STATUS_SUCCESS;
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

infiniStatus_t launch_decode_hd128_i64(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    infiniDtype_t dtype,
    const int64_t *block_tables,
    const int64_t *cache_lens,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    musaStream_t stream) {
    return launch_decode_hd128_impl<int64_t>(
        workspace, workspace_size,
        out, q, k_cache, v_cache, dtype, block_tables, cache_lens, alibi_slopes, num_heads, num_seqs,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride, k_batch_stride, k_row_stride,
        k_head_stride, v_batch_stride, v_row_stride, v_head_stride, o_stride, stream);
}

infiniStatus_t launch_decode_hd128_i32(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    infiniDtype_t dtype,
    const int32_t *block_tables,
    const int32_t *cache_lens,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    musaStream_t stream) {
    return launch_decode_hd128_impl<int32_t>(
        workspace, workspace_size,
        out, q, k_cache, v_cache, dtype, block_tables, cache_lens, alibi_slopes, num_heads, num_seqs,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride, k_batch_stride, k_row_stride,
        k_head_stride, v_batch_stride, v_row_stride, v_head_stride, o_stride, stream);
}

infiniStatus_t launch_decode_hd128_u32(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    infiniDtype_t dtype,
    const uint32_t *block_tables,
    const uint32_t *cache_lens,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    musaStream_t stream) {
    return launch_decode_hd128_impl<uint32_t>(
        workspace, workspace_size,
        out, q, k_cache, v_cache, dtype, block_tables, cache_lens, alibi_slopes, num_heads, num_seqs,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride, k_batch_stride, k_row_stride,
        k_head_stride, v_batch_stride, v_row_stride, v_head_stride, o_stride, stream);
}

} // namespace op::paged_attention::moore
