#pragma once

#include "cudagraph_dispatcher.hpp"

namespace infinilm::engine::dispatch_hist {

/// Record dispatcher mode (FULL / PIECEWISE / NONE + NONE reason). Rank-0 only.
void record_mode(CudaGraphRuntimeMode mode, const char *none_reason);

/// One step-level count for a PIECEWISE run_prefill entry after path is known.
/// Priority: pad_up_eager > mid_chunk_eager > missing_graph > exact_cg.
/// ``pad_up`` here means *eager* pad-up (kill-switch / miss); successful pad-up
/// CG counts as exact_cg. ``mid_chunk_eager`` is true only when the mid path
/// still ran eager; exact-width / pad-up CG mid replay counts as exact_cg.
void record_piecewise_eager(bool pad_up, bool mid_chunk_eager, bool missing_graph);

/// Legacy noisy dump gated by INFINI_RANK_WORKER_PROFILE / hang_trace.
void log_profile_hist(const char *tag);

/// Periodic / teardown one-line dump gated by INFINI_CG_EAGER_HIST=1.
/// Call after each mode record (period) and on process teardown / RankWorker::close.
void maybe_dump_eager(const char *tag);
void dump_eager_final(const char *tag = "teardown");

/// Ensure atexit final dump is registered (idempotent).
void ensure_atexit_dump();

} // namespace infinilm::engine::dispatch_hist
