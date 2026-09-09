#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

class PagedAttentionPrefill {
public:
    /**
     * @brief PagedAttentionPrefill operator signature
     * * Argument order:
     * 1. out: Output tensor (Packed format)
     * 2. q: Current Query tensor (Packed format)
     * 3. k_cache: Physical Key cache (Paged format)
     * 4. v_cache: Physical Value cache (Paged format)
     * 5. block_tables: Mapping table from logical blocks to physical blocks
     * 6. total_kv_lens:  lengths of Complete Key/Value for each request
     * 7. cu_seqlens_q: Cumulative sequence lengths of Query (prefix sum for variable-length batch)
     * 8. alibi_slopes: ALiBi bias slopes (optional)
     * 9. scale: Scaling factor (typically 1/sqrt(head_size))
     */
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>, float);

    static void execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                        Tensor block_tables, Tensor total_kv_lens, Tensor cum_seqlens_q,
                        std::optional<Tensor> alibi_slopes, float scale);

    static common::OpDispatcher<schema> &dispatcher();
};

Tensor paged_attention_prefill(Tensor q,
                               Tensor k_cache,
                               Tensor v_cache,
                               Tensor block_tables,
                               Tensor total_kv_lens,
                               Tensor cum_seqlens_q,
                               std::optional<Tensor> alibi_slopes,
                               float scale);

void paged_attention_prefill_(Tensor out,
                              Tensor q,
                              Tensor k_cache,
                              Tensor v_cache,
                              Tensor block_tables,
                              Tensor total_kv_lens,
                              Tensor cum_seqlens_q,
                              std::optional<Tensor> alibi_slopes,
                              float scale);

} // namespace infinicore::op
