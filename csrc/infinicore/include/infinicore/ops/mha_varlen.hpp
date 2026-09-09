#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

class MultiheadAttentionVarlen : public graph::DispatchableGraphOperator {
public:
    using schema = void (*)(Tensor,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            std::optional<Tensor>,
                            int,
                            int,
                            std::optional<Tensor>,
                            float);
    using plan_schema = void *(*)(Tensor,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  std::optional<Tensor>,
                                  int,
                                  int,
                                  std::optional<Tensor>,
                                  float);

    static common::OpDispatcher<plan_schema> &plan_dispatcher();
    static common::OpDispatcher<run_schema> &run_dispatcher();
    static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher();

    MultiheadAttentionVarlen(Tensor out,
                             const Tensor &q,
                             const Tensor &k,
                             const Tensor &v,
                             const Tensor &cum_seqlens_q,
                             const Tensor &cum_seqlens_kv,
                             std::optional<Tensor> block_table,
                             int max_seqlen_q,
                             int max_seqlen_k,
                             std::optional<Tensor> alibi_slopes,
                             float scale);

    static void execute(Tensor out,
                        const Tensor &q,
                        const Tensor &k,
                        const Tensor &v,
                        const Tensor &cum_seqlens_q,
                        const Tensor &cum_seqlens_kv,
                        std::optional<Tensor> block_table,
                        int max_seqlen_q,
                        int max_seqlen_k,
                        std::optional<Tensor> alibi_slopes,
                        float scale);

    bool is_device_graph_capture_safe() const override {
        return device_graph_capture_safe_;
    }

private:
    bool device_graph_capture_safe_;
};

Tensor mha_varlen(const Tensor &q,
                  const Tensor &k,
                  const Tensor &v,
                  const Tensor &cum_seqlens_q,
                  const Tensor &cum_seqlens_k,
                  std::optional<Tensor> block_table,
                  int max_seqlen_q,
                  int max_seqlen_k,
                  std::optional<Tensor> alibi_slopes,
                  float scale);

void mha_varlen_(Tensor out,
                 const Tensor &q,
                 const Tensor &k,
                 const Tensor &v,
                 const Tensor &cum_seqlens_q,
                 const Tensor &cum_seqlens_k,
                 std::optional<Tensor> block_table,
                 int max_seqlen_q,
                 int max_seqlen_k,
                 std::optional<Tensor> alibi_slopes,
                 float scale);

} // namespace infinicore::op
