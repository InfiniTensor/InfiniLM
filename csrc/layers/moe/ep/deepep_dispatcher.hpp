#pragma once

#include "../dispatcher/base_dispatcher.hpp"
#include "ep_config.hpp"

namespace infinilm::layers::moe {

class DeepEPDispatcher final : public BaseDispatcher {
public:
    explicit DeepEPDispatcher(EPConfig config);

    DispatchOutput dispatch(const infinicore::Tensor &hidden_states,
                            const TopKOutput &topk_output,
                            MoeWorkspace &workspace) const override;

    infinicore::Tensor combine(const CombineInput &combine_input,
                               MoeWorkspace &workspace) const override;

    void dispatch_a(const infinicore::Tensor &hidden_states,
                    const TopKOutput &topk_output) const;
    DispatchOutput dispatch_b() const;

    void combine_a(const CombineInput &combine_input) const;
    infinicore::Tensor combine_b() const;

private:
    EPConfig config_;
};

} // namespace infinilm::layers::moe
