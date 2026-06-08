#pragma once

namespace infinilm::layers {

class FusedWeightsProcessor {
public:
    virtual ~FusedWeightsProcessor() = default;
    virtual void process_fused_weights_after_loading() = 0;
    virtual void reset_fused_runtime_state() const {}
};

} // namespace infinilm::layers
