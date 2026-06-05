#pragma once

namespace infinilm::layers {

class FusedWeightsProcessor {
public:
    virtual ~FusedWeightsProcessor() = default;
    virtual void process_fused_weights_after_loading() = 0;
};

} // namespace infinilm::layers
