#include "models/kimi_k3/kimi_k3_kda_trace.hpp"
#include "execution/submit.hpp"
#include "base_quantization.hpp"

#include "../../engine/distributed/communication_group.hpp"

namespace infinilm::quantization {

infinilm::runtime::Tensor BaseQuantization::forward_allreduce(
    infinilm::runtime::Runtime &runtime,
    const ParamsMap &params,
    const infinilm::runtime::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    auto output = forward(runtime, params, input, false, alpha);
    infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"row_output_local");
    infinilm::engine::distributed::allreduce_(
        runtime, output, output, infinicclSum, communicator);
    infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"row_output_reduced");
    if (has_bias) {
        {
            const auto ops_other = params.at("bias");
            infinilm::execution::submit<infini::ops::Add>(
                runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), output, ops_other, output);
            INFINILM_TRACE_OP(Add, output);
        }
    }
    return output;
}

} // namespace infinilm::quantization
