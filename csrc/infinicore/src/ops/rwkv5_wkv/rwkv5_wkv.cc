#include "infinicore/ops/rwkv5_wkv.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Rwkv5Wkv);

Rwkv5Wkv::Rwkv5Wkv(Tensor out,
                   const Tensor &receptance,
                   const Tensor &key,
                   const Tensor &value,
                   const Tensor &time_decay,
                   const Tensor &time_faaaa,
                   Tensor state) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, receptance, key, value, time_decay, time_faaaa, state);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(), out, receptance, key, value, time_decay, time_faaaa, state);
}

void Rwkv5Wkv::execute(Tensor out,
                       const Tensor &receptance,
                       const Tensor &key,
                       const Tensor &value,
                       const Tensor &time_decay,
                       const Tensor &time_faaaa,
                       Tensor state) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Rwkv5Wkv, out, receptance, key, value, time_decay, time_faaaa, state);
}

Tensor rwkv5_wkv(const Tensor &receptance,
                 const Tensor &key,
                 const Tensor &value,
                 const Tensor &time_decay,
                 const Tensor &time_faaaa,
                 Tensor state) {
    auto output = Tensor::empty(receptance->shape(), receptance->dtype(), receptance->device());
    rwkv5_wkv_(output, receptance, key, value, time_decay, time_faaaa, state);
    return output;
}

void rwkv5_wkv_(Tensor out,
                const Tensor &receptance,
                const Tensor &key,
                const Tensor &value,
                const Tensor &time_decay,
                const Tensor &time_faaaa,
                Tensor state) {
    Rwkv5Wkv::execute(out, receptance, key, value, time_decay, time_faaaa, state);
}

} // namespace infinicore::op
