#pragma once

#include "../../models/infinilm_model.hpp"
#include "../rank_barrier.hpp"

namespace infinilm::engine {

class GraphCompiler {
public:
    using Compiled = std::tuple<
        std::shared_ptr<infinicore::graph::Graph>,
        std::shared_ptr<InfinilmModel::Output>>;

    explicit GraphCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier) : model_(model), barrier_(barrier) {}
    virtual ~GraphCompiler() = default;

    virtual void compile() = 0;
    virtual Compiled get_compiled(const InfinilmModel::Input &input) = 0;

    // Optional companion graph that replays captured greedy-sampling kernels
    // (per-request argmax) for a decode batch size. Only valid right after a
    // successful get_compiled()+run() for the same batch size, since it reads
    // that graph's logits blob. Default: no captured sampling.
    virtual std::pair<std::shared_ptr<infinicore::graph::Graph>, infinicore::Tensor>
    get_sampling_compiled(size_t batch_size) {
        (void)batch_size;
        return {nullptr, {}};
    }

protected:
    std::shared_ptr<InfinilmModel> model_;
    RankBarrier *barrier_;
};

} // namespace infinilm::engine
