#pragma once

#include "../../models/infinilm_model.hpp"

namespace infinilm::engine {

class GraphCompiler {
public:
    using Compiled = std::tuple<
        std::shared_ptr<infinicore::graph::Graph>,
        std::shared_ptr<InfinilmModel::Output>>;

    explicit GraphCompiler(const std::shared_ptr<InfinilmModel> &model) : model_(model) {}
    virtual ~GraphCompiler() = default;

    virtual void compile() = 0;
    virtual Compiled get_compiled(const InfinilmModel::Input &input) = 0;

protected:
    std::shared_ptr<InfinilmModel> model_;
};

} // namespace infinilm::engine
