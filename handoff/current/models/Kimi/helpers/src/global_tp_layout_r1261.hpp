#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace infinilm::parallel {
// Value-only layout. Device IDs stay local; TP ranks span nodes inside a PP
// stage. This has no Runtime, communicator, model or request ownership.
class CrossNodeTpLayoutR1261 {
    int local_, nodes_, node_, pp_, stage_, tp_, world_;
    static int positive(size_t n) {
        if (!n || n > size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("1261 rank count exceeds InfiniCCL limits");
        return int(n);
    }
    static int multiply(int a, int b) {
        if (a > std::numeric_limits<int>::max() / b)
            throw std::invalid_argument("1261 combined rank count overflows");
        return a*b;
    }
public:
    CrossNodeTpLayoutR1261(size_t local, size_t nodes, int node, int pp, int stage)
        : local_(positive(local)), nodes_(positive(nodes)), node_(node),
          pp_(positive(pp > 0 ? size_t(pp) : 0)), stage_(stage),
          tp_(multiply(local_,nodes_)), world_(multiply(tp_,pp_)) {
        if (node < 0 || node >= nodes_ || stage < 0 || stage >= pp_)
            throw std::invalid_argument("1261 invalid TP node or PP stage");
    }
    int local_size() const noexcept { return local_; }
    int node_size() const noexcept { return nodes_; }
    int node_rank() const noexcept { return node_; }
    int tp_size() const noexcept { return tp_; }
    int world_size() const noexcept { return world_; }
    int participant_count() const noexcept { return nodes_*pp_; }
    int participant_rank() const noexcept { return stage_*nodes_+node_; }
    int tp_rank(int local) const {
        if(local<0 || local>=local_)throw std::out_of_range("1261 invalid local rank");
        return node_*local_+local;
    }
    int world_rank(int local) const { return stage_*tp_+tp_rank(local); }
};
}
