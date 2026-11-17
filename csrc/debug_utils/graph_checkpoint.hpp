#pragma once
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace infinilm::debug {
inline bool graph_checkpoints_enabled() {
    static const bool enabled = std::getenv("INFINI_DEBUG_GRAPH_DIR") != nullptr;
    return enabled;
}
inline thread_local size_t checkpoint_step = 0;

inline void dump_checkpoint(const std::string &name, const infinicore::Tensor &tensor) {
    const std::string root = std::getenv("INFINI_DEBUG_GRAPH_DIR");
    std::filesystem::create_directories(root);
    auto cpu = tensor->to(infinicore::Device{infinicore::Device::Type::kCpu});
    const auto type = tensor->dtype();
    const char *dtype = type == infinicore::DataType::kBFloat16 ? "bf16"
        : type == infinicore::DataType::kFloat32 ? "fp32"
        : type == infinicore::DataType::kFloat16 ? "fp16"
        : type == infinicore::DataType::kInt32 ? "int32" : "int64";
    std::ofstream data(root + "/" + name + ".bin", std::ios::binary);
    data.write(reinterpret_cast<const char *>(cpu->data()), cpu->nbytes());
    std::ofstream meta(root + "/" + name + ".json");
    meta << "{\"dtype\":\"" << dtype << "\",\"device_ptr\":\"" << tensor->data() << "\",\"shape\":[";
    for (size_t i=0; i<tensor->shape().size(); ++i) { if(i) meta << ','; meta << tensor->shape()[i]; }
    meta << "]}";
}

class GraphCheckpoint : public infinicore::graph::GraphOperator {
    std::string name_;
    infinicore::graph::GraphTensor tensor_;
    std::optional<infinicore::graph::GraphTensor> index_;
public:
    GraphCheckpoint(std::string name, const infinicore::Tensor &tensor,
                    const infinicore::Tensor &index)
        : name_(std::move(name)), tensor_(tensor) {
        if (index) index_.emplace(index);
    }
    bool is_device_graph_capture_safe() const override { return false; }
    void run() const override {
        if (infinicore::context::isGraphRecording()
            || infinicore::context::getInferencePhase() != infinicore::context::InferencePhase::Decode) return;
        if (name_ == "00_embedding") ++checkpoint_step;
        if (checkpoint_step != 1) return;
        if (index_) {
            auto idx = index_.value()->to(infinicore::Device{infinicore::Device::Type::kCpu});
            const int64_t slot = idx->dtype() == infinicore::DataType::kInt32
                ? *reinterpret_cast<const int32_t *>(idx->data()) : *reinterpret_cast<const int64_t *>(idx->data());
            dump_checkpoint(name_ + "_index", index_.value());
            if (slot < 0 || size_t(slot) >= tensor_->size(0)) throw std::runtime_error("debug state slot out of bounds");
            dump_checkpoint(name_, tensor_->narrow({{0,size_t(slot),1}}));
            dump_checkpoint(name_ + "_zero", tensor_->narrow({{0,0,1}}));
        } else dump_checkpoint(name_, tensor_);
    }
};

inline void graph_checkpoint(const std::string &name, const infinicore::Tensor &tensor,
                             const infinicore::Tensor &index = {}) {
    if (!graph_checkpoints_enabled() || !tensor) return;
    auto op = std::make_shared<GraphCheckpoint>(name,tensor,index);
    if (infinicore::context::isGraphRecording()) infinicore::context::addGraphOperator(op);
    else op->run();
}
} // namespace infinilm::debug
