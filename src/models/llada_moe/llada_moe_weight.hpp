#ifndef LLADAMOE_WEIGHT_HPP
#define LLADAMOE_WEIGHT_HPP


#include <cmath>
inline std::shared_ptr<Tensor> getInEmbd() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getOutNorm(){
    return nullptr;
}

inline std::shared_ptr<Tensor> getOutEmbd(){
    return nullptr;
}

inline std::shared_ptr<Tensor> getAttnNorm() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getAttnQKV(){
    return nullptr;
}

inline std::shared_ptr<Tensor> getAttnQKVBias(){
    return nullptr;
}

inline std::shared_ptr<Tensor> getAttnQNorm() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getAttnKNorm() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getAttnO() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getFFNNorm() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getFFNGateUp() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getFFNDown() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getSinTable() {
    return nullptr;
}

inline std::shared_ptr<Tensor> getCosTable() {
    return nullptr;
}

#endif
