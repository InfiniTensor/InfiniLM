#ifndef _QWEN_WEIGHT_HPP_
#define _QWEN_WEIGHT_HPP_

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <vector>

namespace Qwen {
//
// CPU的权重指针
//
struct MLPCStruct {
    void *_gate_up_proj_weight{nullptr}; // ("_gate_up_proj_weight", c_void_p),
    void *_down_proj_weight{nullptr};    // ("_down_proj_weight", c_void_p),

    void print_info() const {
        printf("\n");
        printf("\t\t\tMLPCStruct:\n");
        printf("\t\t\t\tgate_up_proj_weight : %p\n", _gate_up_proj_weight);
        printf("\t\t\t\tdown_proj_weight : %p\n", _down_proj_weight);
    }
};

struct SparseMLPCStruct {
    size_t _shared_expert_num{0};              // ("_shared_expert_num", c_size_t)
    size_t _num_experts{0};                    // ("_num_experts", c_size_t)
    void *_shared_expert_gate_weight{nullptr}; // ("_shared_expert_gate_weight", c_void_p)
    void *_gate_weight{nullptr};               // ("_gate_weight", c_void_p)
    MLPCStruct _shared_expert;                 // ("_shared_expert", MLPCStruct)
    MLPCStruct *_experts{nullptr};             // ("_experts", POINTER(MLPCStruct))

    void print_info() const {
        printf("\n");
        printf("\t\tSparseMLPCStruct:\n");
        printf("\t\t\tshared_expert_gate_weight : %p\n", _shared_expert_gate_weight);
        printf("\t\t\tgate_weight : %p\n", _gate_weight);

        printf("\t\t\t shared_expert : \n");
        _shared_expert.print_info();

        printf("\t\t\t experts : \n");
        if (_experts) {
            _experts[0].print_info();
        }
    }
};

struct AttentionCStruct {
    void *_qkv_proj_weight{nullptr}; // ("_qkv_proj_weight", c_void_p)
    void *_qkv_proj_bias{nullptr};   // ("_qkv_proj_bias", c_void_p)
    void *_qk_norm_weight{nullptr};  // ("_qk_norm_weight", c_void_p)
    void *_o_proj_weight{nullptr};   // ("_o_proj_weight", c_void_p)

    void print_info() const {
        printf("\t\tAttentionCStruct:\n");
        printf("\t\t\tqkv_proj_weight : %p\n", _qkv_proj_weight);
        printf("\t\t\tqkv_proj_bias : %p\n", _qkv_proj_bias);
        printf("\t\t\tqk_norm_weight : %p\n", _qk_norm_weight);
        printf("\t\t\to_proj_weight : %p\n", _o_proj_weight);
    }
};

template <typename AttentionCStruct, typename FFNCStruct>
struct DecoderLayerCStruct {
    int _ilayer{0};
    void *_post_attention_layernorm_weight{nullptr}; //  ("_post_attention_layernorm_weight", c_void_p),
    void *_input_layernorm_weight{nullptr};          //  ("_input_layernorm_weight", c_void_p),
    AttentionCStruct _self_attn;
    FFNCStruct _mlp;

    void print_info() const {
        printf("\tDecoderLayerCStruct:\n");
        printf("\t\tilayer : %d\n", _ilayer);
        printf("\t\tpost_attention_layernorm_weight : %p\n", _post_attention_layernorm_weight);
        printf("\t\tinput_layernorm_weight : %p\n", _input_layernorm_weight);
        _self_attn.print_info();
        _mlp.print_info();
    }
};

}; // namespace Qwen

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Qwen {
// getAttnNorm getFFNNorm getOutNorm getAttnQKNorm
/**
 * @brief Create a normalization weight tensor
 * @param d Dimension size
 * @param dt_norm Data type for normalization weights
 * @param norm_weight_ptr is cpu pointer, it will be copied to gpu memory
 * @return Shared pointer to the created Tensor
 * @throws std::invalid_argument if norm_weight_ptr is nullptr
 */
inline std::shared_ptr<Tensor> getNorm(size_t d, infiniDtype_t dt_norm, void *norm_weight_ptr) {
    if (norm_weight_ptr == nullptr) {
        throw std::invalid_argument("getNorm: norm_weight_ptr cannot be nullptr");
    }
    auto shape = std::vector<size_t>({d});
    return Tensor::weight(static_cast<char *>(norm_weight_ptr), dt_norm, shape);
}

/**
 * @brief Create a sine table for RoPE (Rotary Position Embedding)
 * @param dh Head dimension
 * @param theta Base frequency parameter
 * @param dctx Maximum context length
 * @param dt_logits Data type for the table
 * @return Shared pointer to the created Tensor
 * @throws std::runtime_error if memory allocation fails
 */
inline std::shared_ptr<Tensor> getSinTable(size_t dh, float theta, size_t dctx, infiniDtype_t dt_logits) {

    if (theta <= 0.0f) {
        throw std::invalid_argument("getSinTable: theta must be positive");
    }

    auto half_dh = dh / 2;
    auto unit = dsize(dt_logits);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));
            if (dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                throw std::invalid_argument("getSinTable: unsupported data type");
            }
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, dt_logits, shape);
    std::free(table);
    return tensor;
}

/**
 * @brief Create a cosine table for RoPE (Rotary Position Embedding)
 * @param dh Head dimension
 * @param theta Base frequency parameter
 * @param dctx Maximum context length
 * @param dt_logits Data type for the table
 * @return Shared pointer to the created Tensor
 * @throws std::runtime_error if memory allocation fails
 */
inline std::shared_ptr<Tensor> getCosTable(size_t dh, float theta, size_t dctx, infiniDtype_t dt_logits) {
    auto half_dh = dh / 2;
    auto unit = dsize(dt_logits);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));
            if (dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                throw std::invalid_argument("getCosTable: unsupported data type");
            }
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, dt_logits, shape);
    std::free(table);
    return tensor;
}

/**
 * @brief Create input embedding tensor
 * @param d Hidden dimension
 * @param dvoc Vocabulary size
 * @param dt_logits Data type
 * @param embed_tokens_weight_ptr is cpu pointer, it will be copied to gpu memory
 * @return Shared pointer to the created Tensor
 * @throws std::invalid_argument if embed_tokens_weight_ptr is nullptr
 */
inline std::shared_ptr<Tensor> getInEmbd(size_t d, size_t dvoc, infiniDtype_t dt_logits, void *embed_tokens_weight_ptr) {
    if (embed_tokens_weight_ptr == nullptr) {
        throw std::invalid_argument("getInEmbd: embed_tokens_weight_ptr cannot be nullptr");
    }
    auto shape = std::vector<size_t>({dvoc, d});
    return Tensor::weight(static_cast<char *>(embed_tokens_weight_ptr), dt_logits, shape);
}

/**
 * @brief Create output embedding (LM head) tensor
 * @param d Hidden dimension
 * @param dvoc Vocabulary size
 * @param dt_logits Data type
 * @param transpose_linear_weights Whether to transpose weights
 * @param lm_head_weight_ptr is cpu pointer, it will be copied to gpu memory
 * @return Shared pointer to the created Tensor
 * @throws std::invalid_argument if lm_head_weight_ptr is nullptr
 */
inline std::shared_ptr<Tensor> getOutEmbd(size_t d, size_t dvoc, infiniDtype_t dt_logits, int transpose_linear_weights, void *lm_head_weight_ptr) {
    if (lm_head_weight_ptr == nullptr) {
        throw std::invalid_argument("getOutEmbd: lm_head_weight_ptr cannot be nullptr");
    }
    if (transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({dvoc, d});
        return Tensor::weight(static_cast<char *>(lm_head_weight_ptr), dt_logits, shape)->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, dvoc});
        return Tensor::weight(static_cast<char *>(lm_head_weight_ptr), dt_logits, shape);
    }
}

}; // namespace Qwen

namespace Qwen {
class BaseMLPTensor {
public:
    std::shared_ptr<Tensor> w_ffn_gate_up;
    std::shared_ptr<Tensor> w_ffn_down;

public:
    BaseMLPTensor() = default;

    void Init(size_t di, size_t d, infiniDtype_t dt_mat, int transpose_linear_weights, size_t idev, size_t ndev, void *gate_up_proj_weight_ptr, void *down_proj_weight_ptr) {
        this->w_ffn_gate_up = this->getFFNGateUp(di, d, dt_mat, transpose_linear_weights, idev, ndev, gate_up_proj_weight_ptr);
        this->w_ffn_down = this->getFFNDown(di, d, dt_mat, transpose_linear_weights, idev, ndev, down_proj_weight_ptr);
    }

private:
    inline std::shared_ptr<Tensor> getFFNGateUp(size_t di, size_t d, infiniDtype_t dt_mat, int transpose_linear_weights, size_t idev, size_t ndev, void *gate_up_proj_weight_ptr) {

        size_t offset = idev * (2 * di / ndev) * d * dsize(dt_mat);
        if (transpose_linear_weights != 0) {
            auto shape = std::vector<size_t>({2 * di / ndev, d});
            return Tensor::weight((char *)(gate_up_proj_weight_ptr) + offset, dt_mat, shape)->permute({1, 0});
        } else {
            auto shape = std::vector<size_t>({d, 2 * di / ndev});
            return Tensor::weight((char *)(gate_up_proj_weight_ptr) + offset, dt_mat, shape);
        }
    }

    inline std::shared_ptr<Tensor> getFFNDown(size_t di, size_t d, infiniDtype_t dt_mat, int transpose_linear_weights, size_t idev, size_t ndev, void *down_proj_weight_ptr) {
        size_t offset = idev * d * (di / ndev) * dsize(dt_mat);
        if (transpose_linear_weights != 0) {
            auto shape = std::vector<size_t>({d, di / ndev});
            return Tensor::weight((char *)(down_proj_weight_ptr) + offset, dt_mat, shape)->permute({1, 0});
        } else {
            auto shape = std::vector<size_t>({di / ndev, d});
            return Tensor::weight((char *)(down_proj_weight_ptr) + offset, dt_mat, shape);
        }
    }
};

class BaseAttentionTensor {
public:
    std::shared_ptr<Tensor> w_attn_qkv;
    std::shared_ptr<Tensor> b_attn_qkv;
    std::shared_ptr<Tensor> w_attn_qk_norm;
    std::shared_ptr<Tensor> w_attn_out;

public:
    BaseAttentionTensor() = default;
    void Init(size_t nkvh,
              size_t nh,
              size_t dh,
              size_t d,
              infiniDtype_t dt_mat,
              infiniDtype_t dt_norm,
              int transpose_linear_weights,
              size_t idev, size_t ndev,
              void *qkv_proj_weight_ptr,
              void *qkv_proj_bias_ptr,
              void *qk_norm_weight_ptr,
              void *o_proj_weight_ptr) {

        this->w_attn_qkv = this->getAttnQKV(nkvh, nh, dh, d, dt_mat, dt_norm, transpose_linear_weights, idev, ndev, qkv_proj_weight_ptr);

        if (qkv_proj_bias_ptr != nullptr) {
            this->b_attn_qkv = this->getAttnQKVBias(nkvh, nh, dh, d, dt_mat, dt_norm, transpose_linear_weights, idev, ndev, qkv_proj_bias_ptr);
        }

        if (qk_norm_weight_ptr != nullptr) {
            this->w_attn_qk_norm = getNorm(dh * 2, dt_norm, qk_norm_weight_ptr);
        }

        this->w_attn_out = this->getAttnO(nkvh, nh, dh, d, dt_mat, dt_norm, transpose_linear_weights, idev, ndev, o_proj_weight_ptr);
    }

private:
    inline std::shared_ptr<Tensor> getAttnQKV(size_t nkvh,
                                              size_t nh,
                                              size_t dh,
                                              size_t d,
                                              infiniDtype_t dt_mat,
                                              infiniDtype_t dt_norm,
                                              int transpose_linear_weights,
                                              size_t idev,
                                              size_t ndev, void *qkv_proj_weight_ptr) {

        size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(dt_mat);
        if (transpose_linear_weights != 0) {
            auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh, d});
            return Tensor::weight((char *)(qkv_proj_weight_ptr) + offset, dt_mat, shape)->permute({1, 0});
        } else {
            auto shape = std::vector<size_t>({d, (nh + 2 * nkvh) / ndev * dh});
            return Tensor::weight((char *)(qkv_proj_weight_ptr) + offset, dt_mat, shape);
        }
    }

    inline std::shared_ptr<Tensor> getAttnQKVBias(size_t nkvh,
                                                  size_t nh,
                                                  size_t dh,
                                                  size_t d,
                                                  infiniDtype_t dt_mat,
                                                  infiniDtype_t dt_norm,
                                                  int transpose_linear_weights,
                                                  size_t idev,
                                                  size_t ndev, void *qkv_proj_bias_ptr) {

        size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * dsize(dt_mat);
        auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh});

        return Tensor::weight((char *)(qkv_proj_bias_ptr) + offset, dt_mat, shape);
    }

    inline std::shared_ptr<Tensor> getAttnO(size_t nkvh,
                                            size_t nh,
                                            size_t dh,
                                            size_t d,
                                            infiniDtype_t dt_mat,
                                            infiniDtype_t dt_norm,
                                            int transpose_linear_weights,
                                            size_t idev,
                                            size_t ndev, void *o_proj_weight_ptr) {

        size_t offset = idev * d * (nh / ndev * dh) * dsize(dt_mat);
        if (transpose_linear_weights != 0) {
            auto shape = std::vector<size_t>({d, nh / ndev * dh});
            return Tensor::weight((char *)(o_proj_weight_ptr) + offset, dt_mat, shape)->permute({1, 0});
        } else {
            auto shape = std::vector<size_t>({nh / ndev * dh, d});
            return Tensor::weight((char *)(o_proj_weight_ptr) + offset, dt_mat, shape);
        }
    }
};

}; // namespace Qwen

//
// 存储 gpu 地址
//
namespace Qwen {

template <typename Meta, typename Weights>
class MLPTensor : public Qwen::BaseMLPTensor {
public:
    MLPTensor(Meta const *meta, Weights const *w, int ilayer, size_t idev, size_t ndev) {
        size_t di = meta->di;
        size_t d = meta->d;
        infiniDtype_t dt_mat = w->_dt_mat;
        int transpose_linear_weights = w->_transpose_linear_weights;
        void *gate_up_proj_weight_ptr = w->_layers[ilayer]._mlp._gate_up_proj_weight;
        void *down_proj_weight_ptr = w->_layers[ilayer]._mlp._down_proj_weight;
        this->Init(di, d, dt_mat, transpose_linear_weights, idev, ndev, gate_up_proj_weight_ptr, down_proj_weight_ptr);
    }

public:
    void print_info() const {
        printf("\t\t\t Qwen3::MLPTensor \n");
        printf("\t\t\t\t w_ffn_gate_up :: %p\t%s \n", w_ffn_gate_up.get(), w_ffn_gate_up->info().c_str());
        printf("\t\t\t\t w_ffn_down :: %p\t%s \n", w_ffn_down.get(), w_ffn_down->info().c_str());
    }
};

template <typename Meta, typename Weights>
class SharedMLPTensor : public Qwen::BaseMLPTensor {
public:
    SharedMLPTensor(Meta const *meta, Weights const *w, int ilayer, size_t idev, size_t ndev) {
        size_t di = meta->_shared_expert_intermediate_size;
        size_t d = meta->d;
        infiniDtype_t dt_mat = w->_dt_mat;
        int transpose_linear_weights = w->_transpose_linear_weights;
        void *gate_up_proj_weight_ptr = w->_layers[ilayer]._mlp._shared_expert._gate_up_proj_weight;
        void *down_proj_weight_ptr = w->_layers[ilayer]._mlp._shared_expert._down_proj_weight;
        this->Init(di, d, dt_mat, transpose_linear_weights, idev, ndev, gate_up_proj_weight_ptr, down_proj_weight_ptr);
    }

public:
    void print_info() const {
        printf("\t\t\t\t SharedMLPTensor \n");
        printf("\t\t\t\t\t w_ffn_gate_up :: %p\t%s \n", w_ffn_gate_up.get(), w_ffn_gate_up->info().c_str());
        printf("\t\t\t\t\t w_ffn_down :: %p\t%s \n", w_ffn_down.get(), w_ffn_down->info().c_str());
    }
};

template <typename Meta, typename Weights>
class RouterMLPTensor : public Qwen::BaseMLPTensor {
public:
    RouterMLPTensor(Meta const *meta, Weights const *w, int ilayer, int iexpert, size_t idev, size_t ndev) {
        size_t di = meta->_moe_intermediate_size;
        size_t d = meta->d;
        infiniDtype_t dt_mat = w->_dt_mat;
        int transpose_linear_weights = w->_transpose_linear_weights;
        void *gate_up_proj_weight_ptr = w->_layers[ilayer]._mlp._experts[iexpert]._gate_up_proj_weight;
        void *down_proj_weight_ptr = w->_layers[ilayer]._mlp._experts[iexpert]._down_proj_weight;
        this->Init(di, d, dt_mat, transpose_linear_weights, idev, ndev, gate_up_proj_weight_ptr, down_proj_weight_ptr);
    }

public:
    void print_info() const {
        printf("\t\t\t\t RouterMLPTensor \n");
        printf("\t\t\t\t\t w_ffn_gate_up :: %p\t%s \n", w_ffn_gate_up.get(), w_ffn_gate_up->info().c_str());
        printf("\t\t\t\t\t w_ffn_down :: %p\t%s \n", w_ffn_down.get(), w_ffn_down->info().c_str());
    }
};

template <typename SharedMLPTensor, typename RouterMLPTensor, typename Meta, typename Weights>
class SparseMLPTensor {
public:
    size_t _shared_expert_num;
    size_t _num_experts;
    std::shared_ptr<Tensor> _shared_expert_gate_weight;
    std::shared_ptr<Tensor> _gate_weight;
    std::shared_ptr<SharedMLPTensor> _shared_expert;
    std::vector<std::shared_ptr<RouterMLPTensor>> _experts;

public:
    SparseMLPTensor(Meta const *meta, Weights const *w, int ilayer, size_t idev, size_t ndev) {
        this->_shared_expert_num = 1;
        this->_num_experts = meta->_num_experts;

        if (w->_layers[ilayer]._mlp._shared_expert_gate_weight) {
            // gate
            void *shared_expert_gate = w->_layers[ilayer]._mlp._shared_expert_gate_weight;
            auto shape = std::vector<size_t>({meta->d, 1});
            this->_shared_expert_gate_weight = Tensor::weight((char *)(shared_expert_gate), w->_dt_mat, shape);

            // 权重
            this->_shared_expert = std::make_shared<SharedMLPTensor>(meta, w, ilayer, idev, ndev);
        }

        //
        void *experts_gate = w->_layers[ilayer]._mlp._gate_weight;
        auto shape = std::vector<size_t>({meta->d, meta->_num_experts});
        this->_gate_weight = Tensor::weight((char *)(experts_gate), w->_dt_mat, shape);

        // experts
        this->_experts.reserve(meta->_num_experts);
        for (size_t iexpert = 0; iexpert < meta->_num_experts; ++iexpert) {
            this->_experts.push_back(
                std::make_shared<RouterMLPTensor>(meta, w, ilayer, iexpert, idev, ndev));
        }
    }

public:
    void print_info() const {
        printf("\t\t\t SparseMLPTensor \n");
        printf("\t\t\t\t shared_expert_num %ld \n", _shared_expert_num);
        printf("\t\t\t\t shared_expert_gate_weight %p  %s \n", _shared_expert_gate_weight.get(), _shared_expert_gate_weight.get() ? _shared_expert_gate_weight->info().c_str() : "");
        printf("\t\t\t\t gate_weight %p  %s \n", _gate_weight.get(), _gate_weight.get() ? _gate_weight->info().c_str() : "");
        printf("\n");
        printf("\t\t\t\t _shared_expert %p  %s \n", _shared_expert.get(), _shared_expert.get() ? "_shared_expert" : "");
        if (_shared_expert) {
            _shared_expert->print_info();
        }
        printf("\n");
        printf("\t\t\t\t _experts size %ld   \n", _experts.size());
        for (auto expert : _experts) {
            expert->print_info();
            break;
        }
    }
};

template <typename Meta, typename Weights>
class AttentionTensor : public Qwen::BaseAttentionTensor {
public:
    AttentionTensor(Meta const *meta, Weights const *w, size_t ilayer, size_t idev, size_t ndev) {
        size_t nkvh = meta->nkvh;
        size_t nh = meta->nh;
        size_t dh = meta->dh;
        size_t d = meta->d;
        infiniDtype_t dt_mat = w->_dt_mat;
        infiniDtype_t dt_norm = w->_dt_norm;
        int transpose_linear_weights = w->_transpose_linear_weights;

        void *qkv_proj_weight_ptr = w->_layers[ilayer]._self_attn._qkv_proj_weight;
        void *qkv_proj_bias_ptr = w->_layers[ilayer]._self_attn._qkv_proj_bias;
        void *qk_norm_weight_ptr = w->_layers[ilayer]._self_attn._qk_norm_weight;
        void *o_proj_weight_ptr = w->_layers[ilayer]._self_attn._o_proj_weight;

        this->Init(nkvh, nh, dh, d, dt_mat, dt_norm, transpose_linear_weights, idev, ndev, qkv_proj_weight_ptr, qkv_proj_bias_ptr, qk_norm_weight_ptr, o_proj_weight_ptr);
    }

    void print_info() const {
        printf("\t\t\t AttentionTensor \n");
        printf("\t\t\t\t w_attn_qkv :: %p\t%s \n", w_attn_qkv.get(), w_attn_qkv->info().c_str());
        printf("\t\t\t\t b_attn_qkv ::  %p\t%s \n", b_attn_qkv.get(), b_attn_qkv.get() ? b_attn_qkv->info().c_str() : "");
        printf("\t\t\t\t w_attn_qk_norm ::  %p\t%s \n", w_attn_qk_norm.get(), w_attn_qk_norm.get() ? w_attn_qk_norm->info().c_str() : "");
        printf("\t\t\t\t w_attn_out :: %p\t%s \n", w_attn_out.get(), w_attn_out->info().c_str());
    }
};

template <typename AttentionTensor, typename FFNTensor, typename Meta, typename Weights>
class DecoderLayerTensor {
public:
    int ilayer;
    std::shared_ptr<Tensor> w_attn_norm;
    std::shared_ptr<Tensor> w_ffn_norm;
    std::shared_ptr<AttentionTensor> self_attn;
    std::shared_ptr<FFNTensor> ffn;

public:
    DecoderLayerTensor(Meta const *meta, Weights const *w, size_t ilayer, size_t idev, size_t ndev) {
        this->ilayer = ilayer;

        size_t d = meta->d;
        infiniDtype_t dt_norm = w->_dt_norm;
        void *att_norm_weight_ptr = w->_layers[ilayer]._input_layernorm_weight;
        void *ffn_norm_weight_ptr = w->_layers[ilayer]._post_attention_layernorm_weight;

        this->w_attn_norm = Qwen::getNorm(d, dt_norm, att_norm_weight_ptr);
        this->w_ffn_norm = Qwen::getNorm(d, dt_norm, ffn_norm_weight_ptr);

        this->self_attn = std::make_shared<AttentionTensor>(meta, w, ilayer, idev, ndev);
        this->ffn = std::make_shared<FFNTensor>(meta, w, ilayer, idev, ndev);
    }
    void print_info() const {
        printf("\n ");
        printf("\t\t DecoderLayerTensor %d \n ", ilayer);
        printf("\t\t\t w_attn_norm :: %p\t%s \n", w_attn_norm.get(), w_attn_norm->info().c_str());
        printf("\t\t\t w_ffn_norm :: %p\t%s \n", w_ffn_norm.get(), w_ffn_norm->info().c_str());
        printf("\n");
        printf("\t\t\t self_attn :: %p \n", self_attn.get());
        self_attn->print_info();
        printf("\n");
        printf("\t\t\t ffn :: %p \n", ffn.get());
        ffn->print_info();
    }
};

}; // namespace Qwen

#endif
