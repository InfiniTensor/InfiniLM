#include "compressor.hpp"
#include "hyper_connection.hpp"
#include "model_config.hpp"
#include "native_ops.hpp"
#include "request_rows.hpp"
#include "rotary_frequencies.hpp"

#include "engine/deepseek_v4_metadata.hpp"
#include "engine/rank_barrier.hpp"
#include "engine/distributed/communication_group.hpp"
#include "execution/submit.hpp"
#include "global_state/forward_context.hpp"
#include "layers/embedding/embedding.hpp"
#include "models/models_registry.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {

using runtime::DataType;
using runtime::Runtime;
using runtime::Shape;
using runtime::Tensor;
using weights::Weight;
using ModelConfig = config::ModelConfig;
using Rank = engine::distributed::RankInfo;
using Linear = nn::Linear;
using ColumnLinear = nn::ColumnParallelLinear;
using RowLinear = nn::RowParallelLinear;
using RMSNorm = layers::RMSNorm;
using Quant = quantization::CompressedTensors;

struct V4Config {
    size_t hidden, vocab, heads, output_groups, q_rank, o_rank, experts, intermediate;
    size_t topk, hash_layers, index_heads, streams, iterations, max_positions;
    double rms_eps, hc_eps, route_scale, swiglu_limit;
    std::vector<size_t> ratios;
    std::shared_ptr<Quant> quant;

    V4Config(const ModelConfig &config, const Rank &rank)
        : hidden(config.get<size_t>("hidden_size")), vocab(config.get<size_t>("vocab_size")),
          heads(config.get<size_t>("num_attention_heads")), output_groups(config.get<size_t>("o_groups")),
          q_rank(config.get<size_t>("q_lora_rank")), o_rank(config.get<size_t>("o_lora_rank")),
          experts(config.get<size_t>("n_routed_experts")), intermediate(config.get<size_t>("moe_intermediate_size")),
          topk(config.get<size_t>("num_experts_per_tok")), hash_layers(config.get<size_t>("num_hash_layers")),
          index_heads(config.get<size_t>("index_n_heads")), streams(config.get<size_t>("hc_mult")),
          iterations(config.get<size_t>("hc_sinkhorn_iters")), max_positions(config.get<size_t>("max_position_embeddings")),
          rms_eps(config.get<double>("rms_norm_eps")), hc_eps(config.get<double>("hc_eps")),
          route_scale(config.get<double>("routed_scaling_factor")), swiglu_limit(config.get<double>("swiglu_limit")),
          ratios(config.get<std::vector<size_t>>("compress_ratios")) {
        const auto layers = config.get<size_t>("num_hidden_layers");
        const auto &json = config.get_config_json();
        if (!json.contains("quantization_config") || !json.at("quantization_config").is_object() ||
            json.at("quantization_config").value("quant_method", "") != std::string("compressed-tensors")) {
            throw std::invalid_argument("DeepSeek V4 mainline currently requires the channel INT8 W8A8 checkpoint");
        }
        if (rank.device.type() != runtime::Device::Type::kHygon || rank.tp_size != 8 || rank.pp_size != 1 ||
            rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size || !rank.comm ||
            config.get_dtype() != DataType::kBFloat16 || config.get<size_t>("head_dim") != 512 ||
            config.get<size_t>("qk_rope_head_dim") != 64 || config.get<size_t>("num_key_value_heads") != 1 ||
            config.get<size_t>("n_shared_experts") != 1 || config.get<size_t>("index_head_dim") != 128 ||
            config.get<size_t>("index_topk") != 512 || config.get<size_t>("sliding_window") != 128 ||
            config.get<std::string>("scoring_func") != "sqrtsoftplus" ||
            !config.get_or<bool>("norm_topk_prob", true) ||
            !heads || heads % 8 || output_groups != 8 || !intermediate || intermediate % 8 ||
            !hidden || !vocab || !q_rank || !o_rank || !experts || !topk || topk > experts ||
            !index_heads || streams != 4 || !iterations || !max_positions || !layers || ratios.size() < layers ||
            hash_layers > layers || !std::isfinite(swiglu_limit) || swiglu_limit <= 0) {
            throw std::invalid_argument("Unsupported DeepSeek V4 Flash W8A8 configuration");
        }
        ratios.resize(layers);
        quant = std::make_shared<Quant>(json.at("quantization_config"));
    }
};

template<class Op, class... Args>
void submit(Runtime &rt, const Args &...args) {
    execution::submit<Op>(rt, Op::DefaultConfig(rt.device().type()), args...);
}

Tensor add(Runtime &rt, const Tensor &a, const Tensor &b) {
    auto out = Tensor::empty(rt, a->shape(), a->dtype());
    submit<infini::ops::Add>(rt, a, b, out);
    return out;
}

Tensor cast(Runtime &rt, const Tensor &value, DataType dtype) {
    auto out = Tensor::empty(rt, value->shape(), dtype);
    submit<infini::ops::Cast>(rt, value, out);
    return out;
}

// The HCU custom reduction used by the reference accumulates rank-local
// BF16 contributions in FP32 and truncates only the final writeback.
Tensor reduce_tp(Runtime &rt, const Tensor &local, infinicclComm_t comm,
                 const std::vector<size_t> &query_offsets) {
    auto sum = cast(rt,local,DataType::kFloat32);
    for_each_request(query_offsets, [&](const RequestRows &rows) {
        auto part = rows.slice(sum);
        engine::distributed::allreduce_(rt, part, part, infinicclSum, comm);
    });
    auto output = Tensor::empty(rt,local->shape(),local->dtype());
    submit<infini::ops::DeepseekV4CastRz>(rt,sum,output);
    return output;
}

Tensor normalize_and_rotate(Runtime &rt, const Tensor &input, const Tensor &weight,
                            float epsilon, const Tensor &position, const Tensor &frequencies,
                            bool round_before_rope = false) {
    auto fp32 = cast(rt, input, DataType::kFloat32);
    auto normalized = Tensor::empty(rt, input->shape(), DataType::kFloat32);
    if (!round_before_rope && input->ndim()==2) {
        const Shape kv_shape{input->size(0),1,input->size(1)};
        submit<infini::ops::RmsNorm>(rt,fp32->view(kv_shape),weight,epsilon,normalized->view(kv_shape));
    } else {
        submit<infini::ops::RmsNorm>(rt,fp32,weight,epsilon,normalized);
    }
    if (round_before_rope) normalized = cast(rt, normalized, input->dtype());
    apply_rotary_tail(rt, normalized, position, frequencies);
    return normalized;
}

struct MixWeights {
    Tensor projection, scale, base;
};

struct BatchMetadata {
    Tensor position, swa_indices, swa_lengths, swa_slots, page_table;
    Tensor c4_slots, c4_positions, c4_lengths, c4_sparse_lengths, c4_sources, c4_state_slots;
    Tensor c128_slots, c128_positions, c128_indices, c128_lengths, c128_sources, c128_state_slots;
    std::vector<size_t> query_offsets;
    bool single_decode{};

    auto tensors() const {
        return std::array<const Tensor*,17>{&position,&swa_indices,&swa_lengths,&swa_slots,&page_table,
            &c4_slots,&c4_positions,&c4_lengths,&c4_sparse_lengths,&c4_sources,&c4_state_slots,
            &c128_slots,&c128_positions,&c128_indices,&c128_lengths,&c128_sources,&c128_state_slots};
    }
};

BatchMetadata make_batch(Runtime &rt, const InfinilmModel::Input &input) {
    const runtime::Device cpu{runtime::Device::Type::kCpu};
    auto host = [&](const std::optional<Tensor> &value) {
        if (!value || !*value) throw std::invalid_argument("DeepSeek V4 requires paged input metadata");
        return (*value)->device() == cpu ? *value : (*value)->to(rt, cpu);
    };
    auto device = [&](const Tensor &t) {
        auto out=Tensor::empty(rt,t->shape(),t->dtype());
        out->copy_from(rt,t);
        return out;
    };
    const auto positions = host(input.position_ids);
    const auto offsets = host(input.input_offsets);
    const auto metadata = engine::build_deepseek_v4_attention_metadata(
        host(input.block_tables), host(input.slot_mapping), positions, offsets, std::nullopt, kV4PageSize);
    const auto c4 = engine::build_deepseek_v4_compression_plan(metadata, positions, offsets, 4);
    const auto c128 = engine::build_deepseek_v4_compression_plan(metadata, positions, offsets, 128);
    const bool decode = positions->numel()==1 &&
        (positions->dtype()==DataType::kInt64
          ? *reinterpret_cast<const int64_t*>(positions->data())>0
          : *reinterpret_cast<const int32_t*>(positions->data())>0);
    const auto *row_offsets = reinterpret_cast<const int32_t*>(offsets->data());
    std::vector<size_t> query_offsets(row_offsets, row_offsets + offsets->numel());
    return {input.position_ids.value()->view({positions->numel()}),
            device(metadata.swa_indices), device(metadata.swa_topk_lengths), device(metadata.raw_out_loc),
            device(metadata.page_table), device(metadata.c4_out_loc), device(metadata.c4_positions),
            device(metadata.c4_topk_lengths_raw), device(metadata.c4_sparse_topk_lengths),
            device(c4.sources), device(c4.state_slots), device(metadata.c128_out_loc),
            device(metadata.c128_positions), device(metadata.c128_page_indices),
            device(metadata.c128_topk_lengths_clamp1), device(c128.sources), device(c128.state_slots),std::move(query_offsets),decode};
}

class Attention final : public weights::ModelWeights {
public:
    Attention(Runtime &rt, const V4Config &c, const Rank &rank, size_t ratio)
        : ratio_(ratio), heads_(c.heads / rank.tp_size), index_heads_(c.index_heads),
          eps_(c.rms_eps) {
        comm_ = rank.comm;
        const auto bf = DataType::kBFloat16;
        const auto rank_id = size_t(rank.tp_rank), ranks = size_t(rank.tp_size);
        sink_ = bind_weight("attn_sink", Weight(Tensor::empty_parameter(rt,{heads_},DataType::kFloat32),0,rank_id,ranks));
        wq_a_ = bind_child("wq_a",std::make_shared<Linear>(rt,c.hidden,c.q_rank,c.quant,false,bf));
        wkv_ = bind_child("wkv",std::make_shared<Linear>(rt,c.hidden,kV4HeadDim,c.quant,false,bf));
        wq_b_ = bind_child("wq_b",std::make_shared<ColumnLinear>(rt,c.q_rank,c.heads*kV4HeadDim,c.quant,false,bf,rank_id,ranks));
        wo_a_ = bind_child("wo_a",std::make_shared<ColumnLinear>(rt,heads_*kV4HeadDim,c.output_groups*c.o_rank,false,bf,rank_id,ranks));
        wo_b_ = bind_child("wo_b",std::make_shared<RowLinear>(rt,c.output_groups*c.o_rank,c.hidden,c.quant,false,bf,rank_id,ranks,rank.comm));
        q_norm_ = bind_child("q_norm",std::make_shared<RMSNorm>(rt,c.q_rank,c.rms_eps,bf));
        kv_norm_ = bind_child("kv_norm",std::make_shared<RMSNorm>(rt,kV4HeadDim,c.rms_eps,bf));
        q_ones_ = Tensor::empty(rt,{kV4HeadDim},DataType::kFloat32);
        std::vector<float> ones(kV4HeadDim,1.0f);
        rt.memcpyH2D(q_ones_->data(),ones.data(),q_ones_->nbytes(),false);
        if (ratio_) {
            compressor_ = bind_child("compressor",std::make_shared<Compressor>(rt,c.hidden,kV4HeadDim,ratio_,c.rms_eps));
        }
        if (ratio_ == 4) {
            index_compressor_ = bind_child("indexer.compressor",std::make_shared<Compressor>(rt,c.hidden,128,4,c.rms_eps));
            index_query_ = bind_child("indexer.wq_b",std::make_shared<Linear>(rt,c.q_rank,index_heads_*128,c.quant,false,bf));
            index_weights_ = bind_child("indexer.weights_proj",std::make_shared<Linear>(rt,c.hidden,index_heads_,false,bf));
        }
    }

    void process_weights_after_loading(Runtime &rt, bool) override {
        kv_norm_fp32_ = cast(rt,kv_norm_->weight(),DataType::kFloat32);
    }

    Tensor forward(Runtime &rt, Tensor x, const BatchMetadata &batch, const LayerPagedCache &cache,
                   const Tensor &frequencies) const {
        const auto rows = x->size(0);
        auto q_lora = q_norm_->forward(rt,wq_a_->forward(rt,x));
        auto q_raw = wq_b_->forward(rt,q_lora)->view({rows,heads_,kV4HeadDim});
        const auto eps = static_cast<float>(eps_);
        auto q = cast(rt, normalize_and_rotate(rt,q_raw,q_ones_,eps,batch.position,frequencies,true),
                      q_raw->dtype());
        auto kv = normalize_and_rotate(rt,wkv_->forward(rt,x),kv_norm_fp32_,eps,
                                       batch.position,frequencies);
        store_mla_rows(rt,kv,batch.swa_slots,cache.swa,kV4PageSize);
        Tensor indices, lengths;
        if (ratio_ == 4) {
            auto compressed = compressor_->forward(rt,x,batch.c4_sources,batch.c4_state_slots,
                                                  cache.compressor_state,batch.c4_positions,frequencies,batch.query_offsets);
            // Compressed NoPE values round to BF16 before FP8 cache quantization.
            store_mla_rows(rt,cast(rt,compressed,DataType::kBFloat16),
                           batch.c4_slots,cache.compressed,kV4PageSize/4);
            auto key = index_compressor_->forward(rt,x,batch.c4_sources,batch.c4_state_slots,
                                                 cache.indexer_state,batch.c4_positions,frequencies,batch.query_offsets);
            key = cast(rt,key,DataType::kBFloat16);
            auto query = index_query_->forward(rt,q_lora)->view({rows,index_heads_,128});
            apply_rotary_tail(rt,query,batch.position,frequencies);
            auto weights = project_requests(rt, batch.query_offsets, [&](const RequestRows &part) {
                auto input = part.slice(x);
                return index_weights_->forward(rt, input);
            });
            // A positive scalar common to all candidates does not change top-k.
            indices = Tensor::empty(rt,{rows,512},DataType::kInt32);
            submit<infini::ops::DeepseekV4C4Index>(rt,key,batch.c4_slots,query,weights,batch.page_table,
                                                  batch.c4_lengths,cache.indexer,indices);
            lengths = batch.c4_sparse_lengths;
        } else if (ratio_ == 128) {
            auto compressed = compressor_->forward(rt,x,batch.c128_sources,batch.c128_state_slots,
                                                  cache.compressor_state,batch.c128_positions,frequencies,batch.query_offsets);
            // Compressed NoPE values round to BF16 before FP8 cache quantization.
            store_mla_rows(rt,cast(rt,compressed,DataType::kBFloat16),
                           batch.c128_slots,cache.compressed,kV4PageSize/128);
            indices = batch.c128_indices;
            lengths = batch.c128_lengths;
        }
        // Match the request's standalone FlashMLA partitioning and GEMM
        // shape even when the scheduler packs multiple requests together.
        auto local = project_requests(rt, batch.query_offsets, [&](const RequestRows &part) {
            auto attended = sparse_mla(rt, part.slice(q), cache, part.slice(batch.swa_indices),
                                       part.slice(batch.swa_lengths), sink_, part.slice(indices),
                                       part.slice(lengths), 1.0 / std::sqrt(double(kV4HeadDim)));
            apply_rotary_tail(rt, attended, part.slice(batch.position), frequencies, true);
            auto flattened = attended->view({part.count, heads_ * kV4HeadDim});
            return wo_a_->forward(rt, flattened);
        });
        auto contribution = static_cast<const nn::BaseLinear&>(*wo_b_).forward(rt,local);
        return reduce_tp(rt, contribution, comm_, batch.query_offsets);
    }

private:
    infinicclComm_t comm_;
    size_t ratio_, heads_, index_heads_;
    double eps_;
    Tensor sink_, q_ones_, kv_norm_fp32_;
    std::shared_ptr<Linear> wq_a_, wkv_, index_query_, index_weights_;
    std::shared_ptr<ColumnLinear> wq_b_, wo_a_;
    std::shared_ptr<RowLinear> wo_b_;
    std::shared_ptr<RMSNorm> q_norm_, kv_norm_;
    std::shared_ptr<Compressor> compressor_, index_compressor_;
};

class FeedForward final : public weights::ModelWeights {
public:
    FeedForward(Runtime &rt, const V4Config &c, const Rank &rank, size_t layer)
        : topk_(c.topk), scale_(c.route_scale), limit_(c.swiglu_limit), comm_(rank.comm) {
        const auto bf = DataType::kBFloat16;
        gate_ = bind_child("gate",std::make_shared<Linear>(rt,c.hidden,c.experts,false,bf,DataType::kFloat32));
        if (layer < c.hash_layers) {
            hash_ = bind_weight("gate.tid2eid",Weight(rt,{c.vocab,c.topk},DataType::kInt64));
        } else {
            bias_ = bind_weight("gate.bias",Weight(rt,{c.experts},DataType::kFloat32));
        }
        experts_ = bind_child("experts",std::make_shared<W8A8ExpertWeights>(
            rt,c.experts,c.hidden,c.intermediate,rank.tp_rank,rank.tp_size));
        shared_one_ = Tensor::empty(rt,{1,1},DataType::kFloat32);
        shared_zero_ = Tensor::empty(rt,{1,1},DataType::kInt64);
        const float one=1.f;
        const int64_t zero=0;
        rt.memcpyH2D(shared_one_->data(),&one,sizeof(one),false);
        rt.memcpyH2D(shared_zero_->data(),&zero,sizeof(zero),false);
        shared_ = bind_child("shared_experts",std::make_shared<W8A8ExpertWeights>(
            rt,1,c.hidden,c.intermediate,rank.tp_rank,rank.tp_size,true));
    }

    Tensor forward(Runtime &rt, Tensor x, const Tensor &tokens,
                   const std::vector<size_t> &query_offsets) const {
        auto logits = project_requests(rt, query_offsets, [&](const RequestRows &part) {
            auto input = part.slice(x);
            return gate_->forward(rt, input);
        });
        using Opt=std::optional<Tensor>;
        const Shape routes{x->size(0),topk_};
        RoutedExperts route{Tensor::empty(rt,routes,DataType::kFloat32),
                            Tensor::empty(rt,routes,DataType::kInt64)};
        submit<infini::ops::DeepseekV4Route>(rt,logits,bias_ ? Opt{bias_} : Opt{},
            hash_ ? Opt{tokens} : Opt{},hash_ ? Opt{hash_} : Opt{},1.0,route.weights,route.indices);
        auto routed = run_routed_experts(rt,x,route,*experts_,limit_,scale_);
        auto ones = shared_one_, zero = shared_zero_;
        if (x->size(0)!=1) {
            ones = Tensor::empty(rt,{x->size(0),1},DataType::kFloat32);
            zero = Tensor::empty(rt,{x->size(0),1},DataType::kInt64);
            submit<infini::ops::Fill>(rt,ones,1.0,ones);
            submit<infini::ops::Fill>(rt,zero,0.0,zero);
        }
        auto shared = run_routed_experts(rt,x,{ones,zero},*shared_,limit_,std::nullopt);
        auto sum = add(rt,routed,shared);
        return reduce_tp(rt, sum, comm_, query_offsets);
    }

private:
    size_t topk_;
    double scale_, limit_;
    infinicclComm_t comm_;
    Tensor hash_, bias_, shared_one_, shared_zero_;
    std::shared_ptr<Linear> gate_;
    std::shared_ptr<W8A8ExpertWeights> experts_, shared_;
};

HyperConnectionMix run_mhc(Runtime &rt, const Tensor &x, const MixWeights &w,
                           double norm_eps, double mix_eps, size_t iterations,
                           const std::vector<size_t> &query_offsets) {
    const auto rows = x->size(0), streams = x->size(1);
    HyperConnectionMix out{Tensor::empty(rt, {rows, x->size(2)}, x->dtype()),
                           Tensor::empty(rt, {rows, streams}, DataType::kFloat32),
                           Tensor::empty(rt, {rows, streams, streams}, DataType::kFloat32)};
    // The Hygon operator selects a projection/reduction from its row count.
    for_each_request(query_offsets, [&](const RequestRows &part) {
        submit<infini::ops::DeepseekV4MhcPre>(
            rt, part.slice(x), w.projection, w.scale, w.base, norm_eps, mix_eps,
            static_cast<int64_t>(iterations), part.slice(out.activation),
            part.slice(out.post), part.slice(out.combination));
    });
    return out;
}

Tensor run_mhc_post(Runtime &rt,const Tensor &x,const Tensor &residual,const HyperConnectionMix &mix) {
    auto out=Tensor::empty(rt,residual->shape(),residual->dtype());
    submit<infini::ops::DeepseekV4MhcPost>(rt,x,residual,mix.post,mix.combination,out);
    return out;
}

class Decoder final : public weights::ModelWeights {
public:
    Decoder(Runtime &rt,const V4Config &c,const Rank &rank,size_t layer)
        : norm_eps_(c.rms_eps), mix_eps_(c.hc_eps), iterations_(c.iterations) {
        const size_t mix = (2+c.streams)*c.streams;
        const auto fp = DataType::kFloat32;
        auto bind_mix = [&](const std::string &prefix) {
            return MixWeights{bind_weight(prefix+"_fn",Weight(rt,{mix,c.streams*c.hidden},fp)),
                              bind_weight(prefix+"_scale",Weight(rt,{3},fp)),
                              bind_weight(prefix+"_base",Weight(rt,{mix},fp))};
        };
        attn_mix_ = bind_mix("hc_attn");
        ffn_mix_ = bind_mix("hc_ffn");
        attn_norm_ = bind_child("attn_norm",std::make_shared<RMSNorm>(rt,c.hidden,c.rms_eps,DataType::kBFloat16));
        ffn_norm_ = bind_child("ffn_norm",std::make_shared<RMSNorm>(rt,c.hidden,c.rms_eps,DataType::kBFloat16));
        attn_ = bind_child("attn",std::make_shared<Attention>(rt,c,rank,c.ratios.at(layer)));
        ffn_ = bind_child("ffn",std::make_shared<FeedForward>(rt,c,rank,layer));
    }

    Tensor forward(Runtime &rt, const Tensor &residual, const Tensor &tokens,
                   const BatchMetadata &batch, const LayerPagedCache &cache, const Tensor &frequencies) const {
        const auto attention = mix(rt,residual,attn_mix_,batch.query_offsets);
        const auto attended = attn_->forward(rt,attn_norm_->forward(rt,attention.activation),batch,cache,frequencies);
        const auto next = run_mhc_post(rt,attended,residual,attention);
        const auto feed_forward = mix(rt,next,ffn_mix_,batch.query_offsets);
        const auto experts = ffn_->forward(rt, ffn_norm_->forward(rt,feed_forward.activation),
                                           tokens, batch.query_offsets);
        return run_mhc_post(rt,experts,next,feed_forward);
    }

private:
    HyperConnectionMix mix(Runtime &rt, const Tensor &x, const MixWeights &w,
                           const std::vector<size_t> &query_offsets) const {
        return run_mhc(rt, x, w, norm_eps_, mix_eps_, iterations_, query_offsets);
    }
    double norm_eps_, mix_eps_;
    size_t iterations_;
    MixWeights attn_mix_, ffn_mix_;
    std::shared_ptr<RMSNorm> attn_norm_, ffn_norm_;
    std::shared_ptr<Attention> attn_;
    std::shared_ptr<FeedForward> ffn_;
};

class Backbone final : public weights::ModelWeights {
public:
    Backbone(Runtime &rt,const V4Config &c,const Rank &rank)
        : hidden_(c.hidden), streams_(c.streams), rms_eps_(c.rms_eps), hc_eps_(c.hc_eps) {
        embedding_ = bind_child("embed_tokens",std::make_shared<layers::Embedding>(
            rt,c.vocab,c.hidden,std::nullopt,DataType::kBFloat16));
        norm_ = bind_child("norm",std::make_shared<RMSNorm>(rt,c.hidden,c.rms_eps,DataType::kBFloat16));
        const auto fp = DataType::kFloat32;
        head_ = {bind_weight("hc_head_fn",Weight(rt,{c.streams,c.streams*c.hidden},fp)),
                 bind_weight("hc_head_scale",Weight(rt,{1},fp)),
                 bind_weight("hc_head_base",Weight(rt,{c.streams},fp))};
        for (size_t i=0;i<c.ratios.size();++i) {
            layers_.push_back(bind_child("layers."+std::to_string(i),std::make_shared<Decoder>(rt,c,rank,i)));
        }
    }

    Tensor forward(Runtime &rt, Tensor tokens, const BatchMetadata &batch,
                   const std::vector<LayerPagedCache> &caches,
                   const Tensor &swa_rope, const Tensor &compressed_rope) const {
        auto embedded = embedding_->forward(rt,tokens)->view({tokens->numel(),hidden_});
        auto expanded = embedded->as_strided({tokens->numel(),streams_,hidden_},
                                             {static_cast<runtime::Stride>(hidden_),0,1});
        auto residual = expanded->contiguous(rt);
        for (size_t i=0;i<layers_.size();++i) {
            residual = layers_[i]->forward(rt,residual,tokens,batch,caches[i],
                                           caches[i].compression_ratio ? compressed_rope : swa_rope);
        }
        return norm_->forward(rt,run_mhc(rt,residual,head_,rms_eps_,hc_eps_,0,batch.query_offsets).activation);
    }

private:
    size_t hidden_, streams_;
    double rms_eps_, hc_eps_;
    MixWeights head_;
    std::shared_ptr<layers::Embedding> embedding_;
    std::shared_ptr<RMSNorm> norm_;
    std::vector<std::shared_ptr<Decoder>> layers_;
};

class DeepseekV4ForCausalLM final : public InfinilmModel {
public:
    DeepseekV4ForCausalLM(Runtime &rt,std::shared_ptr<ModelConfig> config,const Rank &rank,
                         backends::AttentionBackend backend,const layers::moe::EPConfig &ep)
        : cfg_(*config,rank), comm_(rank.comm), sampling_rank_(rank.tp_rank == 0) {
        validate_runtime_construction(config,rt,rank);
        if (backend != backends::AttentionBackend::PAGED_ATTN || ep.backend != layers::moe::EPBackend::Disabled)
            throw std::invalid_argument("DeepSeek V4 requires paged attention and TP expert placement");
        model_config_ = std::move(config);
        backbone_ = bind_child("model",std::make_shared<Backbone>(rt,cfg_,rank));
        head_ = bind_child("lm_head",std::make_shared<ColumnLinear>(
            rt,cfg_.hidden,cfg_.vocab,false,DataType::kBFloat16,rank.tp_rank,rank.tp_size));
    }

    Output forward(const Input &input, global_state::ForwardContext &ctx) const override {
        auto &rt = ctx.require_runtime();
        if (caches_.empty() || !input.input_ids || !*input.input_ids || rt.isGraphRecording())
            throw std::invalid_argument("DeepSeek V4 requires initialized paged cache and eager input preparation");
        auto ids = (*input.input_ids)->view({(*input.input_ids)->numel()});
        auto batch = make_batch(rt,input);
        if (ctx.model_decode_barrier && batch.single_decode) {
            return decode(rt, std::move(ids), std::move(batch), *ctx.model_decode_barrier,
                          ctx.model_decode_greedy);
        }
        return {logits(rt,ids,batch)};
    }

    bool owns_paged_decode_graph() const override {
        return true;
    }

    void reset_cache(const cache::CacheConfig *requested,global_state::ForwardContext &ctx,
                     backends::AttentionBackend backend,const Rank &rank) override {
        auto &rt = ctx.require_runtime();
        cache::validate_cache_runtime(rt,rank);
        cache::validate_backend_cache_config(requested,backend);
        decode_plans_ = {};
        if (!requested) {
            caches_.clear();
            cache_config_.reset();
            swa_rope_.reset();
            compressed_rope_.reset();
            return;
        }
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig*>(requested);
        if (!paged || paged->block_size()!=kV4PageSize || !paged->num_blocks())
            throw std::invalid_argument("DeepSeek V4 requires 256-token cache pages");
        const auto rows = std::min(cfg_.max_positions,paged->num_blocks()*kV4PageSize);
        const auto &json = model_config_->get_config_json();
        auto swa = make_rotary_frequencies(rt,{64,rows,model_config_->get<double>("rope_theta")},true);
        const auto &scaling = json.at("rope_scaling");
        auto compressed = make_rotary_frequencies(rt,{64,rows,model_config_->get<double>("compress_rope_theta"),
            scaling.at("original_max_position_embeddings").get<size_t>(),scaling.at("factor").get<double>(),
            scaling.at("beta_fast").get<double>(),scaling.at("beta_slow").get<double>()},true);
        std::vector<LayerPagedCache> pending;
        pending.reserve(cfg_.ratios.size());
        for (auto ratio : cfg_.ratios)
            pending.push_back(allocate_layer_paged_cache(rt,paged->num_blocks(),paged->num_blocks(),ratio));
        caches_ = std::move(pending);
        swa_rope_ = std::move(swa);
        compressed_rope_ = std::move(compressed);
        cache_config_ = requested->unique_copy();
    }
private:
    struct DecodePlan {
        Tensor ids;
        BatchMetadata batch;
        Tensor logits;
        Tensor sampled_ids;
        bool greedy;
        std::shared_ptr<execution::graph::Graph> graph;
    };

    Tensor logits(Runtime &rt,const Tensor &ids,const BatchMetadata &batch) const {
        auto hidden=backbone_->forward(rt,ids,batch,caches_,swa_rope_,compressed_rope_);
        // A per-token projection retains the reference's M=1 accumulation boundary
        // while preserving the all-position logits requested by existing callers.
        auto output = Tensor::empty(rt,{ids->numel(),cfg_.vocab},hidden->dtype());
        for (size_t row=0;row<ids->numel();++row) {
            auto input = hidden->narrow({{0,row,1}});
            auto local = head_->forward(rt,input);
            engine::distributed::allgather_(rt,output->narrow({{0,row,1}}),local,comm_);
        }
        return output->view({1,ids->numel(),cfg_.vocab});
    }

    Output decode(Runtime &rt, Tensor ids, BatchMetadata batch, engine::RankBarrier &barrier,
                  bool greedy) const {
        auto &recent = decode_plans_.front();
        auto &previous = decode_plans_.back();
        const auto incoming = batch.tensors();
        auto matches = [&](const std::optional<DecodePlan> &plan) {
            if (!plan || plan->greedy != greedy || ids->shape() != plan->ids->shape() ||
                ids->dtype() != plan->ids->dtype()) {
                return false;
            }
            const auto held = plan->batch.tensors();
            for (size_t i = 0; i < held.size(); ++i) {
                if ((*held[i])->shape() != (*incoming[i])->shape() ||
                    (*held[i])->dtype() != (*incoming[i])->dtype()) {
                    return false;
                }
            }
            return true;
        };

        // Collectives must replay graphs from the same capture on every rank.
        // All ranks agree on the slot before moving it or issuing a capture.
        const unsigned local_slots = (matches(recent) ? 1u : 0u) | (matches(previous) ? 2u : 0u);
        const unsigned common_slots = barrier.common_decode_slots(local_slots);
        if (common_slots) {
            if (!(common_slots & 1u)) {
                recent.swap(previous);
            }
            recent->ids->copy_from(rt, ids);
            const auto held = recent->batch.tensors();
            for (size_t i = 0; i < held.size(); ++i) {
                Tensor dest = *held[i];
                dest->copy_from(rt, *incoming[i]);
            }
        } else {
            // Drop the least recently used graph before pinning new allocations.
            previous.reset();
            DecodePlan candidate{std::move(ids), std::move(batch), {}, {}, greedy, {}};
            rt.startGraphRecording(true);
            try {
                candidate.logits = logits(rt, candidate.ids, candidate.batch);
                if (greedy && sampling_rank_) {
                    candidate.sampled_ids = Tensor::empty(rt, {1}, DataType::kInt64);
                    execution::submit<infini::ops::Argmax>(
                        rt, execution::RecordingPolicy::DeviceOnly,
                        infini::ops::Argmax::DefaultConfig(rt.device().type()),
                        candidate.logits->view({cfg_.vocab}), std::optional<int64_t>{}, false,
                        candidate.sampled_ids->view({}));
                }
                // No rank may allocate or synchronously upload descriptor
                // metadata while a peer is capturing on DTK.
                barrier.wait();
                candidate.graph = rt.stopGraphRecording();
                barrier.wait();
            } catch (...) {
                rt.cancelGraphRecording();
                throw;
            }
            previous = std::move(recent);
            recent = std::move(candidate);
        }
        recent->graph->run();
        return {recent->logits, {}, recent->graph->execution_report(), recent->sampled_ids};
    }

    V4Config cfg_;
    std::shared_ptr<Backbone> backbone_;
    std::shared_ptr<ColumnLinear> head_;
    infinicclComm_t comm_;
    const bool sampling_rank_;
    std::vector<LayerPagedCache> caches_;
    Tensor swa_rope_, compressed_rope_;
    // These graphs hold pointers to this model instance's current KV cache.
    // The two most recent shapes cover both sides of adjacent page boundaries.
    mutable std::array<std::optional<DecodePlan>, 2> decode_plans_;
};

} // namespace

INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(deepseek_v4, DeepseekV4ForCausalLM,
                                           create_deepseek_v4_model_config);
} // namespace infinilm::models::deepseek_v4
