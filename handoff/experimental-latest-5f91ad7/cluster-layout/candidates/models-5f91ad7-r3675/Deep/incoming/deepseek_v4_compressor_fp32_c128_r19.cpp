#include "deepseek_v4_temporal_compressor.hpp"
#include "execution/submit.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace rt = infinilm::runtime;
namespace ex = infinilm::execution;
namespace {
struct Work {
    rt::Runtime &r;
    std::vector<rt::Tensor> held;
    ~Work() { try { r.syncStream(); } catch (...) { std::terminate(); } }
    rt::Tensor alloc(rt::Shape shape, rt::DataType type) {
        auto x = rt::Tensor::empty(r, shape, type); held.push_back(x); return x;
    }
    rt::Tensor ids(const std::vector<int64_t> &host) {
        auto x = alloc({host.size()}, rt::DataType::kInt64);
        r.memcpyH2D(x->data(), host.data(), host.size()*8, false); return x;
    }
};
}
void TemporalCompressor::check_owner(rt::Runtime &r) const {
    r.assertOwnerThread();
    if (&r != owner_ || r.isGraphRecording())
        throw std::invalid_argument("DeepSeek compressor Runtime owner/recording differs");
}
TemporalCompressor::TemporalCompressor(rt::Runtime &r, const nlohmann::json &cfg,
    size_t layer, size_t dim, size_t rows, size_t capacity)
    : owner_(&r), dim_(dim), rows_(rows), ratio_(cfg.at("compress_ratios").at(layer)),
      channels_(ratio_ == 4 ? 4 : 2), maximum_(cfg.at("max_position_embeddings")),
      capacity_(capacity), theta_(cfg.at("compress_rope_theta")),
      factor_(cfg.at("rope_scaling").at("factor")),
      beta_fast_(cfg.at("rope_scaling").at("beta_fast")),
      beta_slow_(cfg.at("rope_scaling").at("beta_slow")),
      original_(cfg.at("rope_scaling").at("original_max_position_embeddings")) {
    check_owner(r);
    double eps = cfg.at("rms_norm_eps");
    if (r.device().type()!=rt::Device::Type::kHygon || cfg.at("hidden_size")!=4096 ||
        !rows_ || rows_>128 || !maximum_ || (ratio_!=4 && ratio_!=128) ||
        (dim_!=512 && dim_!=128) || (dim_==128 && ratio_!=4) ||
        (dim_==512 && (!capacity_ || capacity_>maximum_/ratio_)) ||
        (dim_==128 && capacity_) || !std::isfinite(eps) || eps<=0 ||
        !std::isfinite(theta_) || theta_<=1 || !std::isfinite(factor_) || factor_<=0 ||
        !std::isfinite(beta_fast_) || beta_fast_<=0 || !std::isfinite(beta_slow_) || beta_slow_<=0)
        throw std::invalid_argument("DeepSeek real compressor configuration differs");
    try {
        size_t projection = channels_/2*dim_;
        wkv_ = rt::Tensor::empty_parameter(r,{projection,4096},rt::DataType::kBFloat16);
        wgate_ = rt::Tensor::empty_parameter(r,{projection,4096},rt::DataType::kBFloat16);
        // Keep original checkpoint geometry. C4 [4,2D] is NOT directly [8,D].
        ape_ = rt::Tensor::empty_parameter(r,{ratio_,projection},rt::DataType::kFloat32);
        bind_weight("wkv.weight",weights::Weight(wkv_));
        bind_weight("wgate.weight",weights::Weight(wgate_));
        bind_weight("ape",weights::Weight(ape_));
        norm_ = bind_child("norm",std::make_shared<layers::RMSNorm>(r,dim_,eps,rt::DataType::kBFloat16));
        packed_ape_ = rt::Tensor::empty(r,{ratio_==4?size_t(8):ratio_,dim_},rt::DataType::kFloat32);
        state_ = rt::Tensor::empty(r,{rows_*(ratio_==4?size_t(8):ratio_),channels_*dim_},rt::DataType::kBFloat16);
        lengths_.assign(rows_,0);
        r.setDeviceMemory(state_->data(),0,state_->nbytes());
        if(dim_==512) {
            size_t block=ratio_==4?64:2;
            cache_=std::make_unique<PagedLatentCache>(r,(rows_*capacity_+block-1)/block,block);
        }
        r.syncStream();
    } catch(...) { try {r.syncStream();}catch(...){std::terminate();} throw; }
}
void TemporalCompressor::process_weights_after_loading(rt::Runtime&r,bool) {
    check_owner(r); if(ready_)throw std::logic_error("DeepSeek compressor already loaded");
    if(ratio_==4) {
        // First D channels = previous-window Ca, second D = current-window Cb.
        auto layout=ape_->view({4,2,dim_})->permute({1,0,2});
        packed_ape_->view({2,4,dim_})->copy_from(r,layout);
    } else packed_ape_->copy_from(r,ape_);
    r.syncStream(); ready_=true;
}
void TemporalCompressor::rotate(rt::Runtime&r,rt::Tensor x,const std::vector<int64_t>&positions) const {
    Work work{r}; size_t n=positions.size(); std::vector<float>table(n*64);
    constexpr double pi2=6.283185307179586476925286766559;
    double low=0,high=0;
    if(original_ && factor_!=1) {
        auto correction=[&](double rotations){return 64*std::log(double(original_)/(rotations*pi2))/(2*std::log(theta_));};
        low=std::clamp(std::floor(correction(beta_fast_)),0.,63.);
        high=std::clamp(std::ceil(correction(beta_slow_)),0.,63.);
        if(low==high)high+=.001;
    }
    for(size_t t=0;t<n;++t)for(size_t i=0;i<32;++i) {
        double freq=1/std::pow(theta_,double(2*i)/64);
        if(original_ && factor_!=1) {
            double smooth=1-std::clamp((double(i)-low)/(high-low),0.,1.);
            freq=freq/factor_*(1-smooth)+freq*smooth;
        }
        double angle=positions[t]*freq;
        table[t*64+i]=float(std::cos(angle));table[t*64+32+i]=float(std::sin(angle));
    }
    auto cache=work.alloc({n,64},rt::DataType::kFloat32);
    r.memcpyH2D(cache->data(),table.data(),table.size()*4,false);
    std::vector<int64_t>rows(n);std::iota(rows.begin(),rows.end(),0);
    ex::submit<infini::ops::RotaryEmbedding>(r,ex::RecordingPolicy::HostOnly,
        infini::ops::RotaryEmbedding::DefaultConfig(r.device().type()),work.ids(rows),
        x->view({n,1,dim_}),std::optional<rt::Tensor>{},cache,int64_t(dim_),false,int64_t(dim_-64),false);
    r.syncStream();
}
TemporalCompressor::Batch TemporalCompressor::prepare(rt::Runtime&r,const rt::Tensor&hidden,
    const rt::Tensor&positions,const std::vector<Request>&requests) const {
    return prepare_impl(r,hidden,positions,requests,true);
}
TemporalCompressor::Batch TemporalCompressor::prepare_pooled(rt::Runtime&r,const rt::Tensor&hidden,
    const rt::Tensor&positions,const std::vector<Request>&requests) const {
    if(dim_!=128)throw std::invalid_argument("Only D128 indexer uses pooled-only preparation");
    return prepare_impl(r,hidden,positions,requests,false);
}
TemporalCompressor::Batch TemporalCompressor::prepare_impl(rt::Runtime&r,const rt::Tensor&hidden,
    const rt::Tensor&positions,const std::vector<Request>&requests,bool postprocess) const {
    check_owner(r);
    if(!ready_ || !hidden || hidden->device()!=r.device() || hidden->dtype()!=rt::DataType::kBFloat16 ||
        hidden->ndim()!=2 || !hidden->is_contiguous() || !hidden->size(0) || hidden->size(0)>4096 ||
        hidden->size(1)!=4096 || !positions || positions->device()!=r.device() ||
        positions->dtype()!=rt::DataType::kInt64 || !positions->is_contiguous() ||
        positions->shape()!=rt::Shape{hidden->size(0)} || requests.empty() || requests.size()>rows_)
        throw std::invalid_argument("DeepSeek compressor hidden/position contract differs");
    size_t tokens=hidden->size(0),total=0;
    std::vector<int64_t>pos(tokens),writes(tokens),extras(tokens),rotary_positions;
    r.memcpyD2H(pos.data(),positions->data(),tokens*8);
    std::set<size_t>seen;Batch batch;
    for(auto request:requests) {
        if(request.row>=rows_ || !seen.insert(request.row).second || !request.tokens ||
            request.past!=lengths_[request.row] || request.past>maximum_ ||
            request.tokens>maximum_-request.past || request.tokens>tokens-total ||
            (cache_ && (request.past+request.tokens)/ratio_>capacity_))
            throw std::invalid_argument("DeepSeek compressor row/past/count/capacity differs");
        for(size_t t=0;t<request.tokens;++t) {
            size_t i=total+t,p=request.past+t;
            if(pos[i]!=int64_t(p))throw std::invalid_argument("DeepSeek compressor logical position differs");
            writes[i]=ratio_==4?int64_t(request.row*2+(p/4)%2):int64_t(request.row);
            extras[i]=ratio_==4?int64_t(request.row*2+(p<4?0:p/4-1)%2):0;
            if((p+1)%ratio_==0) {
                batch.boundaries.push_back(int64_t(i));
                batch.slots.push_back(int64_t(request.row*capacity_+(p+1)/ratio_-1));
                // A compressed entry rotates at the FIRST token of its group.
                rotary_positions.push_back(int64_t(p/ratio_*ratio_));
            }
        }
        total+=request.tokens;
    }
    if(total!=tokens)throw std::invalid_argument("DeepSeek compressor packed token count differs");
    Work work{r};batch.owner=this;batch.revision=revision_;batch.requests=requests;
    batch.score=work.alloc({tokens,channels_*dim_},rt::DataType::kBFloat16);
    auto projection=[&](const rt::Tensor&weight,size_t start) {
        auto out=work.alloc({tokens,weight->size(0)},rt::DataType::kBFloat16);
        if(ratio_==128){
        // Diagnostic candidate: explicitly convert BF16 operands to exact F32
        // values, perform public F32 GEMM, then retain the original BF16 boundary.
        // This changes accumulation/backend selection; it is not a bit-equivalent
        // replacement until independently qualified at operator and model scope.
        auto hidden32=work.alloc(hidden->shape(),rt::DataType::kFloat32);
        auto weight32=work.alloc(weight->shape(),rt::DataType::kFloat32);
        auto out32=work.alloc(out->shape(),rt::DataType::kFloat32);
        ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),hidden,hidden32);
        ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),weight,weight32);
        ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),
            hidden32,weight32->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},
            std::optional<float>{0},std::optional<int>{},std::optional<int>{},out32);
        ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),out32,out);

        }else{
        ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),
            hidden,weight->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},
            std::optional<float>{0},std::optional<int>{},std::optional<int>{},out);
        }
        batch.score->narrow({{1,start,weight->size(0)}})->copy_from(r,out);
    };
    projection(wkv_,0);projection(wgate_,channels_/2*dim_);
    batch.next_state=work.alloc(state_->shape(),state_->dtype());batch.next_state->copy_from(r,state_);
    batch.pooled=work.alloc({tokens,dim_},rt::DataType::kBFloat16);
    auto status=work.alloc({1},rt::DataType::kInt32);
    ex::submit<infini::ops::DeepseekV4Compress>(r,ex::RecordingPolicy::HostOnly,
        infini::ops::DeepseekV4Compress::DefaultConfig(r.device().type()),batch.pooled,batch.score,
        packed_ape_,batch.next_state,positions,work.ids(writes),work.ids(extras),status,ratio_);
    int flag=0;r.memcpyD2H(&flag,status->data(),sizeof(flag));r.syncStream();
    if(flag)throw std::invalid_argument("DeepSeek real compressor validation status="+std::to_string(flag));
    if(postprocess && !batch.boundaries.empty()) {
        auto selected=work.alloc({batch.boundaries.size(),dim_},rt::DataType::kBFloat16);
        ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),
            batch.pooled,work.ids(batch.boundaries),int64_t{0},selected);
        batch.rotated=norm_->forward(r,selected);work.held.push_back(batch.rotated);
        rotate(r,batch.rotated,rotary_positions);
    }
    r.syncStream();return batch;
}
void TemporalCompressor::commit(rt::Runtime&r,Batch&batch) {
    check_owner(r);
    if(batch.owner!=this || batch.revision!=revision_ || !batch.next_state)
        throw std::invalid_argument("DeepSeek compressor stale/foreign transaction");
    // Cache performs finite/metadata checks before mutation. The old ring remains
    // authoritative until all normalized compressed entries have been stored.
    Work work{r};
    if(cache_ && !batch.boundaries.empty())cache_->store(r,work.ids(batch.slots),batch.rotated);
    state_=std::move(batch.next_state);
    for(auto request:batch.requests)lengths_[request.row]=request.past+request.tokens;
    ++revision_;batch.owner=nullptr;r.syncStream();
}
TemporalCompressor::Batch TemporalCompressor::forward(rt::Runtime&r,const rt::Tensor&x,
    const rt::Tensor&positions,const std::vector<Request>&requests) {
    auto batch=prepare(r,x,positions,requests);commit(r,batch);return batch;
}
rt::Tensor TemporalCompressor::gather(rt::Runtime&r,const std::vector<std::pair<size_t,size_t>>&entries)const {
    check_owner(r);
    if(!ready_ || !cache_ || entries.empty() || entries.size()>65536)
        throw std::invalid_argument("DeepSeek compressor cache selection differs");
    std::vector<int64_t>slots;slots.reserve(entries.size());
    for(auto entry:entries) {
        if(entry.first>=rows_ || entry.second>=lengths_[entry.first]/ratio_)
            throw std::invalid_argument("DeepSeek compressor selection outside valid prefix");
        slots.push_back(int64_t(entry.first*capacity_+entry.second));
    }
    Work work{r};return cache_->gather(r,work.ids(slots));
}
void TemporalCompressor::reset(rt::Runtime&r) {
    check_owner(r);if(cache_)cache_->reset(r);
    r.setDeviceMemory(state_->data(),0,state_->nbytes());r.syncStream();
    std::fill(lengths_.begin(),lengths_.end(),0);++revision_;
}
}
