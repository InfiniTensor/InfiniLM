#include "qwen3vl_impl.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>

void createDeviceResource(Qwen3vlDeviceResource *rsrc, const Qwen3vlMeta *meta,
                          std::shared_ptr<Qwen3vlDeviceWeights> weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    RUN_INFINI(infinirtStreamSynchronize(weights->load_stream));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>();

    *rsrc = Qwen3vlDeviceResource{
        device,
        dev_id,
        handle,
        weights,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(Qwen3vlDeviceResource &res) {
    infinirtDeviceSynchronize();

    res.weights.reset();

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

inline std::shared_ptr<Tensor> get_custom_SinTable(const Qwen3vlMeta &meta, std::vector<std::vector<uint32_t>> &pos_ids ,uint32_t dim, size_t theta) {
    // pos_ids shape:[seq, dim/2] , pos ids acting on each dim
    auto unit = dsize(meta.dtype);
    auto half_dim = dim/2;
    size_t len = pos_ids.size();
    void *table = std::malloc(len * half_dim * unit);

    for (size_t i = 0; i <len; i++) {
        for (size_t j = 0; j < half_dim; j++) {
            float _cos = std::sin(
                static_cast<float>(pos_ids[i][j]) / std::pow(theta, static_cast<float>(j) / half_dim));
            if (meta.dtype == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dim + j] = f32_to_f16(_cos);
            } else if (meta.dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dim + j] = f32_to_bf16(_cos);
            } else if (meta.dtype == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dim + j] = _cos;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({len, half_dim});
    auto tensor = Tensor::weight(table, meta.dtype, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> get_custom_CosTable(const Qwen3vlMeta &meta, std::vector<std::vector<uint32_t>> &pos_ids ,uint32_t dim, size_t theta) {
    // pos_ids shape:[seq, dim/2] , pos ids acting on each dim
    auto unit = dsize(meta.dtype);
    auto half_dim = dim/2;
    size_t len = pos_ids.size();
    void *table = std::malloc(len * half_dim * unit);

    for (size_t i = 0; i <len; i++) {
        for (size_t j = 0; j < half_dim; j++) {
            float _cos = std::cos(
                static_cast<float>(pos_ids[i][j]) / std::pow(theta, static_cast<float>(j) / half_dim));
            if (meta.dtype == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dim + j] = f32_to_f16(_cos);
            } else if (meta.dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dim + j] = f32_to_bf16(_cos);
            } else if (meta.dtype == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dim + j] = _cos;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({len, half_dim});
    auto tensor = Tensor::weight(table, meta.dtype, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> fast_pos_embed_interpolate(const Qwen3vlMeta &meta, Qwen3vlDeviceResource &rsrc, 
                                                        uint32_t* grid_thw, uint32_t num_batch, uint32_t total_patches) {
    auto dtype = meta.dtype;                                        
    auto num_position_embeddings = meta.vis_meta.num_position_embeddings;
    auto hidden_size = meta.vis_meta.hidden_size;
    auto merge_size = meta.vis_meta.spatial_merge_size;
    auto num_grid_per_side = static_cast<uint32_t>(sqrt(num_position_embeddings));

    uint32_t total_pixels_offset = 0;
    std::shared_ptr<Tensor> patch_pos_embeds = Tensor::buffer(dtype,{total_patches, hidden_size},rsrc.memory_pool);
    auto pos_embed_weight = rsrc.weights->w_vis->pos_embed_weight;

    std::vector<std::shared_ptr<Tensor>> pos_embeds(4);
    for (uint32_t i = 0; i < num_batch; ++i) {
        uint32_t t = grid_thw[i * 3];
        uint32_t h = grid_thw[i * 3 + 1];
        uint32_t w = grid_thw[i * 3 + 2];
        auto weight_array = std::vector<uint16_t>(h*w*hidden_size);
        auto weight_tensor = Tensor::buffer(dtype,{h*w, hidden_size},rsrc.memory_pool);

        // 计算插值索引和权重
        std::vector<std::vector<uint32_t>> indices(4);
        std::vector<std::vector<float>> weights(4);

        auto linspace = [](float start, float end, uint32_t num_points) -> std::vector<float> {
            std::vector<float> res(num_points);
            for (uint32_t i = 0; i < num_points; ++i) {
                res[i] = start + (end - start) * i / (num_points - 1);
            }
            return res;
        };

        auto h_idxs = linspace(0, num_grid_per_side - 1, h);
        auto w_idxs = linspace(0, num_grid_per_side - 1, w);

        for (uint32_t ih = 0; ih < h; ++ih) {
            for (uint32_t iw = 0; iw < w; ++iw) {
                float h_idx_f = h_idxs[ih], w_idx_f = w_idxs[iw];
                uint32_t h_idx_floor = static_cast<uint32_t>(floor(h_idx_f)),
                         w_idx_floor = static_cast<uint32_t>(floor(w_idx_f));
                uint32_t h_idx_ceil = std::min(static_cast<uint32_t>(ceil(h_idx_f)), num_grid_per_side - 1),
                         w_idx_ceil = std::min(static_cast<uint32_t>(ceil(w_idx_f)), num_grid_per_side - 1);

                float dh = h_idx_f - h_idx_floor, dw = w_idx_f - w_idx_floor;

                indices[0].push_back((h_idx_floor * num_grid_per_side) + w_idx_floor);
                indices[1].push_back((h_idx_floor * num_grid_per_side) + w_idx_ceil);
                indices[2].push_back((h_idx_ceil * num_grid_per_side) + w_idx_floor);
                indices[3].push_back((h_idx_ceil * num_grid_per_side) + w_idx_ceil);

                weights[0].push_back((1 - dh) * (1 - dw));
                weights[1].push_back((1 - dh) * dw);
                weights[2].push_back(dh * (1 - dw));
                weights[3].push_back(dh * dw);
            }
        }

        // 查表并加权求和
        for (int j = 0; j < 4; ++j) {
            pos_embeds[j] = Tensor::buffer(dtype,{h*w, hidden_size},rsrc.memory_pool);
            // 使用索引和权重获取对应位置嵌入，并乘以权重
            for(size_t i = 0; i < h*w; i++){
                rearrange(pos_embeds[j]->slice(0,i,1),pos_embed_weight->slice(0,indices[j][i],1));
            }
            for(size_t i = 0; i < h*w; i++){
                uint16_t w_value = f32_to_bf16(weights[j][i]);
                for(size_t k=0; k < hidden_size; k++){
                    weight_array[i*hidden_size + k] = w_value;
                }
            }
            RUN_INFINI(infinirtMemcpyAsync(weight_tensor->data(), weight_array.data(), sizeof(uint16_t)*h*w*hidden_size,
                        INFINIRT_MEMCPY_H2D, rsrc.stream));
            mul(pos_embeds[j],pos_embeds[j],weight_tensor);
        }

        // 合并四个方向的结果
        auto patch_pos_embed = pos_embeds[0]; // [h*w, hidden_size]
        for (int j = 1; j < 4; ++j) {
            add(patch_pos_embed,patch_pos_embed, pos_embeds[j]);
        }

        // 对于视频帧数T>1的情况，重复patch_pos_embed T次
        if (t > 1) {
            auto temp_patch_pos_embed = Tensor::buffer(dtype,{t,h*w,hidden_size},rsrc.memory_pool);
            for(size_t i = 0; i < t; i++){
                rearrange(temp_patch_pos_embed->slice(0,i,1), patch_pos_embed);
            }
            patch_pos_embed = temp_patch_pos_embed;
        }
        printf("merge patch pos embed/n");
        fflush(stdout);
        patch_pos_embed = patch_pos_embed
                          ->view({t, h/merge_size, merge_size, w/merge_size, merge_size, hidden_size})
                          ->permute({0, 1, 3, 2, 4, 5})
                          ->view({t*h*w, hidden_size}); //可能因为内存不连续无法再view

        rearrange(patch_pos_embeds->slice(0,total_pixels_offset,t*h*w), patch_pos_embed);
        total_pixels_offset += t*h*w;
    }
    return patch_pos_embeds;
}

inline auto rot_pos_embed(const Qwen3vlMeta &meta, Qwen3vlDeviceResource &rsrc, uint32_t* grid_thw, uint32_t num_batch, uint32_t total_patches) {
    auto dtype = meta.dtype;
    auto hidden_size = meta.vis_meta.hidden_size;
    auto num_heads = meta.vis_meta.num_heads;
    auto head_dim = hidden_size / num_heads;
    auto merge_size = meta.vis_meta.spatial_merge_size;

    std::vector<std::vector<uint32_t>> pos_ids_table_y (
        total_patches,
        std::vector<uint32_t>(head_dim/4) 
    );
    std::vector<std::vector<uint32_t>> pos_ids_table_x (
        total_patches,
        std::vector<uint32_t>(head_dim/4) 
    );
    for (uint32_t b = 0; b < num_batch; ++b) {
        uint32_t offset = b * 3;
        uint32_t num_frames = grid_thw[offset + 0];
        uint32_t height     = grid_thw[offset + 1];
        uint32_t width      = grid_thw[offset + 2];

        uint32_t merged_h = height / merge_size;
        uint32_t merged_w = width / merge_size;

        // 遍历所有块和块内位置
        size_t patch_offset = 0;
        for (uint32_t bh = 0; bh < merged_h; ++bh) {
            for (uint32_t bw = 0; bw < merged_w; ++bw) {
                for (uint32_t ih = 0; ih < merge_size; ++ih) {
                    for (uint32_t iw = 0; iw < merge_size; ++iw) {
                        uint32_t row = bh * merge_size + ih;
                        uint32_t col = bw * merge_size + iw;
                        // 如果是多帧，重复 num_frames 次
                        for (uint32_t f = 0; f < num_frames; ++f) {
                            size_t dim_offset = 0;
                            for(;dim_offset<head_dim/4;dim_offset++){
                                pos_ids_table_y[patch_offset][dim_offset] = row;
                                pos_ids_table_x[patch_offset][dim_offset] = col;
                            }
                            patch_offset++;
                        }
                    }
                }
            }
        }
    }
    auto sin = Tensor::buffer(dtype,{total_patches,head_dim/2},rsrc.memory_pool);
    auto sin_y = get_custom_SinTable(meta,pos_ids_table_y,head_dim/2,10000);
    rearrange(sin->slice(1,0,head_dim/4),sin_y);
    auto sin_x = get_custom_SinTable(meta,pos_ids_table_x,head_dim/2,10000);
    rearrange(sin->slice(1,head_dim/4,head_dim/2),sin_y);
    auto cos = Tensor::buffer(dtype,{total_patches,head_dim/2},rsrc.memory_pool);
    auto cos_y = get_custom_CosTable(meta,pos_ids_table_y,head_dim/2,10000);
    rearrange(cos->slice(1,0,head_dim/4),cos_y);
    auto cos_x = get_custom_CosTable(meta,pos_ids_table_x,head_dim/2,10000);
    rearrange(cos->slice(1,head_dim/4,head_dim/2),cos_y);

    return std::pair{sin,cos};
}

void inferDeviceBatchVision(const Qwen3vlMeta &meta, Qwen3vlDeviceResource &rsrc,
                            uint32_t idev, uint32_t ndev, InferRequest &req) { 
    void *pixel_values = req.pixel_values;
    uint32_t total_patches = req.total_patches;
    uint32_t *image_grid_thw = req.image_grid_thw;
    uint32_t num_images = req.num_images;
    void *pixel_values_videos = req.pixel_values_videos;
    uint32_t total_patches_videos = req.total_patches_videos;
    //uint32_t *video_grid_thw = req.video_grid_thw;
    //uint32_t num_videos = req.num_videos;
    //uint32_t patch_features = req.patch_features;

    auto dtype = meta.dtype;
    auto d = meta.vis_meta.hidden_size;
    auto channels = meta.vis_meta.in_channels;
    auto patch_size = meta.vis_meta.patch_size;
    auto temporal_patch_size = meta.vis_meta.temporal_patch_size;
    //auto stream = rsrc.stream;
    auto weights = rsrc.weights;
    
    auto image_tensor = Tensor::weight(pixel_values, dtype, {total_patches, channels*temporal_patch_size*patch_size*patch_size});
    auto video_tensor = Tensor::weight(pixel_values_videos, dtype, {total_patches_videos, channels*temporal_patch_size*patch_size*patch_size});
    auto hidden_states = Tensor::buffer(dtype, {total_patches, d, 1, 1, 1}, rsrc.memory_pool);

    std::vector<size_t> pads = {0, 0, 0};
    std::vector<ptrdiff_t> strides = {static_cast<long>(temporal_patch_size), static_cast<long>(patch_size), static_cast<long>(patch_size)};
    std::vector<size_t> dilations = {1, 1, 1};
    conv(hidden_states, image_tensor, rsrc.weights->w_vis->patch_embed_weight, rsrc.weights->w_vis->patch_embed_bias,
          pads.data(), strides.data(), dilations.data(), 3);
    hidden_states = hidden_states->view({total_patches, d});

    auto pos_embeds = fast_pos_embed_interpolate(meta,rsrc,image_grid_thw,num_images,total_patches);
    add(hidden_states,hidden_states,pos_embeds);

    auto [sin, cos] = rot_pos_embed(meta,rsrc,image_grid_thw,num_images,total_patches);


}

void inferDeviceBatchText(const Qwen3vlMeta &meta, Qwen3vlDeviceResource &rsrc,
                          uint32_t idev, uint32_t ndev, InferRequest &req) {
    const uint32_t *tokens = req.tokens;
    uint32_t ntok = req.ntok;
    const uint32_t *req_lens = req.req_lens;
    uint32_t nreq = req.nreq;
    const uint32_t *req_pos = req.req_pos;
    struct Qwen3vlCache **caches = req.kv_caches;
    const float *temperature = req.temperature;
    const uint32_t *topk = req.topk;
    const float *topp = req.topp;
    uint32_t *output = req.output;
    void *last_logits = req.logits;

    assert(meta.text_meta.num_attention_heads % ndev == 0);
    assert(meta.text_meta.num_key_value_heads % ndev == 0);

    auto dtype = meta.dtype;
    auto nlayer = meta.text_meta.num_hidden_layers;
    size_t nh = meta.text_meta.num_attention_heads / size_t(ndev);
    size_t nkvh = meta.text_meta.num_key_value_heads / size_t(ndev);
    auto ngroup = nh / nkvh;
    auto dh = meta.text_meta.head_dim;
    auto d = meta.text_meta.hidden_size;
    auto di = meta.text_meta.intermediate_size / size_t(ndev);
    auto dvoc = meta.text_meta.vocab_size;
    float epsilon = meta.text_meta.rms_norm_eps;
    auto stream = rsrc.stream;
    auto weights = rsrc.weights;

    //Allocate buffers
    auto logits_in = Tensor::buffer(dtype, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dtype, {ntok, d}, rsrc.memory_pool);

    //所有请求的当前token
    auto qkv_buf = Tensor::buffer(dtype, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dtype, {ntok, nh * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dtype, {ntok, 2*di}, rsrc.memory_pool);

    auto prob_buf = Tensor::buffer(dtype, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
    auto q_buf = qkv_rope->slice(1, 0, nh);
    auto k_buf = qkv_rope->slice(1, nh, nkvh);

    //Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) { // req_len 本次query长度，req_pos 历史长度
            batch_pos_ids[req_start + i] = req_pos[req] + i;  //batch_pos_ids 展平后每个token的pos
        }
        req_start += req_lens[req];
    }
    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }

    //convert tokens to embeddings
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       weights->w_lang->in_embd->data(tokens[i] * d),
                                       dsize(dtype) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }
    
    auto qk_buf = Tensor::buffer(dtype, {nh * max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dtype, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dtype, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);

    //Compute
    for (uint32_t i = 0; i < nlayer; i++){
        // attn norm
        rmsnorm(logits_out,logits_in,weights->w_lang->layers[i].attn_norm,epsilon);
        // qkv_proj
        linear(qkv_buf,logits_out,weights->w_lang->layers[i].attn_qkv_proj,1.0,0.0,nullptr,nullptr);
        // qk_norm
        rmsnorm(q_buf,q_buf,weights->w_lang->layers[i].attn_q_norm,epsilon);
        rmsnorm(k_buf,k_buf,weights->w_lang->layers[i].attn_k_norm,epsilon);
        // rope 
        rope_v2(q_buf,q_buf,pos_ids_buf,weights->sin_table,weights->cos_table);
        rope_v2(k_buf,k_buf,pos_ids_buf,weights->sin_table,weights->cos_table);
        
        // 逐个req处理
        size_t token_offset = 0;
        for(uint32_t req=0; req < nreq; req++){
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            
            auto o = o_buf->slice(0,token_offset,seq_len)->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});// [nkvh, ngroup, seq_len, dh]
            auto q = qkv_rope->slice({{0,token_offset,seq_len},{1,0,nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});// [nkvh, ngroup, seq_len, dh]
            auto k = qkv_rope->slice({{0,token_offset,seq_len},{1,nh,nkvh}});// [ntok, nkvh, dh]
            auto v = qkv_rope->slice({{0,token_offset,seq_len},{1,nh+nkvh,nkvh}});// [ntok, nkvh, dh]

            // concat to cache 
            rearrange(caches[req]->k_rot[idev][i]->slice(0,past_len,seq_len),k);
            rearrange(caches[req]->v[idev][i]->slice(0,past_len,seq_len),v);

            //fill full_k full_v
            auto full_k_buff = caches[req]->k_rot[idev][i]->slice(0,0,total_len)->permute({1,2,0});// [nkvh, dh, total_len]
            auto full_v_buff = caches[req]->v[idev][i]->slice(0,0,total_len)->permute({1,0,2});// [nkvh, total_len, dh]

            //self-attn
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto attn_score_req = qk_buf->slice(0,0,nh*seq_len*total_len)->view({nkvh, ngroup*seq_len, total_len});
            // [nkvh, ngroup * seq_len, dh] @ [nkvh, dh, total_len] = [nkvh, ngroup * seq_len, total_len]
            linear(attn_score_req,rearrange_q_buf->slice(1, 0, ngroup * seq_len),full_k_buff,1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = attn_score_req->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax,qk_softmax);
            // [nkvh, ngroup * seq_len, total_len] @ [nkvh, total_len, dh] = [nkvh, ngroup * seq_len, dh]
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), attn_score_req, full_v_buff, 1.0, 0.0, nullptr, nullptr);
            //printf("rearrage o; layer[%d]\n",i);
            rearrange(o,attn_val_gemm->slice(2, 0, seq_len));
            token_offset += seq_len;
        }
        linear(logits_in, o_buf, weights->w_lang->layers[i].attn_o_proj, 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dtype,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // mlp norm
        rmsnorm(logits_out,logits_in,weights->w_lang->layers[i].mlp_norm,epsilon);
        // mlp gate_up
        linear(gate_up_buf,logits_out,weights->w_lang->layers[i].mlp_gate_up,1.0,0.0,nullptr,nullptr);
        // silu
        silu(gate_buf,gate_buf);
        mul(gate_buf,gate_buf,up_buf);
        // mlp down
        linear(logits_in,gate_buf,weights->w_lang->layers[i].mlp_down,1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dtype,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }
    // sample and output
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, weights->w_lang->out_norm, epsilon);
            auto last_logits_buf = Tensor::buffer(dtype, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, weights->w_lang->out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(), dsize(dtype) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        weights->w_lang->out_norm,
                        epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), weights->w_lang->out_embd, 1.0, 0.0, nullptr, nullptr);
            std::random_device _rd;
            std::mt19937 gen(_rd());
            token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, req, 1)->view_as({}, {}),
                             prob_buf->slice(0, req, 1)->view_as({dvoc}, {1}),
                             random_val, topp[req], topk[req], temperature[req]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = uint32_t(result_cpu[req]);
            }
        }
    }
}

void inferDeviceBatch(const Qwen3vlMeta &meta, Qwen3vlDeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev, InferState &state, InferRequest &req) {
    // infer vision + sync
    if (req.num_images > 0 || req.num_videos > 0){
        inferDeviceBatchVision(meta, rsrc, idev, ndev, req);

        std::unique_lock<std::mutex> lock(state.mtx_sync);
        state.sync_cnt--;
        if (state.sync_cnt == 0) {
            state.cv_sync.notify_all();
        } else {
            state.cv_sync.wait(lock, [&] {return state.sync_cnt == 0;});
        }
    }
    // infer text
    inferDeviceBatchText(meta, rsrc, idev, ndev, req);
}

__C void
inferBatchQwen3vl(struct Qwen3vlModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    void *pixel_values, uint32_t total_patches,
                    uint32_t *image_grid_thw, uint32_t num_images,
                    void *pixel_values_videos, uint32_t total_patches_videos,
                    uint32_t *video_grid_thw, uint32_t num_videos,
                    uint32_t patch_features,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    struct Qwen3vlCache **kv_caches,
                    const float *temperature, const uint32_t *topk, const float *topp,
                    uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.pixel_values = pixel_values;
    model->req.total_patches = total_patches;
    model->req.image_grid_thw = image_grid_thw;
    model->req.num_images = num_images;
    model->req.pixel_values_videos = pixel_values_videos;
    model->req.total_patches_videos = total_patches_videos;
    model->req.video_grid_thw = video_grid_thw;
    model->req.num_videos = num_videos;
    model->req.patch_features = patch_features;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;
    model->states[0].sync_cnt = model->dev_ids.size();

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
forwardBatchQwen3vl(struct Qwen3vlModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    void *pixel_values, uint32_t total_patches,
                    uint32_t *image_grid_thw, uint32_t num_images,
                    void *pixel_values_videos, uint32_t total_patches_videos,
                    uint32_t *video_grid_thw, uint32_t num_videos,
                    uint32_t patch_features,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    struct Qwen3vlCache **kv_caches,
                    void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.pixel_values = pixel_values;
    model->req.total_patches = total_patches;
    model->req.image_grid_thw = image_grid_thw;
    model->req.num_images = num_images;
    model->req.pixel_values_videos = pixel_values_videos;
    model->req.total_patches_videos = total_patches_videos;
    model->req.video_grid_thw = video_grid_thw;
    model->req.num_videos = num_videos;
    model->req.patch_features = patch_features;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;
    model->states[0].sync_cnt = model->dev_ids.size();

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

void launchDevice(const Qwen3vlMeta &meta, std::shared_ptr<Qwen3vlDeviceWeights> weights, Qwen3vlDeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, state, req);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}


Qwen3vlModel::Qwen3vlModel(const Qwen3vlMeta *_meta, const Qwen3vlWeights *weights) : meta(*_meta) {
    auto device_weights = weights->device_weights;
    int ndev = device_weights.size();
    device = device_weights[0]->device;
    dev_ids.resize(ndev);
    for (int i = 0; i < ndev; i++) {
        dev_ids[i] = device_weights[i]->dev_id;
    }
    dev_resources = std::vector<Qwen3vlDeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, std::cref(meta), device_weights[i], &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct Qwen3vlModel *
createQwen3vlModel(const Qwen3vlMeta *_meta,
                      const Qwen3vlWeights *weights) {
    Qwen3vlModel *model = new Qwen3vlModel(_meta, weights);
    return model;
}

__C void
destroyQwen3vlModel(struct Qwen3vlModel *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}
